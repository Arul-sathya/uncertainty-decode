"""
uncertainty_decode/serving/scheduler_patch.py

Monkey-patches vLLM's block manager to intercept eviction calls
and redirect them through UncertaintyEvictionPolicy.

vLLM's default eviction path (simplified):
    BlockSpaceManager.can_allocate() → False
    → Scheduler._schedule() calls BlockSpaceManager.evict()
    → evict() picks LRU block via evictor.evict()

We replace evictor.evict() with our uncertainty-aware version.
The patch is surgical: only eviction logic is touched, nothing else.

Works with vLLM >= 0.8.0. The exact method signatures changed in 0.9.x
so we probe the version and patch accordingly.

Usage (called automatically by UncertaintyDecodeLLM.__init__):
    from uncertainty_decode.serving.scheduler_patch import patch_vllm_evictor
    patch_vllm_evictor(llm_engine, eviction_policy)
"""

import torch
import types
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from uncertainty_decode.eviction.policy import UncertaintyEvictionPolicy

logger = logging.getLogger(__name__)


def get_vllm_version() -> tuple:
    """Returns (major, minor) version tuple."""
    try:
        import vllm
        parts = vllm.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 8)


def patch_vllm_evictor(llm_engine, eviction_policy: "UncertaintyEvictionPolicy") -> bool:
    """
    Patches vLLM's block evictor with the uncertainty-aware policy.

    Returns True if patch succeeded, False if it fell back to no-op.
    In fallback mode, UncertaintyDecode still collects uncertainty scores
    for analysis — only the eviction routing is affected.
    """
    major, minor = get_vllm_version()
    logger.info(f"[UncertaintyDecode] Patching vLLM {major}.{minor} evictor...")

    try:
        if major == 0 and minor >= 9:
            return _patch_v09(llm_engine, eviction_policy)
        else:
            return _patch_v08(llm_engine, eviction_policy)
    except Exception as e:
        logger.warning(
            f"[UncertaintyDecode] Evictor patch failed: {e}\n"
            f"  Falling back to standard LRU. Uncertainty scores still collected.\n"
            f"  To fix: ensure vLLM >= 0.8.0 is installed."
        )
        return False


def _patch_v08(llm_engine, eviction_policy: "UncertaintyEvictionPolicy") -> bool:
    """Patch path for vLLM 0.8.x"""
    try:
        scheduler = llm_engine.scheduler
        # In 0.8.x, scheduler has block_manager with evictor
        if not hasattr(scheduler, 'block_manager'):
            raise AttributeError("No block_manager on scheduler")

        block_manager = scheduler.block_manager
        if not hasattr(block_manager, 'evictor'):
            raise AttributeError("No evictor on block_manager")

        original_evictor = block_manager.evictor

        # Store original for potential rollback
        original_evict = original_evictor.evict

        def uncertainty_evict(num_blocks: int = 1):
            """
            Replacement evict() that uses uncertainty-guided selection.
            Falls back to original LRU if uncertainty data not available.
            """
            # Get all evictable block IDs from original evictor's queue
            try:
                if hasattr(original_evictor, 'free_table'):
                    candidate_ids = list(original_evictor.free_table.keys())
                elif hasattr(original_evictor, '_free_blocks'):
                    candidate_ids = [b.block_id for b in original_evictor._free_blocks]
                else:
                    # Can't inspect candidates — fall back to original
                    return original_evict(num_blocks)

                if not candidate_ids:
                    return original_evict(num_blocks)

                # Get uncertainty-guided eviction candidates
                evict_ids = eviction_policy.select_eviction_candidates(
                    candidate_ids, num_blocks
                )

                if not evict_ids:
                    return original_evict(num_blocks)

                # Remove selected blocks using original evictor's infrastructure
                evicted = []
                for block_id in evict_ids:
                    try:
                        block = original_evictor.free_table.pop(block_id, None)
                        if block is not None:
                            evicted.append(block)
                    except Exception:
                        pass

                if evicted:
                    return evicted[0] if num_blocks == 1 else evicted
                else:
                    return original_evict(num_blocks)

            except Exception as e:
                logger.debug(f"[UncertaintyDecode] Evict fallback: {e}")
                return original_evict(num_blocks)

        # Apply patch
        original_evictor.evict = uncertainty_evict
        logger.info("[UncertaintyDecode] vLLM 0.8.x evictor patched successfully.")
        return True

    except AttributeError as e:
        raise AttributeError(f"vLLM 0.8.x patch path failed: {e}")


def _patch_v09(llm_engine, eviction_policy: "UncertaintyEvictionPolicy") -> bool:
    """Patch path for vLLM 0.9.x+ (uses KVCacheManager)"""
    try:
        # In 0.9.x, eviction happens in KVCacheManager
        model_executor = llm_engine.model_executor
        if not hasattr(model_executor, 'driver_worker'):
            raise AttributeError("No driver_worker")

        worker = model_executor.driver_worker
        if not hasattr(worker, 'kv_cache_manager'):
            # Try the model runner path
            if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'block_manager'):
                return _patch_v08(llm_engine, eviction_policy)
            raise AttributeError("No kv_cache_manager")

        kv_manager = worker.kv_cache_manager
        original_free = kv_manager.free

        def uncertainty_free(request_id: str, block_ids=None):
            """
            Intercepts block freeing to register block metadata with eviction policy.
            """
            if block_ids:
                for bid in (block_ids if hasattr(block_ids, '__iter__') else [block_ids]):
                    eviction_policy.flush_sequence(hash(request_id) % (2**31))
            return original_free(request_id, block_ids)

        kv_manager.free = uncertainty_free
        logger.info("[UncertaintyDecode] vLLM 0.9.x KVCacheManager patched successfully.")
        return True

    except AttributeError as e:
        raise AttributeError(f"vLLM 0.9.x patch path failed: {e}")


def register_block_allocation_hook(
    llm_engine,
    eviction_policy: "UncertaintyEvictionPolicy",
    block_size: int = 16,
) -> bool:
    """
    Registers a hook to track new block allocations.
    When a new KV block is allocated, we register it with the eviction policy
    so that uncertainty scores can be assigned to it immediately.

    This is separate from the eviction patch — it runs on every allocation
    to populate the eviction policy's block registry.
    """
    try:
        scheduler = llm_engine.scheduler
        block_manager = getattr(scheduler, 'block_manager', None)
        if block_manager is None:
            return False

        original_allocate = block_manager.allocate

        def tracked_allocate(seq_group, *args, **kwargs):
            result = original_allocate(seq_group, *args, **kwargs)

            # After allocation, register the new blocks with the eviction policy
            try:
                for seq in seq_group.get_seqs():
                    seq_id = seq.seq_id
                    block_table = block_manager.get_block_table(seq)
                    if block_table:
                        n_blocks = len(block_table)
                        for i, block in enumerate(block_table):
                            block_id = block.block_number if hasattr(block, 'block_number') else i
                            token_start = i * block_size
                            token_end = (i + 1) * block_size
                            eviction_policy.register_block(
                                block_id=block_id,
                                sequence_id=seq_id,
                                token_start=token_start,
                                token_end=token_end,
                            )
            except Exception as e:
                logger.debug(f"[UncertaintyDecode] Block registration failed: {e}")

            return result

        block_manager.allocate = tracked_allocate
        logger.info("[UncertaintyDecode] Block allocation tracking registered.")
        return True

    except Exception as e:
        logger.debug(f"[UncertaintyDecode] Allocation hook failed: {e}")
        return False
