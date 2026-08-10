# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the FULL-cudagraph DBO descriptor/dispatch logic.

These exercise `_init_candidates`/`_is_compatible`/`dispatch` directly rather
than through a real `VllmConfig` and GPU forward pass -- that combination is
covered by GPU-gated capture/replay tests, but the descriptor-matching logic
itself is plain Python and worth pinning without needing a GPU to do it.
"""

from types import SimpleNamespace

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    _is_compatible,
)
from vllm.v1.worker.ubatch_utils import is_last_ubatch_empty


def _make_manager(
    capture_sizes: list[int],
    num_ubatches: int | None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL,
) -> CudaGraphManager:
    """Build a CudaGraphManager without going through __init__'s VllmConfig
    machinery. `_init_candidates` only reads the attributes stubbed here."""
    mgr = object.__new__(CudaGraphManager)
    mgr.vllm_config = SimpleNamespace(speculative_config=None)
    mgr.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=capture_sizes,
        max_cudagraph_capture_size=max(capture_sizes) if capture_sizes else 0,
    )
    mgr.cudagraph_mode = cudagraph_mode
    mgr.decode_query_len = 1
    mgr.max_num_reqs = 64
    mgr.lora_capture_cases = [0]
    mgr.dp_size = 2
    mgr.tp_size = 1
    mgr.ubatch_runner = (
        SimpleNamespace(num_ubatches=num_ubatches) if num_ubatches else None
    )
    mgr._candidates = {}
    mgr._capture_descs = {}
    mgr._init_candidates()
    return mgr


def test_is_compatible_requires_matching_num_ubatches():
    """A graph captured for 2 microbatches can't serve a 1-way (or 3-way)
    split, and vice versa -- num_ubatches has to be an exact match, not just
    a >= comparison like num_tokens/num_reqs get."""
    desc_ubatch = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=256, num_reqs=64, num_ubatches=2
    )
    desc_plain = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=256, num_reqs=64, num_ubatches=1
    )

    assert _is_compatible(
        desc_ubatch,
        num_reqs=64,
        num_tokens=256,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=2,
    )
    assert not _is_compatible(
        desc_ubatch,
        num_reqs=64,
        num_tokens=256,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=1,
    )
    assert not _is_compatible(
        desc_plain,
        num_reqs=64,
        num_tokens=256,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=2,
    )


def test_is_last_ubatch_empty_gates_degenerate_capture_sizes():
    # 2 real tokens padded to 4, split 2 ways: the second microbatch is
    # entirely inside the padding region.
    assert is_last_ubatch_empty(2, 4, 2)
    # 3 real tokens padded to 4, split 2 ways: the second microbatch still
    # holds one real token.
    assert not is_last_ubatch_empty(3, 4, 2)


def test_init_candidates_generates_dbo_descriptors_for_full_mode():
    mgr = _make_manager(capture_sizes=[64, 128, 256], num_ubatches=2)

    full_descs = mgr._capture_descs.get(CUDAGraphMode.FULL, [])
    ubatch_descs = [d for d in full_descs if d.num_ubatches == 2]
    plain_descs = [d for d in full_descs if d.num_ubatches == 1]

    assert {d.num_tokens for d in ubatch_descs} == {64, 128, 256}
    assert {d.num_tokens for d in plain_descs} == {64, 128, 256}
    # DBO descriptors use the same request-padding rule as the plain FULL
    # sweep they're generated alongside.
    for d in ubatch_descs:
        assert d.num_reqs == min(d.num_tokens, mgr.max_num_reqs)


def test_init_candidates_skips_no_ubatch_runner():
    """No DBO configured: the manager never generates num_ubatches>1
    descriptors, so the capture loop can't waste time/memory on them."""
    mgr = _make_manager(capture_sizes=[64, 128], num_ubatches=None)
    full_descs = mgr._capture_descs.get(CUDAGraphMode.FULL, [])
    assert all(d.num_ubatches == 1 for d in full_descs)


def test_dispatch_routes_to_captured_dbo_graph():
    mgr = _make_manager(capture_sizes=[64, 128, 256], num_ubatches=2)
    mgr._graphs_captured = True

    desc = mgr.dispatch(
        num_reqs=64,
        num_tokens=128,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=2,
    )
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_ubatches == 2
    assert desc.num_tokens == 128


def test_dispatch_falls_back_to_eager_when_no_dbo_graph_captured():
    """dispatch()'s generic fallback (no matching candidate -> NONE) already
    covers the "requested a split no graph was captured for" case, so a
    microbatched step never crashes for lack of a captured graph -- it just
    runs eager for that shape."""
    mgr = _make_manager(capture_sizes=[64, 128, 256], num_ubatches=2)
    mgr._graphs_captured = True

    desc = mgr.dispatch(
        num_reqs=64,
        num_tokens=128,
        uniform_token_count=None,
        num_active_loras=0,
        num_ubatches=3,  # no graph was captured with 3 microbatches
    )
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_ubatches == 3
