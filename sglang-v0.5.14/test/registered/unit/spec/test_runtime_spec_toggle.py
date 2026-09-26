from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.io_struct import ToggleSpeculativeDecodingReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.eagle_worker_v2 import (
    EAGLEWorkerV2,
    _generate_nonsd_cuda_graph_batch_sizes,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_worker():
    worker = object.__new__(EAGLEWorkerV2)
    worker._runtime_toggle_supported = True
    worker.speculative_decoding_enabled = True
    worker._enabled_runtime_state = "startup-sd-state"
    worker._target_only_attn_backend = "target-only-attn"
    worker._target_only_graph_runner = "target-only-graph"
    worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE3
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            attn_backend="sd-attn",
            decode_cuda_graph_runner="sd-graph",
        )
    )
    worker._capture_runtime_state = MagicMock(return_value="latest-sd-state")
    worker.apply_runtime_state = MagicMock()
    return worker


def test_nonsd_cuda_graph_buckets_cover_one_through_128():
    capture_bs = _generate_nonsd_cuda_graph_batch_sizes()

    assert capture_bs == [1, 2, 4, 8, 16, 32, 64, 128]
    assert max(capture_bs) == 128
    assert all(
        any(bucket >= raw_bs for bucket in capture_bs)
        for raw_bs in range(1, 129)
    )


def test_nonsd_cuda_graph_buckets_disable_padding_captures_every_size():
    assert _generate_nonsd_cuda_graph_batch_sizes(disable_padding=True) == list(
        range(1, 129)
    )


def test_nonsd_metrics_interval_does_not_divide_by_zero():
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.scheduler = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        server_args=SimpleNamespace(
            speculative_num_draft_tokens=6,
            speculative_num_steps=5,
        ),
    )
    reporter.spec_num_accept_tokens = 0
    reporter.spec_num_forward_ct = 0
    reporter.spec_total_num_accept_tokens = 10
    reporter.spec_total_num_forward_ct = 4

    assert reporter._consume_spec_interval_stats() == (0.0, 0.0, False)
    assert reporter.spec_total_num_accept_tokens == 10
    assert reporter.spec_total_num_forward_ct == 4


def test_runtime_metrics_interval_consumes_only_spec_samples():
    reporter = object.__new__(SchedulerMetricsReporter)
    reporter.scheduler = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        server_args=SimpleNamespace(
            speculative_num_draft_tokens=6,
            speculative_num_steps=5,
        ),
    )
    reporter.spec_num_accept_tokens = 12
    reporter.spec_num_forward_ct = 4
    reporter.spec_total_num_accept_tokens = 0
    reporter.spec_total_num_forward_ct = 0

    assert reporter._consume_spec_interval_stats() == (3.0, 0.4, True)
    assert reporter.spec_num_accept_tokens == 0
    assert reporter.spec_num_forward_ct == 0
    assert reporter.spec_total_num_accept_tokens == 12
    assert reporter.spec_total_num_forward_ct == 4


def test_runtime_toggle_restores_latest_enabled_state():
    worker = _make_worker()

    assert worker.toggle_speculative_decoding(False) == (
        True,
        False,
        "Speculative decoding disabled.",
    )
    worker._capture_runtime_state.assert_called_once_with()
    worker.apply_runtime_state.assert_not_called()
    assert worker._enabled_runtime_state == "latest-sd-state"
    assert worker.speculative_decoding_enabled is False
    assert worker._target_worker.model_runner.spec_algorithm.is_none()
    assert worker._target_worker.model_runner.attn_backend == "target-only-attn"
    assert (
        worker._target_worker.model_runner.decode_cuda_graph_runner
        == "target-only-graph"
    )

    assert worker.toggle_speculative_decoding(True) == (
        True,
        True,
        "Speculative decoding enabled.",
    )
    worker.apply_runtime_state.assert_called_once_with("latest-sd-state")
    assert worker.speculative_decoding_enabled is True


def test_runtime_toggle_is_idempotent():
    worker = _make_worker()

    assert worker.toggle_speculative_decoding(True) == (
        True,
        True,
        "Speculative decoding state is unchanged.",
    )
    worker._capture_runtime_state.assert_not_called()
    worker.apply_runtime_state.assert_not_called()


def test_batch_uses_effective_non_spec_algorithm_when_runtime_disabled():
    batch = ScheduleBatch(
        reqs=[],
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        speculative_decoding_enabled=False,
    )

    assert batch.use_speculative_decoding() is False
    assert batch.effective_spec_algorithm().is_none()


def test_scheduler_reports_unsupported_algorithm_without_mutation():
    scheduler = object.__new__(Scheduler)
    scheduler.draft_worker = SimpleNamespace()
    scheduler.server_args = SimpleNamespace(speculative_algorithm="NGRAM")
    scheduler.forward_ct = 7

    output = scheduler.toggle_speculative_decoding(
        ToggleSpeculativeDecodingReqInput(enabled=False)
    )

    assert output.success is False
    assert output.enabled is True
    assert "not supported" in output.message


def test_scheduler_applies_toggle_at_control_request_boundary():
    toggle = MagicMock()
    scheduler = object.__new__(Scheduler)
    scheduler.draft_worker = SimpleNamespace(
        toggle_speculative_decoding=toggle,
        _runtime_toggle_supported=True,
    )
    scheduler.server_args = SimpleNamespace(speculative_algorithm="EAGLE3")
    scheduler.forward_ct = 12
    scheduler._runtime_spec_decoding_enabled = True
    scheduler._runtime_spec_decoding_requested = True
    scheduler._runtime_spec_recovery_armed = False
    scheduler._runtime_spec_recovery_inflight = False

    output = scheduler.toggle_speculative_decoding(
        ToggleSpeculativeDecodingReqInput(enabled=False)
    )

    toggle.assert_not_called()
    assert output.success is True
    assert output.enabled is True
    assert scheduler._runtime_spec_transition_pending()


def test_scheduler_disable_materializes_spec_lengths_before_target_decode():
    scheduler = object.__new__(Scheduler)
    scheduler._runtime_spec_decoding_enabled = True
    scheduler._runtime_spec_decoding_requested = False
    scheduler._runtime_spec_recovery_armed = False
    scheduler._runtime_spec_recovery_inflight = False
    scheduler.chunked_req = None
    scheduler.enable_overlap = True
    scheduler.forward_ct = 3
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False,
        use_speculative_decoding=lambda: True,
        speculative_decoding_enabled=True,
    )
    scheduler.future_map = SimpleNamespace(resolve_seq_lens_cpu=MagicMock())
    scheduler.draft_worker = SimpleNamespace(
        toggle_speculative_decoding=MagicMock(
            return_value=(True, False, "disabled")
        )
    )

    scheduler._apply_runtime_spec_transition()

    scheduler.future_map.resolve_seq_lens_cpu.assert_called_once_with(
        scheduler.running_batch
    )
    assert scheduler._runtime_spec_decoding_enabled is False
    assert scheduler.running_batch.speculative_decoding_enabled is False


def test_scheduler_enable_arms_one_iteration_recovery_for_active_requests():
    scheduler = object.__new__(Scheduler)
    scheduler._runtime_spec_decoding_enabled = False
    scheduler._runtime_spec_decoding_requested = True
    scheduler._runtime_spec_recovery_armed = False
    scheduler._runtime_spec_recovery_inflight = False
    scheduler.chunked_req = None
    scheduler.enable_overlap = True
    scheduler.forward_ct = 4
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False,
        batch_is_full=True,
        speculative_decoding_enabled=False,
    )
    scheduler.draft_worker = SimpleNamespace(
        toggle_speculative_decoding=MagicMock(),
        prepare_speculative_recovery=MagicMock(),
    )

    scheduler._apply_runtime_spec_transition()

    scheduler.draft_worker.toggle_speculative_decoding.assert_not_called()
    assert scheduler._runtime_spec_decoding_enabled is False
    assert scheduler._runtime_spec_transition_pending()
    assert scheduler._runtime_spec_recovery_armed
    scheduler.draft_worker.prepare_speculative_recovery.assert_called_once_with()


def test_scheduler_enable_waits_for_chunked_prefill_before_recovery():
    scheduler = object.__new__(Scheduler)
    scheduler._runtime_spec_decoding_enabled = False
    scheduler._runtime_spec_decoding_requested = True
    scheduler._runtime_spec_recovery_armed = False
    scheduler._runtime_spec_recovery_inflight = False
    scheduler.chunked_req = object()
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: False)
    scheduler.draft_worker = SimpleNamespace(
        toggle_speculative_decoding=MagicMock(),
        prepare_speculative_recovery=MagicMock(),
    )

    scheduler._apply_runtime_spec_transition()

    assert scheduler._runtime_spec_transition_pending()
    assert not scheduler._runtime_spec_recovery_armed
    scheduler.draft_worker.prepare_speculative_recovery.assert_not_called()
    scheduler.draft_worker.toggle_speculative_decoding.assert_not_called()

    scheduler.chunked_req = None
    scheduler._apply_runtime_spec_transition()

    assert scheduler._runtime_spec_recovery_armed
    scheduler.draft_worker.prepare_speculative_recovery.assert_called_once_with()


def test_scheduler_enable_switches_after_recovery_iteration():
    scheduler = object.__new__(Scheduler)
    scheduler._runtime_spec_decoding_enabled = False
    scheduler._runtime_spec_decoding_requested = True
    scheduler._runtime_spec_recovery_armed = False
    scheduler._runtime_spec_recovery_inflight = True
    scheduler.chunked_req = None
    scheduler.enable_overlap = True
    scheduler.forward_ct = 5
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False,
        speculative_decoding_enabled=False,
    )
    scheduler.draft_worker = SimpleNamespace(
        toggle_speculative_decoding=MagicMock(
            return_value=(True, True, "enabled")
        )
    )

    scheduler._apply_runtime_spec_transition()

    scheduler.draft_worker.toggle_speculative_decoding.assert_called_once_with(
        True
    )
    assert scheduler._runtime_spec_decoding_enabled is True
    assert not scheduler._runtime_spec_transition_pending()


def test_scheduler_recovery_marker_is_consumed_after_successful_iteration():
    scheduler = object.__new__(Scheduler)
    scheduler._runtime_spec_recovery_armed = True
    scheduler._runtime_spec_recovery_inflight = False
    batch = SimpleNamespace(speculative_recovery_iteration=True)

    scheduler._complete_runtime_spec_recovery_iteration(batch)

    assert batch.speculative_recovery_iteration is False
    assert scheduler._runtime_spec_recovery_armed is False
    assert scheduler._runtime_spec_recovery_inflight is True


def test_scheduler_discards_mixed_prefill_spec_info_when_disabling():
    scheduler = object.__new__(Scheduler)
    scheduler.draft_worker = SimpleNamespace(_runtime_toggle_supported=True)
    scheduler._runtime_spec_decoding_enabled = True
    scheduler._runtime_spec_decoding_requested = False
    scheduler._runtime_spec_recovery_inflight = False
    scheduler.running_batch = SimpleNamespace(spec_info="running-sd-state")
    scheduler.last_batch = SimpleNamespace(spec_info=None)

    scheduler._discard_stale_runtime_spec_info_before_merge()

    assert scheduler.running_batch.spec_info is None
    assert scheduler.last_batch.spec_info is None


def test_scheduler_preserves_rebuilt_spec_info_after_recovery():
    scheduler = object.__new__(Scheduler)
    scheduler.draft_worker = SimpleNamespace(_runtime_toggle_supported=True)
    scheduler._runtime_spec_decoding_enabled = False
    scheduler._runtime_spec_decoding_requested = True
    scheduler._runtime_spec_recovery_inflight = True
    scheduler.running_batch = SimpleNamespace(spec_info="fresh-recovery-state")
    scheduler.last_batch = SimpleNamespace(spec_info="fresh-recovery-state")

    scheduler._discard_stale_runtime_spec_info_before_merge()

    assert scheduler.running_batch.spec_info == "fresh-recovery-state"
    assert scheduler.last_batch.spec_info == "fresh-recovery-state"


def test_disabled_iteration_does_not_update_adaptive_controller():
    processor = object.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(processor, "model_worker", MagicMock())
    result = GenerationBatchResult(
        next_token_ids=torch.empty(0, dtype=torch.int64),
        accept_lens=torch.empty(0, dtype=torch.int32),
        speculative_num_draft_tokens=1,
        speculative_decoding_enabled=False,
    )
    batch = SimpleNamespace(reqs=[])

    assert processor._resolve_spec_v2_tokens(result, batch) == []
    processor.model_worker.on_verify_complete_cpu.assert_not_called()
