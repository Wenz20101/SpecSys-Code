# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Set as AbstractSet
from dataclasses import replace
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.lora.utils import get_captured_lora_counts

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False
        self.specialize_lora_count = (
            self.vllm_config.lora_config.specialize_active_lora
            if self.vllm_config.lora_config is not None
            else False
        )
        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE

    @staticmethod
    def _compute_bs_to_padded_graph_size_for_capture_sizes(
        capture_sizes: list[int],
    ) -> list[int]:
        """Pre-compute the mapping from batch size to padded graph size."""
        max_size = capture_sizes[-1]
        bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        for end, start in zip(
            capture_sizes + [max_size + 1],
            [0] + capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    bs_to_padded_graph_size[bs] = start
                else:
                    bs_to_padded_graph_size[bs] = end
        return bs_to_padded_graph_size

    def _compute_bs_to_padded_graph_size(self) -> None:
        """Pre-compute the mapping from batch size to padded graph size."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        assert capture_sizes is not None, (
            "Cudagraph capture sizes must be set when cudagraphs are enabled."
        )
        self._bs_to_padded_graph_size = (
            self._compute_bs_to_padded_graph_size_for_capture_sizes(capture_sizes)
        )

        # Validate that compile_sizes won't be changed by padding.
        # Only validate when cudagraphs are actually being used.
        if (
            self.compilation_config.compile_sizes
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            for size in self.compilation_config.compile_sizes:
                size = int(size)
                if size < len(self._bs_to_padded_graph_size):
                    padded = self._bs_to_padded_graph_size[size]
                    if padded != size:
                        raise ValueError(
                            f"compile_sizes contains {size} which would be "
                            f"padded to {padded}. All compile_sizes must be "
                            "values that won't be changed by cudagraph padding. "
                            "Use values from cudagraph_capture_sizes."
                        )

    def _get_lora_cases(self) -> list[int]:
        """
        Returns list of has_lora values for CUDA graph capture.
        This is the single source of truth for LoRA capture cases.
        """
        lora_config = self.vllm_config.lora_config
        if lora_config is None:
            # No LoRA configured - single case with no LoRA
            return [0]

        # LoRA is enabled - capture graphs based on cudagraph_specialize_lora
        if self.compilation_config.cudagraph_specialize_lora:
            captured_counts = get_captured_lora_counts(
                lora_config.max_loras, self.specialize_lora_count
            )
            # Specialize: capture separate graphs for with and without LoRA
            return [0] + captured_counts
        else:
            # No specialization: only capture graphs with LoRA active
            return [lora_config.max_loras + 1]

    def _get_non_spec_decode_capture_sizes(self) -> list[int]:
        """Return the qlen=1 decode graph sizes used by a non-spec engine.

        Spec decode rounds the global cudagraph_capture_sizes up to multiples
        of num_speculative_tokens + 1. For a runtime toggle back to ordinary
        decode, we still need the unrounded qlen=1 sizes (1, 2, 4, 8, ...).
        """
        max_size = min(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.compilation_config.max_cudagraph_capture_size,
        )
        if max_size <= 0:
            return []

        if getattr(self.vllm_config, "performance_mode", None) == "interactivity":
            sizes = list(range(1, min(max_size, 32) + 1))
        else:
            sizes = [size for size in (1, 2, 4) if size <= max_size]

        if max_size >= 8:
            sizes.extend(range(8, min(max_size + 1, 256), 8))
        if max_size >= 256:
            sizes.extend(range(256, max_size + 1, 16))

        return sorted(set(sizes))

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
        uniform_decode_query_len: int | None = None,
        exact_num_tokens: bool = False,
        aux_hidden_state_outputs: bool = False,
        use_non_spec_decode_padding: bool = False,
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = (
            uniform_decode_query_len or self.uniform_decode_query_len
        )
        bs_to_padded_graph_size = (
            self._bs_to_non_spec_decode_padded_graph_size
            if use_non_spec_decode_padding
            else self._bs_to_padded_graph_size
        )
        num_tokens_padded = (
            num_tokens if exact_num_tokens else bs_to_padded_graph_size[num_tokens]
        )

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)
            uniform_decode_query_len = 1

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            uniform_decode_query_len=uniform_decode_query_len,
            aux_hidden_state_outputs=aux_hidden_state_outputs,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ):
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode

        # Early exit if cudagraphs are disabled
        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        # Get LoRA cases to capture
        lora_cases = self._get_lora_cases()
        self.captured_lora_counts = [
            lora_count for lora_count in lora_cases if lora_count
        ]
        uses_aux_hidden_states = (
            self.vllm_config.speculative_config is not None
            and self.vllm_config.speculative_config.method
            in ("eagle3", "extract_hidden_states")
        )
        piecewise_aux_hidden_state_cases = [True] if uses_aux_hidden_states else [False]
        spec_decode_aux_hidden_state_cases = (
            [True] if uses_aux_hidden_states else [False]
        )
        exact_decode_aux_hidden_state_cases = (
            [False, True] if uses_aux_hidden_states else [False]
        )

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when mixed mode is enabled."
            )
            for bs, num_active_loras, aux_hidden_state_outputs in product(
                self.compilation_config.cudagraph_capture_sizes,
                lora_cases,
                piecewise_aux_hidden_state_cases,
            ):
                batch_desc = self._create_padded_batch_descriptor(
                    bs,
                    False,
                    num_active_loras > 0,
                    num_active_loras,
                    aux_hidden_state_outputs=aux_hidden_state_outputs,
                )
                # Only relax for PIECEWISE mode. FULL mode needs exact num_reqs
                # because FA3's scheduler_metadata computation depends on it.
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                    batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
                self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            non_spec_decode_capture_sizes: list[int] = []
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            assert self.compilation_config.cudagraph_capture_sizes is not None, (
                "Cudagraph capture sizes must be set when full mode is enabled."
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, num_active_loras, aux_hidden_state_outputs in product(
                cudagraph_capture_sizes_for_decode,
                lora_cases,
                spec_decode_aux_hidden_state_cases,
            ):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs,
                        True,
                        num_active_loras > 0,
                        num_active_loras,
                        aux_hidden_state_outputs=aux_hidden_state_outputs,
                    ),
                )

            # A speculative engine can be toggled into ordinary non-spec decode
            # mode at runtime. Register the same query_len=1 decode graph set
            # that a non-spec engine would have, so SD-as-NonSD does not fall
            # back to eager for batch sizes larger than one.
            if (
                self.vllm_config.speculative_config is not None
                and uniform_decode_query_len > 1
            ):
                non_spec_decode_capture_sizes = self._get_non_spec_decode_capture_sizes()
                self._bs_to_non_spec_decode_padded_graph_size = (
                    self._compute_bs_to_padded_graph_size_for_capture_sizes(
                        non_spec_decode_capture_sizes
                    )
                )
                for bs, num_active_loras, aux_hidden_state_outputs in product(
                    non_spec_decode_capture_sizes,
                    lora_cases,
                    exact_decode_aux_hidden_state_cases,
                ):
                    self.add_cudagraph_key(
                        CUDAGraphMode.FULL,
                        self._create_padded_batch_descriptor(
                            bs,
                            True,
                            num_active_loras > 0,
                            num_active_loras,
                            uniform_decode_query_len=1,
                            aux_hidden_state_outputs=aux_hidden_state_outputs,
                            use_non_spec_decode_padding=True,
                        ),
                    )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        uniform_decode_query_len: int | None = None,
        exact_uniform_decode: bool = False,
        aux_hidden_state_outputs: bool = False,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using piecewise only),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).

        Args:
            num_tokens: Number of tokens in the batch.
            uniform_decode: Whether the batch is uniform decode (i.e. uniform and query
                length is uniform_decode_query_len).
            has_lora: Whether LoRA is active.
            num_active_loras: Number of distinct active LoRA adapters.
            valid_modes: Set of cudagraph modes that are allowed. None means
                all modes are allowed.
            invalid_modes: Set of cudagraph modes to exclude. Subtracted from
                valid_modes to compute allowed modes. (e.g., {FULL} for
                features like cascade attention not supported by full
                cudagraphs). None means no modes are excluded.
        """
        allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()

        if invalid_modes:
            allowed_modes -= invalid_modes

        assert len(allowed_modes) >= 1, (
            f"No allowed cudagraph modes: valid_modes={valid_modes}, "
            f"invalid_modes={invalid_modes}"
        )

        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or num_tokens > self.compilation_config.max_cudagraph_capture_size
            or allowed_modes <= {CUDAGraphMode.NONE}
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        use_non_spec_decode_padding = (
            self.vllm_config.speculative_config is not None
            and uniform_decode
            and (uniform_decode_query_len or self.uniform_decode_query_len) == 1
            and hasattr(self, "_bs_to_non_spec_decode_padded_graph_size")
        )
        if use_non_spec_decode_padding and (
            num_tokens >= len(self._bs_to_non_spec_decode_padded_graph_size)
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        effective_num_active_loras = num_active_loras
        if has_lora and num_active_loras > 0:
            if self.specialize_lora_count:
                # Find the smallest captured `num_active_loras` that is >= the current
                # `num_active_loras`. This is because we only capture graphs for
                # a subset of possible `num_active_loras` values (powers of 2).
                import bisect

                idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
                if idx < len(self.captured_lora_counts):
                    effective_num_active_loras = self.captured_lora_counts[idx]
            else:
                # When not specializing, graphs are captured only with max_loras + 1,
                # so we must use max_loras + 1 for dispatch to find a matching graph.
                assert self.vllm_config.lora_config is not None, (
                    "LoRA config must be set when has_lora is True."
                )
                effective_num_active_loras = self.vllm_config.lora_config.max_loras + 1

        batch_desc = self._create_padded_batch_descriptor(
            num_tokens,
            uniform_decode,
            has_lora,
            effective_num_active_loras,
            uniform_decode_query_len=uniform_decode_query_len,
            exact_num_tokens=exact_uniform_decode and uniform_decode,
            aux_hidden_state_outputs=aux_hidden_state_outputs,
            use_non_spec_decode_padding=use_non_spec_decode_padding,
        )

        if CUDAGraphMode.FULL in allowed_modes:
            # check if key exists for full cudagraph
            # For pure FULL mode, keys are registered with uniform=False.
            batch_desc_to_check = batch_desc
            if self.cudagraph_mode == CUDAGraphMode.FULL:
                batch_desc_to_check = replace(batch_desc, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc_to_check

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            # also check if the relaxed key exists for more "general"
            # piecewise cudagraph
            batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, batch_desc_to_check

        assert CUDAGraphMode.NONE in allowed_modes, (
            f"No matching cudagraph found and NONE is not in "
            f"allowed_modes={allowed_modes}"
        )
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """
        Returns capture descriptors for cudagraph capturing.

        Returns:
            List of (runtime_mode, batch_descriptors) tuples, ordered PIECEWISE
            first then FULL. Batch descriptors are sorted largest-first for
            memory efficiency.
        """
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []

        result = []
        # Return in order: PIECEWISE first, then FULL
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            descs = list(self.cudagraph_keys[mode])
            if descs:
                # Sort by num_tokens descending (largest first)
                descs.sort(key=lambda d: d.num_tokens, reverse=True)
                if (
                    mode == CUDAGraphMode.FULL
                    and self.vllm_config.speculative_config is not None
                ):
                    # Capture EAGLE3/spec-decode graphs before ordinary
                    # qlen=1 Non-SD graphs. The target model's compiled
                    # forward path must first see the aux-hidden-state output
                    # structure; otherwise later SD iterations can miss aux
                    # states even when the runtime flag is enabled.
                    descs.sort(
                        key=lambda d: (
                            not d.aux_hidden_state_outputs,
                            d.uniform_decode_query_len == 1,
                            -d.num_tokens,
                        )
                    )
                result.append((mode, descs))

        return result
