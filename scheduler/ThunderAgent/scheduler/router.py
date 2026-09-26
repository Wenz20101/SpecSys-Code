"""Router with program state tracking - supports multiple backends."""
import asyncio
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable, TextIO, Tuple

import httpx
from fastapi.responses import Response

from ..backend import BackendState, MetricsClient, SGLangMetricsClient, VLLMMetricsClient
from ..backend.skyrl_metrics import SkyRLMetricsClient
from ..program import Program, ProgramStatus, ProgramState
from ..profile.state import ProfileState
from ..config import get_config
from .predictor import StepInfo, StepMap
from .vllm_request_processor import (
    forward_streaming_request,
    forward_non_streaming_request,
    forward_get_request,
)

logger = logging.getLogger(__name__)


@dataclass
class PausedInfo:
    """Metadata for a paused program stored in the global paused pool."""
    program_id: str
    total_tokens: int
    paused_at: float
    origin_backend: Optional[str]  # None for new programs that haven't been assigned yet
    step_count: int


class MultiBackendRouter:
    """Router with program state tracking, supports multiple backends."""

    @staticmethod
    def _create_metrics_client(backend_type: str, url: str) -> MetricsClient:
        """Create a metrics client for the given backend type.
        
        Args:
            backend_type: Backend type ("vllm" or "sglang")
            url: Backend base URL
        
        Returns:
            MetricsClient implementation for the backend type
        """
        if backend_type == "vllm":
            return VLLMMetricsClient(url)
        if backend_type == "sglang":
            return SGLangMetricsClient(url)
        if backend_type == "skyrl":
            return SkyRLMetricsClient(url)
        raise ValueError(f"Unsupported backend_type: {backend_type}")

    def __init__(
        self, 
        backend_urls: str | List[str],
        *, 
        profile_enabled: bool = False,
        scheduling_enabled: bool = True,
        scheduler_interval: float = 5.0,
        scheduler_policy: str = "original",
        dynamic_sd_enabled: bool = False,
        sd_switch_threshold: int = 64,
        backend_type: str = "vllm",
        acting_token_weight: float = 1.0,
        use_acting_token_decay: bool = False,
        llm_task_stats_interval: float = 1.0,
        llm_task_stats_file: Optional[str] = None,
    ) -> None:
        valid_scheduler_policies = {
            "original",
            "remaining_steps",
            "predicted_remaining_steps",
        }
        if scheduler_policy not in valid_scheduler_policies:
            raise ValueError(
                f"Unsupported scheduler_policy={scheduler_policy!r}; "
                f"expected one of {sorted(valid_scheduler_policies)}"
            )
        if sd_switch_threshold <= 0:
            raise ValueError("sd_switch_threshold must be positive")
        if dynamic_sd_enabled and backend_type != "vllm":
            raise ValueError("dynamic SD requires backend_type='vllm'")
        if llm_task_stats_file and llm_task_stats_interval <= 0:
            raise ValueError("llm_task_stats_interval must be positive")

        # Support single URL string or list of URLs
        if isinstance(backend_urls, str):
            backend_urls = [url.strip() for url in backend_urls.split(",") if url.strip()]
        
        # Weight for acting tokens in capacity calculation
        self.acting_token_weight = acting_token_weight
        
        # All backends (pass acting_token_weight as tool_coefficient)
        self.backends: Dict[str, BackendState] = {}
        for url in backend_urls:
            metrics_client = self._create_metrics_client(backend_type, url)
            self.backends[url] = BackendState(
                url=url,
                tool_coefficient=acting_token_weight,
                metrics_client=metrics_client,
                use_acting_token_decay=use_acting_token_decay,
            )
        
        # All programs (single source of truth)
        # Key: program_id, Value: Program (which includes backend_url)
        self.programs: Dict[str, Program] = {}

        # Global paused pool shared across backends (program_id -> PausedInfo).
        self.global_waiting_queue: Dict[str, PausedInfo] = {}

        # Lock for atomic claim (select + pop) from global_waiting_queue.
        self.pause_resume_lock = asyncio.Lock()
        
        # Profile configuration
        self.profile_enabled = profile_enabled
        
        # Scheduling mode: True = "tr" (capacity scheduling), False = "default" (pure proxy)
        self.scheduling_enabled = scheduling_enabled

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        
        # Scheduler task for periodic capacity check
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_stop = False
        self._scheduler_interval = scheduler_interval
        self._scheduler_policy = scheduler_policy
        self._dynamic_sd_enabled = dynamic_sd_enabled
        self._sd_switch_threshold = sd_switch_threshold
        self._sd_enable_threshold = int(sd_switch_threshold * 0.75)
        self._sd_enabled_by_backend: Dict[str, Optional[bool]] = {
            url: None for url in self.backends
        }
        self._sd_projected_load_by_backend: Dict[str, Optional[int]] = {
            url: None for url in self.backends
        }
        self._sd_resumed_by_backend: Dict[str, int] = {
            url: 0 for url in self.backends
        }
        self._remaining_step_prediction_enabled = (
            self.scheduling_enabled
            and scheduler_policy == "predicted_remaining_steps"
        )
        
        # Global char-to-token ratio for token estimation
        # Initial value is 5.0 (1 token ≈ 5 chars), updated with momentum after each request
        self.char_to_token_ratio: float = 5.0
        self._ratio_initialized: bool = False

        # Weight sync coordination: when active, new data-plane requests block
        # on _weight_sync_event and the scheduler loop skips ticks. This lets a
        # trainer pause vLLM backends (e.g. to broadcast new weights) without
        # racing the ThunderAgent scheduler.
        self._weight_sync_active: bool = False
        self._weight_sync_event: asyncio.Event = asyncio.Event()
        self._weight_sync_event.set()  # Not blocking initially
        self._weight_sync_watchdog: Optional[asyncio.Task] = None

        self._llm_task_stats_interval = llm_task_stats_interval
        self._llm_task_stats_path = (
            Path(llm_task_stats_file).expanduser()
            if llm_task_stats_file
            else None
        )
        self._llm_task_stats_task: Optional[asyncio.Task] = None
        self._llm_task_stats_handle: Optional[TextIO] = None
        self._llm_task_stats_writer = None
        self._llm_task_stats_started_monotonic: Optional[float] = None

        self.step_map = StepMap(
            load_history=self._remaining_step_prediction_enabled
        )

    async def start(self):
        """Start the router."""
        logger.info(f"Started router with {len(self.backends)} backend(s): {list(self.backends.keys())}")
        
        # Always fetch cache config (needed for active_program_tokens_ratio)
        for backend in self.backends.values():
            await backend.fetch_cache_config()
        
        # Start metrics monitoring on each backend if enabled
        config = get_config()
        if config.metrics_enabled or self._dynamic_sd_enabled:
            for backend in self.backends.values():
                await backend.start_monitoring(config.metrics_interval)
            if self._dynamic_sd_enabled and not config.metrics_enabled:
                logger.info("Started backend metrics monitoring for dynamic SD")
        
        # Start the shared control loop for capacity scheduling and/or dynamic SD.
        if self.scheduling_enabled or self._dynamic_sd_enabled:
            self._scheduler_stop = False
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info(
                "Started periodic control loop "
                f"(interval={self._scheduler_interval}s, "
                f"policy={self._scheduler_policy}, dynamic_sd={self._dynamic_sd_enabled})"
            )

        self._start_llm_task_stats_monitor()

    async def stop(self):
        """Stop the router."""
        await self._stop_llm_task_stats_monitor()

        # Stop the weight sync watchdog if running
        if self._weight_sync_watchdog is not None:
            self._weight_sync_watchdog.cancel()
            try:
                await self._weight_sync_watchdog
            except asyncio.CancelledError:
                pass
            self._weight_sync_watchdog = None

        # Stop the scheduler
        if self._scheduler_task:
            self._scheduler_stop = True
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
            logger.info("Stopped scheduler loop")
        
        # Stop metrics monitoring on each backend
        for backend in self.backends.values():
            await backend.stop_monitoring()
        
        await self.client.aclose()
        logger.info("Router stopped")

    def get_inference_task_count(self) -> int:
        """Count programs whose requests are currently executing on a backend."""
        return sum(
            1
            for state in self.programs.values()
            if state.state == ProgramState.ACTIVE
            and state.status == ProgramStatus.REASONING
            and state.backend_url in self.backends
        )

    def _write_llm_task_stats_sample(self) -> None:
        if self._llm_task_stats_writer is None:
            return
        now = time.time()
        now_monotonic = time.monotonic()
        started_at = self._llm_task_stats_started_monotonic or now_monotonic
        self._llm_task_stats_writer.writerow(
            [
                f"{now:.6f}",
                f"{max(0.0, now_monotonic - started_at):.6f}",
                self.get_inference_task_count(),
            ]
        )
        if self._llm_task_stats_handle is not None:
            self._llm_task_stats_handle.flush()

    def _start_llm_task_stats_monitor(self) -> None:
        if self._llm_task_stats_path is None or self._llm_task_stats_task is not None:
            return
        self._llm_task_stats_path.parent.mkdir(parents=True, exist_ok=True)
        self._llm_task_stats_handle = self._llm_task_stats_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._llm_task_stats_writer = csv.writer(self._llm_task_stats_handle)
        self._llm_task_stats_writer.writerow(
            ["timestamp", "elapsed_seconds", "llm_task_count"]
        )
        self._llm_task_stats_started_monotonic = time.monotonic()
        self._write_llm_task_stats_sample()
        self._llm_task_stats_task = asyncio.create_task(
            self._llm_task_stats_loop()
        )
        logger.info(
            "Started LLM task-count sampling (interval=%.3fs, output=%s)",
            self._llm_task_stats_interval,
            self._llm_task_stats_path,
        )

    async def _stop_llm_task_stats_monitor(self) -> None:
        if self._llm_task_stats_task is not None:
            self._llm_task_stats_task.cancel()
            try:
                await self._llm_task_stats_task
            except asyncio.CancelledError:
                pass
            self._llm_task_stats_task = None
        if self._llm_task_stats_handle is not None:
            self._write_llm_task_stats_sample()
            self._llm_task_stats_handle.close()
            self._llm_task_stats_handle = None
            self._llm_task_stats_writer = None
            logger.info("Wrote LLM task-count samples to %s", self._llm_task_stats_path)

    async def _llm_task_stats_loop(self) -> None:
        while True:
            await asyncio.sleep(self._llm_task_stats_interval)
            try:
                self._write_llm_task_stats_sample()
            except Exception:
                logger.exception("Failed to write LLM task-count sample")
                return

    # -------------------------------------------------------------------------
    # Weight Sync Coordination
    # -------------------------------------------------------------------------

    @property
    def weight_sync_active(self) -> bool:
        return self._weight_sync_active

    async def begin_weight_sync(self, timeout: float = 300.0) -> None:
        """Enter weight sync mode.

        Effects:
        1. Sets weight_sync_active flag
        2. Clears _weight_sync_event so new data-plane requests block
        3. Scheduler loop will skip ticks while flag is set
        4. Starts a watchdog timer that auto-exits after ``timeout`` seconds

        Idempotent: a second begin_weight_sync call while already active logs
        a warning and returns without resetting the watchdog.
        """
        if self._weight_sync_active:
            logger.warning("begin_weight_sync called while already in weight sync mode")
            return
        self._weight_sync_active = True
        self._weight_sync_event.clear()
        self._weight_sync_watchdog = asyncio.create_task(self._weight_sync_timeout(timeout))
        logger.info("Weight sync mode ENTERED - holding new requests, suspending scheduler")

    async def end_weight_sync(self) -> None:
        """Exit weight sync mode and release any held requests.

        Idempotent: end_weight_sync called while not active logs a warning
        and returns.
        """
        if not self._weight_sync_active:
            logger.warning("end_weight_sync called while not in weight sync mode")
            return
        if self._weight_sync_watchdog is not None:
            self._weight_sync_watchdog.cancel()
            try:
                await self._weight_sync_watchdog
            except asyncio.CancelledError:
                pass
            self._weight_sync_watchdog = None
        self._weight_sync_active = False
        self._weight_sync_event.set()
        logger.info("Weight sync mode EXITED - releasing held requests, resuming scheduler")

    async def _weight_sync_timeout(self, timeout: float) -> None:
        """Watchdog: auto-exit weight sync mode if end_weight_sync is never called."""
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if self._weight_sync_active:
            logger.error(
                "Weight sync watchdog fired after %.0fs - auto-exiting weight sync mode to unblock requests",
                timeout,
            )
            self._weight_sync_active = False
            self._weight_sync_event.set()
            self._weight_sync_watchdog = None

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------

    def get_backend(self, url: str) -> Optional[BackendState]:
        """Get a backend by URL."""
        return self.backends.get(url)

    def get_default_backend(self) -> BackendState:
        """Get the first backend (for simple single-backend usage)."""
        return next(iter(self.backends.values()))

    def select_backend_for_new_program_default(self) -> BackendState:
        """Select the least loaded backend for a new program."""
        # Count programs per backend
        #### TODO change backend assign logistics
        backend_load: Dict[str, int] = {url: 0 for url in self.backends}
        for state in self.programs.values():
            if state.backend_url in backend_load:
                backend_load[state.backend_url] += 1
        
        # Find the backend with least programs (only consider healthy ones)
        min_load = float('inf')
        best_backend = None
        for url, load in backend_load.items():
            backend = self.backends[url]
            if backend.healthy and load < min_load:
                min_load = load
                best_backend = backend
        
        # Fallback to first backend if all unhealthy
        return best_backend or self.get_default_backend()

    # -------------------------------------------------------------------------
    # Program State Management
    # -------------------------------------------------------------------------

    @staticmethod
    def _estimate_system_prompt_tokens(payload: Dict[str, Any]) -> int:
        """Estimate system prompt tokens from the first request payload."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0
        
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or item.get("input_text")
                    if isinstance(text, str):
                        parts.append(text)
        
        if not parts:
            return 0
        
        text = "\n".join(parts)
        return max(0, len(text) // 5)

    def get_or_create_program(
        self,
        program_id: str,
        type_id: str,
        total_steps: int,
        *,
        agent_type: str = "unknown",
        num_plan_steps: Optional[int] = None,
    ) -> Program:
        """Get existing program or create new one.
        
        Only creates the program and estimates token count.
        Backend assignment is deferred to update_program_before_request.
        
        Args:
            program_id: Unique identifier for the program
            payload: Request payload, used to estimate token count for new programs
        """
        if program_id not in self.programs:
            profile = ProfileState(program_id=program_id) if self.profile_enabled else None
            state = Program(
                program_id=program_id,
                type_id=type_id,
                agent_type=self.step_map.normalize_agent_type(agent_type),
                num_plan_steps=num_plan_steps,
                total_steps=total_steps,
                backend_url=None,
                status=ProgramStatus.REASONING,  # Request arrived, program is reasoning
                state=ProgramState.ACTIVE,
                profile=profile,
            )
            # Token estimation is done in update_program_before_request
            self.programs[program_id] = state
            logger.debug(f"Created program {program_id}")

        state = self.programs[program_id]
        normalized_type = self.step_map.normalize_agent_type(agent_type)
        if normalized_type != "unknown":
            state.agent_type = normalized_type
        if num_plan_steps is not None and num_plan_steps > 0:
            state.num_plan_steps = num_plan_steps
        return state
    
    def _select_backend_for_new_program(self, estimated_tokens: int = 0) -> Optional[str]:
        """Select a backend for a new program (TR mode only).
        
        This function is only called when scheduling_enabled=True.
        
        Args:
            estimated_tokens: Estimated token count for the new program
        
        Returns:
            backend_url if can assign directly (queue empty + has capacity)
            None if should wait in queue
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # Only assign if queue is empty and backend has capacity
        if len(self.global_waiting_queue) > 0:
            return None  # Queue not empty, must wait for fairness
        
        # Find backend with least active tokens that has capacity for this program
        best_backend = None
        min_tokens = float('inf')
        required_capacity = estimated_tokens + BUFFER_PER_PROGRAM
        
        for backend in self.backends.values():
            if not backend.healthy:
                continue
            if backend.remaining_capacity() < required_capacity:
                continue  # Not enough capacity for this program
            if backend.active_program_tokens < min_tokens:
                min_tokens = backend.active_program_tokens
                best_backend = backend
        
        return best_backend.url if best_backend else None

    def _select_backend_for_new_program_specsys(self, state: Program) -> Optional[str]:
        """Select a backend for a new program based on predicted time and load.
        
        This function is only called when scheduling_enabled=True.
        
        Args:
            state: The Program state containing total_tokens and predicted_remaining_time
        
        Returns:
            backend_url if can assign directly (queue empty + has capacity)
            None if should wait in queue
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # Only assign if queue is empty (Strict fairness: wait in queue if others are waiting)
        if len(self.global_waiting_queue) > 0:
            return None 
        
        required_capacity = state.total_tokens + BUFFER_PER_PROGRAM
        
        # 1. Filter eligible backends (must be healthy and have enough capacity)
        eligible_backends = []
        for backend in self.backends.values():
            if not backend.healthy:
                continue
            if backend.remaining_capacity() >= required_capacity:
                eligible_backends.append(backend)
                
        if not eligible_backends:
            return None  # No backend has enough capacity right now
            
        # 2. Score eligible backends based on Execution Time and Load
        def backend_score(backend: BackendState) -> tuple[float, int]:
            # Primary Metric: Predicted remaining time on this specific backend
            time_cost = float('inf')
            if hasattr(state, 'predicted_remaining_time') and state.predicted_remaining_time:
                time_cost = state.predicted_remaining_time.get(backend.url, float('inf'))
            
            # Secondary Metric (Tie-breaker & Load Balancer): Active program tokens
            # If times are identical (or unknown), pick the least loaded backend
            load_cost = backend.active_program_tokens
            
            return (time_cost, load_cost)
            
        # 3. Sort backends (lower score is better) and pick the optimal one
        eligible_backends.sort(key=backend_score)
        best_backend = eligible_backends[0]
        
        return best_backend.url

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get the backend assigned to a program."""
        state = self.programs.get(program_id)
        if state and state.backend_url in self.backends:
            return self.backends[state.backend_url]
        return self.get_default_backend()

    def update_backend_for_program(self, program_id: str) -> bool:
        """
        Check the load of the current backend for the program. 
        If it's overloaded, migrate the program to a healthier/less-loaded backend.
        Returns True if migrated, False otherwise.
        """
        state = self.programs.get(program_id)
        if not state or not state.backend_url:
            return

        current_backend = self.backends.get(state.backend_url)
        
        def get_backend_load(backend: BackendState) -> float:
            running, waiting = backend.metrics_num_requests
            return running + waiting

        current_load = get_backend_load(current_backend)

        MIGRATION_MARGIN = 16

        best_backend_url = state.backend_url
        min_load = current_load

        for backend_url, backend in self.backends.items():
            if backend_url == state.backend_url or not backend.healthy:
                continue
            b_load = get_backend_load(backend)
            if b_load < (min_load - MIGRATION_MARGIN):
                min_load = b_load
                best_backend_url = backend_url

        if best_backend_url != state.backend_url:
            new_backend = self.backends[best_backend_url]
            
            logger.info(
                f"[Load Balancing] Migrating program {program_id} "
                f"from {state.backend_url} (load: {current_load}) "
                f"to {best_backend_url} (load: {min_load})"
            )

            if current_backend:
                current_backend.unregister_program(program_id)
            state.backend_url = new_backend.url
            new_backend.register_program(program_id, state)

            return True
            
        return False
                

    async def update_program_before_request(self, program_id: str, state: Program, payload: Dict[str, Any]) -> bool:
        """Update program state before sending request to vLLM.
        
        If scheduling_enabled=False (default mode): pure proxy, no capacity checks.
        If scheduling_enabled=True (tr mode): wait for scheduler to resume if PAUSED.
        
        Returns: True if can proceed
        """
        # Block if weight sync is in progress - all data-plane requests wait here
        if not self._weight_sync_event.is_set():
            logger.debug(f"Program {program_id} waiting for weight sync to complete")
            await self._weight_sync_event.wait()

        state.step_count += 1
        is_new_program = state.step_count == 1

        if self._remaining_step_prediction_enabled:
            arrived_time = time.time()
            prefix_k = max(0, state.step_count - 1)
            normalized_type = self.step_map.normalize_agent_type(state.agent_type)
            if normalized_type in {"B", "C"}:
                context = self.step_map.prediction_context_from_payload(
                    payload,
                    normalized_type,
                    state.prefix_states,
                )
                state.prediction_context = context
                state_bucket = str(context.get("state_bucket") or "normal")
                while len(state.prefix_states) < prefix_k:
                    state.prefix_states.append("normal")
                if len(state.prefix_states) == prefix_k:
                    state.prefix_states.append(state_bucket)
                else:
                    state.prefix_states[prefix_k] = state_bucket
                previous_state = state.prefix_states[prefix_k - 1] if prefix_k else "bos"
            else:
                state_bucket = None
                previous_state = None

            state.predicted_remaining_steps = self.step_map.predict_remaining_steps(
                state.type_id,
                state.step_count,
                agent_type=state.agent_type,
                num_plan_steps=state.num_plan_steps,
                prefix_k=prefix_k,
                state_bucket=state_bucket,
                previous_state=previous_state,
            )
            logger.debug(
                "Predicted remaining steps for program %s at step %d "
                "(prefix=%d, state=%s, previous=%s): %s",
                program_id,
                state.step_count,
                prefix_k,
                state_bucket,
                previous_state,
                state.predicted_remaining_steps,
            )
            result = self.step_map.predict_remaining_time(
                state.type_id,
                state.step_count,
                self.backends.keys(),
                agent_type=state.agent_type,
                num_plan_steps=state.num_plan_steps,
                prefix_k=prefix_k,
                state_bucket=state_bucket,
                previous_state=previous_state,
            )
            if result:
                state.predicted_remaining_time = {
                    key: value["remaining_time"]
                    for key, value in result["backends"].items()
                }

            if is_new_program:
                state.last_arrived_time = arrived_time
            else:
                last_step_time = arrived_time - state.last_arrived_time
                self.step_map.update_step_time(
                    state.type_id,
                    last_step_time,
                    state.backend_url,
                )
                state.last_arrived_time = arrived_time
        
        # Update context_len and estimate total_tokens using char_to_token_ratio
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        state.total_tokens = int(state.context_len / self.char_to_token_ratio)
        
        # ---------------------------------------------------------------------
        # Default mode: pure proxy, no scheduling
        # ---------------------------------------------------------------------
        if not self.scheduling_enabled:
            backend = self.backends.get(state.backend_url)
            if not backend:
                # Assign to least loaded backend
                backend = self.select_backend_for_new_program_default()
                state.backend_url = backend.url
            
            if is_new_program:
                backend.register_program(program_id, state)
            # Status change is enough - token stats are computed from program status
            state.status = ProgramStatus.REASONING
            state.acting_since = None
            return True
        
        # ---------------------------------------------------------------------
        # TR mode: scheduler-based capacity management
        # ---------------------------------------------------------------------
        # Step 1: Handle PAUSED programs (wait for resume)
        # Note: waiting_event is created in _pause_program and cleared in _resume_program.
        # If waiting_event is not None, the program is still PAUSED and we must wait.
        # Set status to REASONING before waiting so the scheduler sees the correct
        # priority: REASONING programs (pending request) > ACTING programs (idle).
        if state.waiting_event is not None:
            state.status = ProgramStatus.REASONING
            state.acting_since = None
            await self._wait_for_resume(program_id, state)
            # After _wait_for_resume returns, _resume_program has been called which:
            # - Registered program with target backend
            # - Set state.state = ACTIVE
            # - Cleared waiting_event = None
            # Fall through to Step 3
        
        # Step 2: Handle new programs (assign backend or queue)
        if is_new_program and state.backend_url is None:
            backend_url = self._select_backend_for_new_program(state.total_tokens)
            # backend_url = self._select_backend_for_new_program_specsys(state)
            if backend_url:
                # Direct assignment: register program with backend
                state.backend_url = backend_url
                backend = self.backends[backend_url]
                backend.register_program(program_id, state)
                logger.debug(f"Assigned new program {program_id} to {backend_url}")
                state.status = ProgramStatus.REASONING
                state.acting_since = None
                return True
            else:
                # Queue and wait — set REASONING before waiting for priority
                state.status = ProgramStatus.REASONING
                state.acting_since = None
                state.waiting_event = asyncio.Event()
                state.state = ProgramState.PAUSED
                self._add_to_global_waiting_queue_sync(program_id, state, backend=None)
                logger.debug(f"Queued new program {program_id} (tokens={state.total_tokens})")
                await self._wait_for_resume(program_id, state)
                # After resume: program registered with backend by _resume_program
                return True
        
        # new step: update state.backend
        # self.update_backend_for_program(program_id)


        # Step 3: Normal case - existing ACTIVE program with backend
        backend = self.backends.get(state.backend_url)
        if not backend:
            logger.error(f"Program {program_id} has no valid backend")
            return False
        
        state.status = ProgramStatus.REASONING
        state.acting_since = None
        return True

    def update_program_after_request(
        self, program_id: str, state: Program, total_tokens: int, prompt_tokens: int = 0
    ) -> None:
        """Update program state after receiving response from vLLM.
        
        Transitions to ACTING (off GPU, executing tool).
        Updates token counts. If marked for pause, pause immediately.
        Also updates the global char_to_token_ratio for future token estimation.
        
        Args:
            program_id: The program ID
            state: The program state
            total_tokens: Total tokens from vLLM response (prompt + completion)
            prompt_tokens: Prompt/prefill tokens from vLLM response
        """
        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        state.acting_since = time.time()
        
        # Update global char_to_token_ratio based on actual prefill
        # ratio = context_len / prompt_tokens (chars per token)
        if prompt_tokens > 0 and state.context_len > 0:
            current_ratio = state.context_len / prompt_tokens
            if not self._ratio_initialized:
                # First request: directly assign
                self.char_to_token_ratio = current_ratio
                self._ratio_initialized = True
                logger.debug(f"Initialized char_to_token_ratio={self.char_to_token_ratio:.2f}")
            else:
                # Subsequent requests: momentum update (0.2 new + 0.8 old)
                self.char_to_token_ratio = 0.2 * current_ratio + 0.8 * self.char_to_token_ratio
                logger.debug(f"Updated char_to_token_ratio={self.char_to_token_ratio:.2f} (sample={current_ratio:.2f})")
        
        # Update total_tokens - token stats are computed from program state
        state.total_tokens = total_tokens
        
        # If marked for pause, pause now (while in ACTING state)
        if state.marked_for_pause:
            self._clear_mark_and_pause(program_id, state)

    def update_program_tokens_streaming(self, state: Program, delta_tokens: int) -> None:
        """Update program tokens incrementally during streaming.
        
        Called periodically (e.g., every 20 tokens) during streaming to 
        update the token counts in real-time. The program is in REASONING
        state during streaming.
        
        Args:
            state: The program state
            delta_tokens: Number of new tokens since last update
        """
        # Just update program's total_tokens - backend stats are computed from program state
        state.total_tokens += delta_tokens

    async def release_program(self, program_id: str) -> bool:
        """Stop a program and release its resources.
        
        Removes tokens from tracking. Resume of waiting programs is handled by the scheduler.
        """
        if program_id not in self.programs:
            return False
        
        arrived_time = time.time()

        state = self.programs[program_id]
        backend = self.backends.get(state.backend_url) if state.backend_url else None
        
        # Clean up based on current lifecycle state
        if state.state == ProgramState.PAUSED:
            # Remove from waiting queue
            await self._remove_from_global_waiting_queue(program_id)
            if state.waiting_event:
                state.waiting_event.set()  # Unblock any waiting coroutine
        elif backend and state.state == ProgramState.ACTIVE:
            backend.unregister_program(program_id)
        
        # Clear mark if was marked
        if backend and state.marked_for_pause:
            backend.future_paused_tokens -= state.total_tokens
            if backend.future_paused_tokens < 0:
                backend.future_paused_tokens = 0
            state.marked_for_pause = False
        
        state.state = ProgramState.TERMINATED
        
        # Update step_map (step_cnt, step_time)
        self.step_map.update_step_cnt(
            state.type_id,
            state.step_count,
            agent_type=state.agent_type,
            prefix_states=state.prefix_states,
        )
        if state.last_arrived_time is not None:
            last_step_time = arrived_time - state.last_arrived_time
            self.step_map.update_step_time(state.type_id, last_step_time, state.backend_url)

        # Remove from programs dict
        del self.programs[program_id]
        
        logger.info(f"Released and removed program: {program_id}")
        return True

    def get_programs_on_backend(self, backend_url: str) -> Dict[str, Program]:
        """Get all programs assigned to a specific backend."""
        return {
            pid: state for pid, state in self.programs.items()
            if state.backend_url == backend_url
        }

    # -------------------------------------------------------------------------
    # Global Paused Pool
    # -------------------------------------------------------------------------

    def _add_to_global_waiting_queue_sync(
        self, program_id: str, state: Program, backend: Optional[BackendState] = None,
    ) -> None:
        """Add a program to the global waiting queue (synchronous version)."""
        paused_info = PausedInfo(
            program_id=program_id,
            total_tokens=state.total_tokens,
            paused_at=time.time(),
            origin_backend=backend.url if backend else None,
            step_count=state.step_count,
        )
        self.global_waiting_queue[program_id] = paused_info

    async def _remove_from_global_waiting_queue(self, program_id: str) -> Optional[PausedInfo]:
        """Remove a program from the global paused pool."""
        async with self.pause_resume_lock:
            return self.global_waiting_queue.pop(program_id, None)

    def _get_paused_programs_sorted(
        self, ascending: bool = True
    ) -> List[Tuple[str, Program, PausedInfo]]:
        """Get paused programs from the global pool, sorted by total_tokens.

        Callers that require consistency should hold pause_resume_lock.
        Resume priority is applied separately by scheduler_policy.
        """
        programs: List[Tuple[str, Program, PausedInfo]] = []
        for pid, info in self.global_waiting_queue.items():
            state = self.programs.get(pid)
            if not state:
                continue
            programs.append((pid, state, info))
        return sorted(programs, key=lambda x: x[2].total_tokens, reverse=not ascending)

    @staticmethod
    def _remaining_steps_priority(state: Program) -> tuple[float, int, str]:
        """Sort key: programs closer to completion are scheduled first."""
        try:
            total_steps = int(state.total_steps)
        except (TypeError, ValueError):
            total_steps = 0

        if total_steps <= 0:
            remaining_steps = float("inf")
        else:
            remaining_steps = total_steps - state.step_count
        return (remaining_steps, state.step_count, state.program_id)

    def _get_resume_candidates_original(
        self, paused_programs: List[Tuple[str, Program, PausedInfo]]
    ) -> list[Program]:
        """Original priority: group by status, then sort each group by total_tokens."""
        reasoning_group: list[Program] = []
        new_program_group: list[Program] = []
        acting_group: list[Program] = []

        for _, state, _ in paused_programs:
            if state.step_count == 1:
                new_program_group.append(state)
            elif state.status == ProgramStatus.REASONING:
                reasoning_group.append(state)
            else:
                acting_group.append(state)

        reasoning_group.sort(key=lambda state: state.total_tokens)
        new_program_group.sort(key=lambda state: state.total_tokens)
        acting_group.sort(key=lambda state: state.total_tokens)

        return reasoning_group + new_program_group + acting_group

    def _get_resume_candidates_remaining_steps(
        self, paused_programs: List[Tuple[str, Program, PausedInfo]]
    ) -> list[Program]:
        """Ideal remaining-step priority using the caller-provided total_steps."""
        candidates = [state for _, state, _ in paused_programs]
        candidates.sort(key=self._remaining_steps_priority)
        return candidates

    @staticmethod
    def _predicted_remaining_steps_priority(
        state: Program,
    ) -> tuple[float, int, str]:
        """Sort key using only the table-predicted remaining step count."""
        predicted = state.predicted_remaining_steps
        if predicted is None or not math.isfinite(predicted) or predicted < 0:
            predicted = float("inf")
        return (predicted, state.step_count, state.program_id)

    def _get_resume_candidates_predicted_remaining_steps(
        self, paused_programs: List[Tuple[str, Program, PausedInfo]]
    ) -> list[Program]:
        """Predicted SRTF priority without reading the ideal total_steps field."""
        predicted: list[Program] = []
        unavailable: list[Tuple[str, Program, PausedInfo]] = []
        for paused in paused_programs:
            state = paused[1]
            value = state.predicted_remaining_steps
            if value is not None and math.isfinite(value) and value >= 0:
                predicted.append(state)
            else:
                unavailable.append(paused)

        predicted.sort(key=self._predicted_remaining_steps_priority)
        return predicted + self._get_resume_candidates_original(unavailable)

    def _get_resume_candidates(
        self, paused_programs: List[Tuple[str, Program, PausedInfo]]
    ) -> list[Program]:
        if self._scheduler_policy == "remaining_steps":
            return self._get_resume_candidates_remaining_steps(paused_programs)
        if self._scheduler_policy == "predicted_remaining_steps":
            return self._get_resume_candidates_predicted_remaining_steps(paused_programs)
        return self._get_resume_candidates_original(paused_programs)

    def get_paused_counts_by_backend(self) -> Dict[str, int]:
        """Count paused programs per backend using paused pool metadata."""
        counts = {url: 0 for url in self.backends}
        for info in self.global_waiting_queue.values():
            if info.origin_backend in counts:
                counts[info.origin_backend] += 1
        return counts

    # -------------------------------------------------------------------------
    # Capacity-based Scheduling (pause/resume)
    # -------------------------------------------------------------------------

    def _get_acting_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, Program]]:
        """Get ACTING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url and state.status == ProgramStatus.ACTING
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _get_reasoning_programs_sorted(self, backend_url: str, ascending: bool = True) -> List[Tuple[str, Program]]:
        """Get REASONING programs on a backend, sorted by total_tokens.
        
        Args:
            ascending: If True, smallest first. If False, largest first.
        """
        programs = [
            (pid, state) for pid, state in self.programs.items()
            if state.backend_url == backend_url 
            and state.status == ProgramStatus.REASONING
            and not state.marked_for_pause  # Exclude already marked
        ]
        return sorted(programs, key=lambda x: x[1].total_tokens, reverse=not ascending)

    def _pause_program(self, program_id: str, state: Program) -> None:
        """Pause a program: remove from active and total, add to global paused pool.
        
        Only call this for ACTING programs. REASONING programs should be marked instead.
        Sets backend_url to None and saves origin_backend for resume.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Unregister from backend
        backend.unregister_program(program_id)
        
        # Add to global paused pool
        self._add_to_global_waiting_queue_sync(program_id, state, backend)
        
        # Save origin backend and clear current backend
        state.origin_backend = state.backend_url
        state.backend_url = None
        state.state = ProgramState.PAUSED
        
        # Create waiting event if needed
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        
        logger.info(f"Paused program {program_id} from {state.origin_backend} (tokens={state.total_tokens})")

    def _mark_program_for_pause(self, program_id: str, state: Program) -> None:
        """Mark a REASONING program for pause. It will be paused on next request.
        
        Adds the program's tokens to future_paused_tokens for capacity calculation.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        state.marked_for_pause = True
        backend.future_paused_tokens += state.total_tokens
        
        logger.info(f"Marked program {program_id} for pause (tokens={state.total_tokens}, future_paused={backend.future_paused_tokens})")

    def _clear_mark_and_pause(self, program_id: str, state: Program) -> None:
        """Clear the mark from a program and pause it.
        
        Called when a marked program's next request arrives.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return
        
        # Clear the mark and subtract from future_paused_tokens
        state.marked_for_pause = False
        backend.future_paused_tokens -= state.total_tokens
        if backend.future_paused_tokens < 0:
            backend.future_paused_tokens = 0
        
        # Now actually pause the program
        self._pause_program(program_id, state)

    def _resume_program(
        self,
        state: Program,
        target_backend: Optional[BackendState] = None,
    ) -> None:
        """Resume a paused program after it has been claimed from the pool.
        
        Args:
            state: The program state (contains program_id, origin_backend, etc.)
            target_backend: Backend to resume to (may differ from origin for migration)
        
        Uses state.origin_backend as fallback if target_backend is not provided.
        """
        # Use origin_backend as fallback (backend_url is None when paused)
        origin_backend = self.backends.get(state.origin_backend) if state.origin_backend else None
        backend = target_backend or origin_backend
        if not backend:
            return

        if state.state != ProgramState.PAUSED:
            return

        # Register with target backend
        backend.register_program(state.program_id, state)
        state.backend_url = backend.url
        state.origin_backend = None  # Clear origin_backend after resume
        state.state = ProgramState.ACTIVE
        
        # Signal waiting event and clear it (resume completes the pause-resume cycle)
        if state.waiting_event:
            state.waiting_event.set()
            state.waiting_event = None
        
        logger.info(f"Resumed program {state.program_id} to {backend.url} (status={state.status.value}, tokens={state.total_tokens}, active={backend.active_program_tokens})")

    async def _claim_specific_paused(
        self, program_id: str
    ) -> Optional[Program]:
        """Claim a specific paused program in a lock-protected way."""
        async with self.pause_resume_lock:
            info = self.global_waiting_queue.get(program_id)
            if info is None:
                return None
            state = self.programs.get(program_id)
            if not state or state.state != ProgramState.PAUSED:
                self.global_waiting_queue.pop(program_id, None)
                return None
            self.global_waiting_queue.pop(program_id, None)
            return state

    # -------------------------------------------------------------------------
    # Periodic Scheduler
    # -------------------------------------------------------------------------

    async def _scheduler_loop(self):
        """Run enabled capacity-scheduling and dynamic-SD controls periodically."""
        while not self._scheduler_stop:
            try:
                await asyncio.sleep(self._scheduler_interval)
                if self._weight_sync_active:
                    continue  # Skip scheduler tick during weight sync
                await self._scheduled_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

    async def _scheduled_check(self):
        """Periodic check: resume waiting programs, enforce capacity, update metrics."""
        # Step 1: Fetch fresh metrics
        for backend in self.backends.values():
            await backend.fetch_metrics()

        # Step 2: Resume waiting programs and count only requests that will be
        # forwarded to vLLM during this tick. Idle ACTING programs do not add to
        # the projected running batch.
        resumed_by_backend: Dict[str, int] = {}
        if self.scheduling_enabled:
            resumed_by_backend = await self._greedy_resume()
            # await self._greedy_resume_specsys()

        # Step 3: Reconcile SD against the projected next batch. The full TA
        # queue is deliberately not counted because it cannot enter vLLM at once.
        if self._dynamic_sd_enabled:
            await self._update_dynamic_sd(resumed_by_backend)

        if not self.scheduling_enabled:
            return
        
        # Step 4: Check thrashing and pause if needed (uses original capacity, no decay)
        for url, backend in self.backends.items():
            if backend.cache_config and backend.remaining_capacity() < 0:
                await self._pause_until_safe(backend)

    @staticmethod
    def _dynamic_sd_target_state(
        projected_batch_size: int,
        enable_threshold: int,
        disable_threshold: int,
        current_state: Optional[bool],
    ) -> Optional[bool]:
        """Apply hysteresis and return the desired SD state."""
        if projected_batch_size >= disable_threshold:
            return False
        if projected_batch_size <= enable_threshold:
            return True
        return current_state

    @staticmethod
    def _speculative_decoding_url(backend_url: str) -> str:
        base = backend_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/v1/engine/speculative_decoding"

    async def _set_backend_sd(
        self,
        backend: BackendState,
        enabled: bool,
        running_batch_size: int,
        resumed_requests: int,
        waiting_queue_size: int,
    ) -> None:
        """Apply one SD state transition and remember only successful changes."""
        if self._sd_enabled_by_backend[backend.url] is enabled:
            return

        projected_batch_size = running_batch_size + resumed_requests
        try:
            response = await self.client.post(
                self._speculative_decoding_url(backend.url),
                json={"enabled": enabled},
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") != "success":
                raise RuntimeError(f"vLLM ignored SD toggle: {result}")
        except Exception as exc:
            logger.warning(
                "Failed to set speculative decoding=%s on %s (running=%d, "
                "resumed=%d, ta_waiting=%d, projected=%d, thresholds=%d/%d): %s",
                enabled,
                backend.url,
                running_batch_size,
                resumed_requests,
                waiting_queue_size,
                projected_batch_size,
                self._sd_enable_threshold,
                self._sd_switch_threshold,
                exc,
            )
            return

        self._sd_enabled_by_backend[backend.url] = enabled
        logger.info(
            "Set speculative decoding=%s on %s (running=%d, resumed=%d, "
            "ta_waiting=%d, projected=%d, thresholds=%d/%d)",
            enabled,
            backend.url,
            running_batch_size,
            resumed_requests,
            waiting_queue_size,
            projected_batch_size,
            self._sd_enable_threshold,
            self._sd_switch_threshold,
        )

    async def _update_dynamic_sd(
        self, resumed_by_backend: Optional[Dict[str, int]] = None
    ) -> None:
        """Reconcile SD using running requests plus imminent resumed requests."""
        resumed_by_backend = resumed_by_backend or {}
        queue_size = len(self.global_waiting_queue)
        updates = []
        for backend in self.backends.values():
            metrics = backend.latest_metrics
            if metrics is None or not backend.healthy:
                logger.warning(
                    "Skipping dynamic SD decision for %s: fresh metrics unavailable",
                    backend.url,
                )
                continue
            resumed_requests = resumed_by_backend.get(backend.url, 0)
            projected_batch_size = metrics.num_requests_running + resumed_requests
            self._sd_projected_load_by_backend[backend.url] = projected_batch_size
            self._sd_resumed_by_backend[backend.url] = resumed_requests
            enabled = self._dynamic_sd_target_state(
                projected_batch_size,
                self._sd_enable_threshold,
                self._sd_switch_threshold,
                self._sd_enabled_by_backend[backend.url],
            )
            if enabled is None:
                continue
            updates.append(
                self._set_backend_sd(
                    backend,
                    enabled,
                    metrics.num_requests_running,
                    resumed_requests,
                    queue_size,
                )
            )
        if updates:
            await asyncio.gather(*updates)

    async def _pause_until_safe(self, backend: BackendState):
        """Pause programs until backend is within capacity.
        
        Priority: ACTING first (smallest tokens), then REASONING (smallest tokens).
        """
        paused_count = 0
        
        while backend.remaining_capacity() < 0:
            # Priority 1: Pause ACTING programs (smallest first)
            acting_programs = self._get_acting_programs_sorted(backend.url, ascending=True)
            if acting_programs:
                program_id, state = acting_programs[0]
                self._pause_program(program_id, state)
                paused_count += 1
                logger.info(f"Scheduler paused ACTING program {program_id} (tokens={state.total_tokens})")
                continue
            
            # Priority 2: Pause REASONING programs (smallest first)
            # Note: REASONING programs are on GPU, we just mark them for pause
            reasoning_programs = self._get_reasoning_programs_sorted(backend.url, ascending=True)
            if reasoning_programs:
                program_id, state = reasoning_programs[0]
                self._mark_program_for_pause(program_id, state)
                paused_count += 1
                logger.info(f"Scheduler marked REASONING program {program_id} for pause (tokens={state.total_tokens})")
                # After marking, we've accounted for future_paused_tokens, continue checking
                continue
            
            # No more programs to pause
            break
        
        if paused_count > 0:
            logger.info(f"Scheduler paused/marked {paused_count} programs on {backend.url}")

    async def _greedy_resume(self) -> Dict[str, int]:
        """Resume waiting programs using Best Fit Decreasing (BFD) bin-packing.
        
        Algorithm:
        1. Compute total backend capacity across all healthy backends.
        2. Select programs in priority order whose cumulative required tokens
           ≤ total capacity — the maximum set we can possibly resume.
        3. BFD placement: sort selected programs descending by required tokens,
           place the largest on the backend with the highest remaining capacity,
           re-sort backends after each placement.
        4. Terminate when all selected programs are placed or the smallest
           remaining program exceeds every backend's remaining capacity.
        
        Candidate priority is selected by scheduler_policy:
        - original: REASONING step>1, then new programs, then ACTING; each group uses total_tokens ascending.
        - remaining_steps: ideal SRTF using total_steps - step_count.
        - predicted_remaining_steps: SRTF using StepMap's predicted remaining steps.
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # --- Collect backend capacities ---
        # Use decay-adjusted capacity when enabled (optimistic for resume)
        backend_caps: list[tuple[BackendState, int]] = []
        total_capacity = 0
        for url, backend in self.backends.items():
            if not backend.cache_config or not backend.healthy:
                continue
            remaining = (backend.remaining_capacity_with_decay()
                         if backend.use_acting_token_decay
                         else backend.remaining_capacity())
            if remaining > BUFFER_PER_PROGRAM:
                backend_caps.append((backend, remaining))
                total_capacity += remaining
        
        if not backend_caps or total_capacity <= 0:
            return {}
        
        async with self.pause_resume_lock:
            paused_programs = self._get_paused_programs_sorted(ascending=True)
            
            if not paused_programs:
                return {}
            
            candidates_by_priority = self._get_resume_candidates(paused_programs)
            
            # --- Step 1: Select max programs fitting within total capacity ---
            # Walk through priority order; include a program if cumulative fits.
            resumable_programs: list[Program] = []
            cumulative_tokens = 0
            for state in candidates_by_priority:
                required_tokens = state.total_tokens + BUFFER_PER_PROGRAM
                if cumulative_tokens + required_tokens <= total_capacity:
                    resumable_programs.append(state)
                    cumulative_tokens += required_tokens
            
            if not resumable_programs:
                return {}
            
            # --- Step 2: BFD placement (largest first → highest capacity backend) ---
            resumable_programs.sort(key=lambda s: -s.total_tokens)  # descending by tokens
            backend_caps.sort(key=lambda x: -x[1])                  # descending by capacity
            
            resumed_count = 0
            pending_resumed_by_backend: Dict[str, int] = {}
            reasoning_resumed = new_resumed = acting_resumed = 0
            min_required_tokens = resumable_programs[-1].total_tokens + BUFFER_PER_PROGRAM
            
            for state in resumable_programs:
                if not backend_caps:
                    break
                
                required_tokens = state.total_tokens + BUFFER_PER_PROGRAM
                max_backend_capacity = backend_caps[0][1]
                
                # Early termination: if even the smallest program
                # can't fit the largest backend, nothing more can be placed.
                if min_required_tokens > max_backend_capacity:
                    break
                
                # Current program too large for the largest backend — skip it,
                # smaller programs following may still fit.
                if required_tokens > max_backend_capacity:
                    continue
                
                # Place on the backend with the most remaining capacity
                target_backend, target_remaining = backend_caps[0]
                has_pending_request = state.status == ProgramStatus.REASONING
                self.global_waiting_queue.pop(state.program_id, None)
                self._resume_program(state, target_backend=target_backend)
                resumed_count += 1
                if has_pending_request:
                    pending_resumed_by_backend[target_backend.url] = (
                        pending_resumed_by_backend.get(target_backend.url, 0) + 1
                    )
                
                # Track which priority group
                if state.step_count == 1:
                    new_resumed += 1
                elif state.status == ProgramStatus.REASONING:
                    reasoning_resumed += 1
                else:
                    acting_resumed += 1
                
                # Update capacity and re-sort backends
                updated_remaining = target_remaining - required_tokens
                if updated_remaining > BUFFER_PER_PROGRAM:
                    backend_caps[0] = (target_backend, updated_remaining)
                    backend_caps.sort(key=lambda x: -x[1])  # re-sort after update
                else:
                    backend_caps.pop(0)  # backend full, remove from candidates
            
            if resumed_count > 0:
                logger.info(
                    f"Scheduler resumed {resumed_count} programs "
                    f"(reasoning={reasoning_resumed}, new={new_resumed}, acting={acting_resumed})"
                )
            return pending_resumed_by_backend

    async def _greedy_resume_specsys(self):
        """Resume waiting programs using Shortest Job First (SJF) based on remaining time.
        
        Algorithm:
        1. Compute remaining token capacity across all healthy backends.
        2. Sort paused programs according to scheduler_policy.
        3. Iterate through the sorted candidates. For each program, find all backends with 
           enough token capacity.
        4. Assign the program to the eligible backend that offers the SHORTEST predicted 
           remaining time for this specific program.
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # --- Collect backend capacities ---
        # Use decay-adjusted capacity when enabled (optimistic for resume)
        backend_caps: list[tuple[BackendState, int]] = []
        total_capacity = 0
        for url, backend in self.backends.items():
            if not backend.cache_config or not backend.healthy:
                continue
            remaining = (backend.remaining_capacity_with_decay()
                         if backend.use_acting_token_decay
                         else backend.remaining_capacity())
            if remaining > BUFFER_PER_PROGRAM:
                backend_caps.append((backend, remaining))
                total_capacity += remaining
        
        if not backend_caps or total_capacity <= 0:
            return
        
        async with self.pause_resume_lock:
            paused_programs = self._get_paused_programs_sorted(ascending=True)
            
            if not paused_programs:
                return
            
            candidates_by_priority = self._get_resume_candidates(paused_programs)
            
            # --- Select max programs fitting within total capacity ---
            resumable_programs: list[Program] = []
            cumulative_tokens = 0
            for state in candidates_by_priority:
                required_tokens = state.total_tokens + BUFFER_PER_PROGRAM
                if cumulative_tokens + required_tokens <= total_capacity:
                    resumable_programs.append(state)
                    cumulative_tokens += required_tokens
            
            if not resumable_programs:
                return
            
            # --- Placement (Shortest Job First + Fastest Backend) ---
            # We iterate in the SJF order. 
            resumed_count = 0
            reasoning_resumed = new_resumed = acting_resumed = 0
            
            for state in resumable_programs:
                if not backend_caps:
                    break
                
                required_tokens = state.total_tokens + BUFFER_PER_PROGRAM
                
                # 1. Find all backends that have enough token capacity for this program
                eligible_backends = [
                    (idx, b_state, b_cap) for idx, (b_state, b_cap) in enumerate(backend_caps)
                    if b_cap >= required_tokens
                ]
                
                if not eligible_backends:
                    # Unlike BFD (which sorts by tokens), we are sorting by TIME.
                    # A later program might have a longer time but SMALLER tokens, so it might fit.
                    # Therefore we use 'continue' instead of 'break'.
                    continue
                
                # 2. Score eligible backends to find the fastest one
                def backend_score(item) -> tuple[float, int]:
                    _, b_state, b_cap = item
                    time_cost = float('inf')
                    # Look up the specific predicted time for this backend
                    if hasattr(state, 'predicted_remaining_time') and state.predicted_remaining_time:
                        time_cost = state.predicted_remaining_time.get(b_state.url, float('inf'))
                    
                    # Sort primarily by time cost (lowest is best)
                    # Break ties by remaining token capacity (highest is best)
                    return (time_cost, -b_cap)
                
                eligible_backends.sort(key=backend_score)
                best_idx, target_backend, target_remaining = eligible_backends[0]
                
                # 3. Resume the program to the chosen optimal backend
                self.global_waiting_queue.pop(state.program_id, None)
                self._resume_program(state, target_backend=target_backend)
                resumed_count += 1
                
                # Track which priority group
                if state.step_count == 1:
                    new_resumed += 1
                elif state.status == ProgramStatus.REASONING:
                    reasoning_resumed += 1
                else:
                    acting_resumed += 1
                
                # 4. Update capacity and maintain backend_caps
                updated_remaining = target_remaining - required_tokens
                if updated_remaining > BUFFER_PER_PROGRAM:
                    backend_caps[best_idx] = (target_backend, updated_remaining)
                else:
                    # Backend is essentially full, remove from available pool
                    backend_caps.pop(best_idx)
            
            if resumed_count > 0:
                logger.info(
                    f"Scheduler resumed {resumed_count} programs "
                    f"(reasoning={reasoning_resumed}, new={new_resumed}, acting={acting_resumed})"
                )

    async def _wait_for_resume(self, program_id: str, state: Program, timeout: float = 1800.0) -> None:
        """Wait for a paused program to be resumed.
        
        If timeout (30 min), force resume the program regardless of capacity.
        """
        if state.waiting_event is None:
            return
        
        try:
            await asyncio.wait_for(state.waiting_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Program {program_id} wait timeout after {timeout}s, forcing resume")
            claimed_state = await self._claim_specific_paused(program_id)
            if claimed_state is None:
                return
            
            # For new programs without origin backend, force assign to least loaded backend
            target_backend = None
            if claimed_state.origin_backend is None:
                target_backend = self.select_backend_for_new_program_default()
                logger.info(f"Force assigning new program {program_id} to {target_backend.url}")
            
            self._resume_program(claimed_state, target_backend=target_backend)

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = len(self.global_waiting_queue)
        marked = sum(1 for p in self.programs.values() if p.marked_for_pause)
        
        # Per-backend stats
        paused_counts = self.get_paused_counts_by_backend()
        per_backend = {}
        for url, backend in self.backends.items():
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": paused_counts.get(url, 0),
                "marked_for_pause": sum(1 for p in progs.values() if p.marked_for_pause),
                "future_paused_tokens": backend.future_paused_tokens,
                "speculative_decoding_enabled": self._sd_enabled_by_backend[url],
                "dynamic_sd_projected_batch": self._sd_projected_load_by_backend[url],
                "dynamic_sd_resumed_this_tick": self._sd_resumed_by_backend[url],
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
            "marked_for_pause": marked,
            "scheduler_policy": self._scheduler_policy,
            "dynamic_sd_enabled": self._dynamic_sd_enabled,
            "sd_enable_threshold": (
                self._sd_enable_threshold if self._dynamic_sd_enabled else None
            ),
            "sd_switch_threshold": (
                self._sd_switch_threshold if self._dynamic_sd_enabled else None
            ),
            "per_backend": per_backend,
        }

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_total_tokens(payload: Any) -> Optional[int]:
        """Extract total_tokens from the response."""
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                return int(val)
        return None

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        on_usage: Callable[[int, Optional[int], Optional[int], Optional[int]], Awaitable[None]] | None = None,
        on_first_token: Callable[[], None] | None = None,
        on_token: Callable[[], None] | None = None,
        on_token_progress: Callable[[int], None] | None = None,
    ) -> Response:
        """Proxy request to a specific backend.
        
        Args:
            backend: Target backend
            payload: Request payload
            on_usage: Callback with (total_tokens, prompt_tokens, completion_tokens, cached_tokens)
            on_first_token: Callback when first token is received (streaming only)
            on_token: Callback for each token (streaming only)
            on_token_progress: Callback with delta token count at intervals (streaming only)
        """
        url = backend.completions_url
        
        if payload.get("stream"):
            return await forward_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
                on_first_token=on_first_token,
                on_token=on_token,
                on_token_progress=on_token_progress,
            )
        else:
            return await forward_non_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
            )

    async def proxy_get(self, backend_url: str, path: str) -> Response:
        """Proxy a GET request to a backend."""
        url = f"{backend_url.rstrip('/')}{path}"
        return await forward_get_request(self.client, url)
