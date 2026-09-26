"""ThunderAgent FastAPI application entry point."""
import logging
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Config, get_config
from .scheduler import MultiBackendRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_program_id(payload: Dict[str, Any], headers: Optional[Mapping[str, str]] = None) -> str:
    """Extract program_id from the request.

    Resolution order:
      1. ``payload["program_id"]``
      2. ``payload["extra_body"]["program_id"]``
      3. ``X-Session-ID`` request header (case-insensitive), if ``headers`` is
         provided. This lets clients that already route by session id (e.g.
         SkyRL) reuse the session id as the ThunderAgent program id without
         needing to inject ``program_id`` into the payload.
      4. ``"default"`` as a last-resort fallback.
    """
    if "program_id" in payload:
        return str(payload["program_id"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "program_id" in extra_body:
        return str(extra_body["program_id"])
    if headers is not None:
        session_id = headers.get("X-Session-ID") or headers.get("x-session-id")
        if session_id:
            return str(session_id)
    return "default"


def get_type_id(payload: Dict[str, Any]) -> str:
    """
    Extract type_id (Agent WorkFlow & Agent WorkLoad) from the request.
    Resolution order:
    1. ``payload["workflow"] & payload["workload"]``
    2. ``payload["extra_body"]["workflow"] &  payload["extra_body"]["workload"]``
    3. ``"default"`` as a last-resort fallback.
    """
    def format_type_id(workflow: object, workload: object) -> str:
        workflow_name = str(workflow).strip().lower().replace("_", "-")
        if workflow_name in {"react", "re-act"}:
            workflow_name = "react"
        elif workflow_name in {"plan-solve", "plansolve"}:
            workflow_name = "plan-solve"
        workload_name = str(workload).strip().lower()
        return f"workflow[{workflow_name}]_workload[{workload_name}]"

    type_id = "default"
    if "workflow" in payload and "workload" in payload:
        type_id = format_type_id(payload["workflow"], payload["workload"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "workflow" in extra_body and "workload" in extra_body:
        type_id = format_type_id(extra_body["workflow"], extra_body["workload"])
    return type_id


def get_agent_type(payload: Dict[str, Any]) -> str:
    """Extract Type-A/B/C metadata from the payload or extra_body."""
    extra_body = payload.get("extra_body", {})
    sources = [payload]
    if isinstance(extra_body, dict):
        sources.append(extra_body)
    for source in sources:
        for key in ("agent_type", "workflow_type", "type"):
            value = source.get(key)
            if value is not None:
                return str(value)
    return "unknown"


def get_num_plan_steps(payload: Dict[str, Any]) -> Optional[int]:
    """Extract the Type-A plan length without using ideal total_steps."""
    extra_body = payload.get("extra_body", {})
    sources = [payload]
    if isinstance(extra_body, dict):
        sources.append(extra_body)
    for source in sources:
        value = source.get("num_plan_steps")
        if value is None:
            context = source.get("prediction_context")
            if isinstance(context, dict):
                value = context.get("num_plan_steps")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None

def get_total_steps(payload: Dict[str, Any]) -> int:
    """
    Extract total_steps from the request.
    Resolution order:
    1. ``payload["total_steps"]``
    2. ``payload["extra_body"]["total_steps"]``
    3. ``0`` when the value is invalid.
    """
    total_steps = 10
    if "total_steps" in payload:
        total_steps = payload['total_steps']
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "total_steps" in extra_body:
        total_steps = extra_body['total_steps']
    try:
        return int(total_steps)
    except (TypeError, ValueError):
        return 0


def register_routes(app: FastAPI, ta_router: MultiBackendRouter, config: Optional[Config] = None) -> None:
    """Register all ThunderAgent HTTP routes on the given FastAPI app.

    This is the shared route bundle used by both the standalone ThunderAgent
    server (see bottom of this module) and external FastAPI apps that embed
    ThunderAgent (e.g. an RL training framework that wraps the router with
    its own integration layer).

    Args:
        app: FastAPI application to register routes on.
        ta_router: ThunderAgent MultiBackendRouter instance. The caller is
            responsible for ``await ta_router.start()`` / ``stop()`` on the
            app lifecycle.
        config: Optional config override. If None, uses ``get_config()`` at
            request time.
    """

    def _get_config() -> Config:
        return config if config is not None else get_config()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Handle chat completions request."""
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        # Get or create program state (new programs go to waiting queue)
        program_id = get_program_id(payload, request.headers)
        type_id = get_type_id(payload)
        agent_type = get_agent_type(payload)
        num_plan_steps = get_num_plan_steps(payload)
        total_steps = get_total_steps(payload)
        program_state = ta_router.get_or_create_program(
            program_id,
            type_id,
            total_steps,
            agent_type=agent_type,
            num_plan_steps=num_plan_steps,
        )

        # Profile: record request arrival BEFORE pause check (for accurate tool_call_time)
        if program_state.profile:
            program_state.profile.on_request_arrive()

        # Update state: check capacity, may pause and wait (max 20 min, then force resume)
        await ta_router.update_program_before_request(program_id, program_state, payload)

        # Profile: record request start AFTER pause (captures pause_time)
        if program_state.profile:
            program_state.profile.on_request_start()

        # Resolve backend after any pause/resume to honor migrations.
        backend = ta_router.get_backend_for_program(program_id)

        # Callback to update state after response
        async def on_usage(
            total_tokens: int,
            prompt_tokens: int | None,
            completion_tokens: int | None,
            cached_tokens: int | None,
        ) -> None:
            ta_router.update_program_after_request(
                program_id,
                program_state,
                total_tokens,
                prompt_tokens or 0,
            )
            # Profile: record request end with KV cache info
            if program_state.profile:
                program_state.profile.on_request_end(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                )

        # Callback for streaming token progress updates (every 20 tokens)
        def on_token_progress(delta_tokens: int) -> None:
            ta_router.update_program_tokens_streaming(program_state, delta_tokens)

        # Forward to vLLM (sticky unless rescheduled from the global paused pool)
        # Pass profile callbacks for token timing
        return await ta_router.proxy_request(
            backend, payload,
            on_usage=on_usage,
            on_first_token=program_state.profile.on_first_token if program_state.profile else None,
            on_token=program_state.profile.on_token if program_state.profile else None,
            on_token_progress=on_token_progress,
        )

    @app.get("/programs")
    async def list_programs():
        """List all programs (includes profile data if profiling enabled)."""
        result = {}
        for pid, state in ta_router.programs.items():
            program_data = {
                "backend": state.backend_url,
                "agent_type": state.agent_type,
                "type_id": state.type_id,
                "num_plan_steps": state.num_plan_steps,
                "predicted_remaining_steps": state.predicted_remaining_steps,
                "prefix_states": state.prefix_states,
                "prediction_context": state.prediction_context,
                "context_len": state.context_len,
                "total_tokens": state.total_tokens,
                "step_count": state.step_count,
                "status": state.status.value,
                "state": state.state.value,
            }
            # Include profile data if available
            if state.profile:
                program_data["profile"] = state.profile.to_dict()
            result[pid] = program_data
        return JSONResponse(result)

    @app.post("/programs/release")
    async def release_program(request: Request):
        """Release a program."""
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc

        program_id = payload.get("program_id")
        if not program_id:
            raise HTTPException(status_code=400, detail="Missing program_id")
        program_id = str(program_id)

        released = await ta_router.release_program(program_id)
        return JSONResponse({"program_id": program_id, "released": released})

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        cfg = _get_config()
        stats = ta_router.get_program_stats()
        return JSONResponse({
            "status": "ok",
            "router_mode": cfg.router_mode,
            "scheduling_enabled": ta_router.scheduling_enabled,
            "scheduler_policy": stats["scheduler_policy"],
            "dynamic_sd_enabled": stats["dynamic_sd_enabled"],
            "sd_enable_threshold": stats["sd_enable_threshold"],
            "sd_switch_threshold": stats["sd_switch_threshold"],
            "backends": list(ta_router.backends.keys()),
            "programs_count": stats["total"],
            "reasoning_count": stats["reasoning"],
            "acting_count": stats["acting"],
            "paused_count": stats["paused"],
            "per_backend": stats["per_backend"],
            "profile_enabled": ta_router.profile_enabled,
            "llm_task_stats_enabled": cfg.llm_task_stats_file is not None,
            "llm_task_stats_interval": cfg.llm_task_stats_interval,
            "llm_task_stats_file": cfg.llm_task_stats_file,
            "step_history_source": ta_router.step_map.history_source,
            "step_history_types": len(ta_router.step_map.map),
            "step_history_tasks": sum(
                sum(info.step_cnt.values())
                for info in ta_router.step_map.map.values()
            ),
        })

    @app.get("/profiles")
    async def list_profiles():
        """List all program profiles (timing metrics)."""
        if not ta_router.profile_enabled:
            return JSONResponse({"error": "Profiling not enabled. Start with --profile flag."}, status_code=400)
        result = {}
        for pid, state in ta_router.programs.items():
            if state.profile:
                result[pid] = state.profile.to_dict()
        return JSONResponse(result)

    @app.get("/profiles/{program_id}")
    async def get_profile(program_id: str):
        """Get profile for a specific program."""
        if not ta_router.profile_enabled:
            return JSONResponse({"error": "Profiling not enabled. Start with --profile flag."}, status_code=400)
        state = ta_router.programs.get(program_id)
        if state is None or state.profile is None:
            raise HTTPException(status_code=404, detail=f"Profile not found for program: {program_id}")
        return JSONResponse(state.profile.to_dict())

    @app.get("/v1/models")
    async def list_models():
        """List available models (proxy to first backend)."""
        # Forward to the first available backend
        if not ta_router.backends:
            return JSONResponse({"object": "list", "data": []})

        backend_url = next(iter(ta_router.backends.keys()))
        return await ta_router.proxy_get(backend_url, "/models")

    @app.get("/metrics")
    async def get_metrics():
        """Get vLLM metrics from all backends."""
        cfg = _get_config()
        paused_counts = ta_router.get_paused_counts_by_backend()
        return JSONResponse({
            "metrics_enabled": cfg.metrics_enabled,
            "metrics_interval": cfg.metrics_interval if cfg.metrics_enabled else None,
            "backends": {
                url: backend.to_dict(paused_program_count=paused_counts.get(url, 0))
                for url, backend in ta_router.backends.items()
            },
        })

    # -- Weight sync coordination --

    @app.post("/weight_sync/begin")
    async def weight_sync_begin():
        """Notify ThunderAgent that weight sync is starting. Holds new requests."""
        await ta_router.begin_weight_sync()
        return JSONResponse({"status": "ok", "weight_sync_active": True})

    @app.post("/weight_sync/end")
    async def weight_sync_end():
        """Notify ThunderAgent that weight sync has completed. Releases held requests."""
        await ta_router.end_weight_sync()
        return JSONResponse({"status": "ok", "weight_sync_active": False})


# =============================================================================
# Application Factory (Standalone Mode)
# =============================================================================

def create_app(config_override: Optional[Config] = None) -> FastAPI:
    """Factory function to dynamically create the FastAPI app."""
    # 获取最新的配置
    cfg = config_override if config_override is not None else get_config()
    
    # 根据最新配置创建 Router
    ta_router = MultiBackendRouter(
        cfg.backends,
        profile_enabled=cfg.profile_enabled,
        scheduling_enabled=(cfg.router_mode == "tr"),
        scheduler_interval=cfg.scheduler_interval,
        scheduler_policy=cfg.scheduler_policy,
        dynamic_sd_enabled=cfg.dynamic_sd_enabled,
        sd_switch_threshold=cfg.sd_switch_threshold,
        backend_type=cfg.backend_type,
        acting_token_weight=cfg.acting_token_weight,
        use_acting_token_decay=cfg.use_acting_token_decay,
        llm_task_stats_interval=cfg.llm_task_stats_interval,
        llm_task_stats_file=cfg.llm_task_stats_file,
    )

    app = FastAPI(title="ThunderAgent - Program State Tracking Proxy")

    @app.on_event("startup")
    async def startup_event():
        await ta_router.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        await ta_router.stop()

    # 注册路由并注入配置
    register_routes(app, ta_router, config=cfg)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="info")
