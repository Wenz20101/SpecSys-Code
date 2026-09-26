"""ThunderAgent entry point for `python -m ThunderAgent`."""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ThunderAgent - Program State Tracking Proxy for vLLM",
        prog="python -m ThunderAgent",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--backends", default="http://172.16.33.142:8001",
                        help="Comma-separated list of vLLM backend URLs")
    parser.add_argument("--router", default="tr", choices=["default", "tr"],
                        help="Router mode: 'default' (pure proxy) or 'tr' (capacity scheduling)")
    parser.add_argument("--backend-type", default="vllm", choices=["vllm", "sglang", "skyrl"],
                        help="Backend type: 'vllm', 'sglang', or 'skyrl'")
    parser.add_argument("--profile", action="store_true", 
                        help="Enable profiling (track prefill/decode/tool_call times)")
    parser.add_argument("--profile-dir", default="/tmp/thunderagent_profiles", 
                        help="Directory for profile CSV output")
    parser.add_argument("--metrics", action="store_true",
                        help="Enable vLLM metrics monitoring")
    parser.add_argument("--metrics-interval", type=float, default=5.0,
                        help="Interval in seconds between metrics fetches (default: 5.0)")
    parser.add_argument(
        "--llm-task-stats-interval",
        type=float,
        default=1.0,
        help="Interval in seconds between active LLM task-count samples (default: 1.0)",
    )
    parser.add_argument(
        "--llm-task-stats-file",
        default=None,
        help="CSV output path for LLM task-count samples; disabled when omitted",
    )
    parser.add_argument("--scheduler-interval", type=float, default=5.0,
                        help="Interval in seconds between scheduler checks (default: 5.0)")
    parser.add_argument(
        "--scheduler-policy",
        default="original",
        choices=["original", "remaining_steps", "predicted_remaining_steps"],
        help=(
            "Scheduler policy: 'original', ideal 'remaining_steps', or "
            "'predicted_remaining_steps' (default: original)"
        ),
    )
    parser.add_argument(
        "--dynamic-sd",
        action="store_true",
        help="Dynamically switch vLLM speculative decoding based on request load",
    )
    parser.add_argument(
        "--sd-switch-threshold",
        type=int,
        default=64,
        help=(
            "Disable speculative decoding when the projected next running batch "
            "reaches this value; re-enable below 75%% of it (default: 64)"
        ),
    )
    parser.add_argument("--acting-token-weight", type=float, default=1.0,
                        help="Weight for acting tokens in capacity calculation (default: 1.0)")
    parser.add_argument("--use-acting-token-decay", action="store_true",
                        help="Use 2^(-t) decay for acting tokens in resume capacity calculation")
    args = parser.parse_args()

    # Set config BEFORE importing app
    from .config import Config, set_config
    
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    config = Config(
        backends=backends,
        router_mode=args.router,
        backend_type=args.backend_type,
        profile_enabled=args.profile,
        profile_dir=args.profile_dir,
        metrics_enabled=args.metrics,
        metrics_interval=args.metrics_interval,
        llm_task_stats_interval=args.llm_task_stats_interval,
        llm_task_stats_file=args.llm_task_stats_file,
        scheduler_interval=args.scheduler_interval,
        scheduler_policy=args.scheduler_policy,
        dynamic_sd_enabled=args.dynamic_sd,
        sd_switch_threshold=args.sd_switch_threshold,
        acting_token_weight=args.acting_token_weight,
        use_acting_token_decay=args.use_acting_token_decay,
    )
    set_config(config)
    
    print(f"🚀 Router mode: {args.router}")
    if args.profile:
        print(f"📊 Profiling enabled - CSV output: {args.profile_dir}/step_profiles.csv")
    
    if args.metrics:
        print(f"📈 Metrics monitoring enabled - interval: {args.metrics_interval}s")

    if args.llm_task_stats_file:
        print(
            "LLM task-count sampling enabled - "
            f"interval: {args.llm_task_stats_interval}s, "
            f"output: {args.llm_task_stats_file}"
        )
    
    if args.router == "tr":
        print(f"⏱️  Scheduler interval: {args.scheduler_interval}s")
        print(f"🧭 Scheduler policy: {args.scheduler_policy}")
        print(f"⚖️  Acting token weight: {args.acting_token_weight}")
        if args.use_acting_token_decay:
            print(f"📉 Acting token decay: enabled (2^-t)")
    if args.dynamic_sd:
        print(f"Dynamic SD enabled - switch threshold: {args.sd_switch_threshold}")

    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn", file=sys.stderr)
        return 1

    # Import app after config is set
    from .app import create_app
    app_instance = create_app(config_override=config)
    uvicorn.run(app_instance, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
