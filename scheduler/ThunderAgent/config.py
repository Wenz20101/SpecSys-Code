"""ThunderAgent configuration."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """ThunderAgent configuration (set via command line args)."""
    # Backend configuration
    backends: List[str] = field(default_factory=lambda: ["http://localhost:8000"])
    
    # Router mode: "default" (pure proxy) or "tr" (capacity scheduling)
    router_mode: str = "tr"

    # Backend type: "vllm", "sglang", or "skyrl"
    backend_type: str = "vllm"
    
    # Profile configuration
    profile_enabled: bool = False
    profile_dir: str = "/tmp/thunderagent_profiles"
    
    # Metrics monitoring configuration
    metrics_enabled: bool = False
    metrics_interval: float = 5.0  # seconds between metrics fetch

    # Inference-task count sampling. Disabled when no output file is configured.
    llm_task_stats_interval: float = 1.0
    llm_task_stats_file: Optional[str] = None
    
    # Scheduler configuration
    scheduler_interval: float = 5.0  # seconds between scheduler checks
    # original, remaining_steps, predicted_remaining_steps
    scheduler_policy: str = "original"
    dynamic_sd_enabled: bool = False
    # Disable SD at this projected batch size; enable again below 75% of it.
    sd_switch_threshold: int = 64
    acting_token_weight: float = 1.0  # weight for acting tokens in capacity calculation
    use_acting_token_decay: bool = False  # use 2^(-t) decay for acting tokens in resume logic


# Global config instance (set by __main__.py before app starts)
_config: Config = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
