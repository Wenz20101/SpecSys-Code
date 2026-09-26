"""Offline table-driven remaining-step prediction."""
import json
import logging
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import ijson

logger = logging.getLogger(__name__)




class StepInfo:
    def __init__(self):
        self.step_cnt = Counter()
        self.state_step_cnt: Dict[tuple[int, str], Counter] = defaultdict(Counter)
        self.transition_step_cnt: Dict[tuple[int, str, str], Counter] = defaultdict(Counter)
        self.step_time: Dict[str, deque] = {}


class StepMap:
    DEFAULT_TRACE_DIR = Path("/home/user/Trace-Replay/traces/trainset")
    MIN_STATE_SAMPLES = 3
    _trace_history_cache: Dict[str, tuple[int, Dict[str, Dict[str, Any]]]] = {}

    PROBLEM_WORDS = (
        "error",
        "failed",
        "no results",
        "no result",
        "not found",
        "unable",
        "cannot",
        "invalid",
        "exception",
        "traceback",
    )
    POSITIVE_WORDS = (
        "success",
        "passed",
        "verified",
        "sufficient",
        "answer",
        "final",
    )
    TYPE_C_ERROR_WORDS = (
        "error executing",
        "doesn't exist",
        "does not exist",
        "no such",
        "unknown column",
        "syntax error",
        "traceback",
        "exception",
    )
    SCHEMA_WORDS = ("show tables", "desc ", "describe ", "pragma ", "schema")

    @staticmethod
    def _normalize_workflow(value: object, filename: str) -> str:
        workflow = str(value or "").strip().lower().replace("_", "-")
        if workflow in {"plan-solve", "plansolve"}:
            return "plan-solve"
        if workflow in {"react", "re-act"}:
            return "react"
        if "agentflow" in workflow:
            return "agentflow"

        name = filename.lower().replace("_", "-")
        if "plan-solve" in name:
            return "plan-solve"
        if "agentflow" in name:
            return "agentflow"
        if "react" in name or "re-act" in name:
            return "react"
        return "unknown"

    @staticmethod
    def _infer_workload(value: object, filename: str) -> str:
        workload = str(value or "").strip().lower()
        if workload:
            return workload
        name = filename.lower()
        for candidate in ("nl2bash", "spider", "2wiki", "mbpp", "bamboogle"):
            if candidate in name:
                return candidate
        return "unknown"

    @staticmethod
    def _nested(value: object, *keys: str) -> object:
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    @staticmethod
    def _searchable_text(*values: object) -> str:
        return " ".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            for value in values
            if value is not None
        ).lower()

    @classmethod
    def _trace_state_b(cls, step: Dict[str, Any]) -> str:
        parsed = step.get("parsed") or {}
        output = cls._nested(step, "tool_call", "output")
        text = cls._searchable_text(
            parsed,
            cls._nested(output, "observation"),
            cls._nested(output, "raw_observation"),
        )
        if any(word in text for word in cls.PROBLEM_WORDS):
            return "problem"
        if any(word in text for word in cls.POSITIVE_WORDS):
            return "positive"
        return "neutral"

    @classmethod
    def _trace_action(cls, step: Dict[str, Any]) -> str:
        parsed = step.get("parsed") or {}
        return str(parsed.get("action_parsed") or parsed.get("action") or "").strip().lower()

    @classmethod
    def _trace_state_c(
        cls,
        step: Dict[str, Any],
        previous_step: Optional[Dict[str, Any]],
    ) -> str:
        output = cls._nested(step, "tool_call", "output")
        action = cls._trace_action(step)
        observation = cls._searchable_text(
            cls._nested(output, "observation"),
            cls._nested(output, "raw_observation"),
        )
        valid_action = cls._nested(output, "valid_action")
        done = cls._nested(output, "done")
        if done is True or "submit" in action or "final" in action:
            return "final_or_submit"
        if valid_action is False or any(word in observation for word in cls.TYPE_C_ERROR_WORDS):
            return "error"
        if previous_step is not None and action and action == cls._trace_action(previous_step):
            return "repeat"
        if any(word in action for word in cls.SCHEMA_WORDS):
            return "schema_lookup"
        if action:
            return "query_or_command"
        return "normal"

    @classmethod
    def _states_from_trace_steps(
        cls,
        workflow: str,
        steps: List[Dict[str, Any]],
        total_steps: int,
    ) -> List[str]:
        states = ["start"]
        for prefix_k in range(1, total_steps):
            step_index = prefix_k - 1
            if step_index >= len(steps) or not isinstance(steps[step_index], dict):
                states.append("normal")
                continue
            step = steps[step_index]
            if workflow == "agentflow":
                states.append(cls._trace_state_b(step))
            else:
                previous_step = steps[step_index - 1] if step_index > 0 else None
                states.append(cls._trace_state_c(step, previous_step))
        return states

    @staticmethod
    def _message_text(message: Dict[str, Any]) -> str:
        values = [message.get("content"), message.get("tool_calls")]
        return StepMap._searchable_text(*values)

    @staticmethod
    def _message_content(message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            current = {
                key: content[key]
                for key in ("context", "sub_goal", "tool_name", "result", "observation", "status")
                if key in content
            }
            action_steps: List[tuple[int, object]] = []
            containers = [content]
            if isinstance(content.get("memory"), dict):
                containers.append(content["memory"])
            for container in containers:
                for key, value in container.items():
                    match = re.fullmatch(r"Action Step\s+(\d+)", str(key), re.IGNORECASE)
                    if match:
                        action_steps.append((int(match.group(1)), value))
            if action_steps:
                current["latest_action_step"] = max(action_steps, key=lambda item: item[0])[1]
            if current:
                content = current
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def _action_observation_from_prompt(cls, prompt: str) -> tuple[str, str, str]:
        """Parse the latest ReAct action/observation from a flattened prompt."""
        actions = re.findall(r"(?im)^Action\s*\d*\s*:\s*(.*)$", prompt)
        observations = re.findall(
            r"(?ims)^Observation\s*\d*\s*:\s*(.*?)"
            r"(?=^(?:Thought|Action|Question)\s*\d*\s*:|\Z)",
            prompt,
        )
        action = actions[-1].strip().lower() if actions else ""
        previous_action = actions[-2].strip().lower() if len(actions) >= 2 else ""
        observation = observations[-1].strip().lower() if observations else ""
        return action, observation, previous_action

    @classmethod
    def prediction_context_from_payload(
        cls,
        payload: Dict[str, Any],
        agent_type: str,
        prefix_states: List[str],
    ) -> Dict[str, Any]:
        """Extract the latest completed action/observation from chat messages."""
        explicit = payload.get("prediction_context")
        extra_body = payload.get("extra_body")
        if not isinstance(explicit, dict) and isinstance(extra_body, dict):
            explicit = extra_body.get("prediction_context")
        context = dict(explicit) if isinstance(explicit, dict) else {}
        if context.get("state_bucket"):
            return context

        messages = payload.get("messages")
        if not isinstance(messages, list):
            context["state_bucket"] = "start" if not prefix_states else "normal"
            return context

        assistants = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        if not assistants:
            if not prefix_states:
                context["state_bucket"] = "start"
                return context
            latest_message = next(
                (message for message in reversed(messages) if isinstance(message, dict)),
                {},
            )
            prompt = cls._message_content(latest_message)
            action, observation, previous_action = cls._action_observation_from_prompt(prompt)
            if not action and not observation:
                observation = prompt.lower()
        else:
            last_assistant = assistants[-1]
            action = cls._message_text(last_assistant)
            previous_action = cls._message_text(assistants[-2]) if len(assistants) >= 2 else ""
            assistant_index = max(
                index for index, message in enumerate(messages) if message is last_assistant
            )
            observations = [
                cls._message_text(message)
                for message in messages[assistant_index + 1 :]
                if isinstance(message, dict) and message.get("role") in {"tool", "user"}
            ]
            observation = " ".join(observations)
        context.update(
            {
                "action": action,
                "observation": observation,
                "previous_action": previous_action,
            }
        )

        normalized_type = cls.normalize_agent_type(agent_type)
        combined = f"{action} {observation}".lower()
        if normalized_type == "B":
            if any(word in combined for word in cls.PROBLEM_WORDS):
                state_bucket = "problem"
            elif any(word in combined for word in cls.POSITIVE_WORDS):
                state_bucket = "positive"
            else:
                state_bucket = "neutral"
        else:
            if "submit" in action or "final" in action:
                state_bucket = "final_or_submit"
            elif any(word in observation for word in cls.TYPE_C_ERROR_WORDS):
                state_bucket = "error"
            elif action and action == previous_action:
                state_bucket = "repeat"
            elif any(word in action for word in cls.SCHEMA_WORDS):
                state_bucket = "schema_lookup"
            elif action:
                state_bucket = "query_or_command"
            else:
                state_bucket = "normal"
        context["state_bucket"] = state_bucket
        return context

    def load_data_from_traces(self, trace_dir: Path) -> tuple[int, int]:
        """Load endpoint and prefix-state tables from trainset trace JSON."""
        if not trace_dir.exists() or not trace_dir.is_dir():
            return 0, 0

        cache_key = str(trace_dir.resolve())
        cached = self._trace_history_cache.get(cache_key)
        if cached is not None:
            loaded_files, cached_history = cached
            for type_id, history in cached_history.items():
                step_info = self.map.setdefault(type_id, StepInfo())
                step_info.step_cnt.update(history["step_cnt"])
                for key, counts in history["state_step_cnt"].items():
                    step_info.state_step_cnt[key].update(counts)
                for key, counts in history["transition_step_cnt"].items():
                    step_info.transition_step_cnt[key].update(counts)
            return loaded_files, sum(
                sum(history["step_cnt"].values())
                for history in cached_history.values()
            )

        loaded_files = 0
        loaded_tasks = 0
        for file_path in sorted(trace_dir.glob("*.json")):
            try:
                with file_path.open("rb") as handle:
                    args = next(ijson.items(handle, "args"), {})
            except (OSError, ijson.JSONError) as exc:
                logger.warning("Failed to load step history from %s: %s", file_path, exc)
                continue

            args = args or {}
            if not isinstance(args, dict):
                args = {}
            workflow = self._normalize_workflow(
                args.get("agent") or args.get("workflow"),
                file_path.stem,
            )
            workload = self._infer_workload(args.get("workload"), file_path.stem)
            if workflow == "unknown" or workload == "unknown":
                logger.warning("Cannot infer workflow/workload from trace %s", file_path)
                continue

            type_id = f"workflow[{workflow}]_workload[{workload}]"
            file_counts: Counter = Counter()
            task_totals: List[Optional[int]] = []
            try:
                with file_path.open("rb") as handle:
                    summaries = ijson.items(handle, "tasks.item.summary")
                    for summary in summaries:
                        if not isinstance(summary, dict):
                            task_totals.append(None)
                            continue
                        try:
                            total_steps = int(summary.get("turns_taken"))
                        except (TypeError, ValueError):
                            task_totals.append(None)
                            continue
                        if total_steps <= 0:
                            task_totals.append(None)
                            continue
                        task_totals.append(total_steps)
                        file_counts[total_steps] += 1
            except (OSError, ijson.JSONError) as exc:
                logger.warning("Failed to stream task summaries from %s: %s", file_path, exc)
                continue

            file_state_counts: Dict[tuple[int, str], Counter] = defaultdict(Counter)
            file_transition_counts: Dict[tuple[int, str, str], Counter] = defaultdict(Counter)
            if workflow in {"agentflow", "react"}:
                try:
                    with file_path.open("rb") as handle:
                        task_steps = ijson.items(handle, "tasks.item.steps")
                        for task_index, steps in enumerate(task_steps):
                            if task_index >= len(task_totals):
                                break
                            total_steps = task_totals[task_index]
                            if total_steps is None or not isinstance(steps, list):
                                continue
                            states = self._states_from_trace_steps(
                                workflow,
                                steps,
                                total_steps,
                            )
                            for prefix_k, state_bucket in enumerate(states):
                                file_state_counts[(prefix_k, state_bucket)][total_steps] += 1
                                previous_state = states[prefix_k - 1] if prefix_k else "bos"
                                file_transition_counts[
                                    (prefix_k, previous_state, state_bucket)
                                ][total_steps] += 1
                except (OSError, ijson.JSONError) as exc:
                    logger.warning(
                        "Failed to stream task steps from %s; using endpoint table only: %s",
                        file_path,
                        exc,
                    )
                    file_state_counts.clear()
                    file_transition_counts.clear()

            file_tasks = sum(file_counts.values())
            if file_tasks:
                step_info = self.map.setdefault(type_id, StepInfo())
                step_info.step_cnt.update(file_counts)
                for key, counts in file_state_counts.items():
                    step_info.state_step_cnt[key].update(counts)
                for key, counts in file_transition_counts.items():
                    step_info.transition_step_cnt[key].update(counts)
                loaded_files += 1
                loaded_tasks += file_tasks
                logger.info(
                    "Loaded %d step-history tasks for %s from %s",
                    file_tasks,
                    type_id,
                    file_path,
                )
        if loaded_tasks:
            self._trace_history_cache[cache_key] = (
                loaded_files,
                {
                    type_id: {
                        "step_cnt": Counter(step_info.step_cnt),
                        "state_step_cnt": {
                            key: Counter(counts)
                            for key, counts in step_info.state_step_cnt.items()
                        },
                        "transition_step_cnt": {
                            key: Counter(counts)
                            for key, counts in step_info.transition_step_cnt.items()
                        },
                    }
                    for type_id, step_info in self.map.items()
                },
            )
        return loaded_files, loaded_tasks

    def __init__(
        self,
        trace_dir: Path = DEFAULT_TRACE_DIR,
        *,
        load_history: bool = True,
    ):
        self.map: Dict[str, StepInfo] = {}
        self.history_source = "none"
        if not load_history:
            return
        loaded_files, loaded_tasks = self.load_data_from_traces(Path(trace_dir))
        if loaded_tasks:
            self.history_source = str(trace_dir)
            logger.info(
                "Loaded step prediction history from %d trace files (%d tasks)",
                loaded_files,
                loaded_tasks,
            )

    def update_step_cnt(
        self,
        type_id: str,
        end_step: int,
        *,
        agent_type: str = "unknown",
        prefix_states: Optional[List[str]] = None,
    ) -> None:
        if type_id not in self.map:
            self.map[type_id] = StepInfo()
        step_info = self.map[type_id]
        step_info.step_cnt[end_step] += 1
        if self.normalize_agent_type(agent_type) not in {"B", "C"}:
            return
        states = prefix_states or []
        for prefix_k, state_bucket in enumerate(states[:end_step]):
            step_info.state_step_cnt[(prefix_k, state_bucket)][end_step] += 1
            previous_state = states[prefix_k - 1] if prefix_k else "bos"
            step_info.transition_step_cnt[
                (prefix_k, previous_state, state_bucket)
            ][end_step] += 1
        # logger.info(f"[DEBUG] update_step_cnt! {step_info.step_cnt}")

    def get_max_prob_step(self, type_id, cur_step):
        step_cnt = self.get_conditional_step_counts(type_id, cur_step)
        if not step_cnt:
            return None
        total_weighted_steps = 0
        total_counts = 0
        for step, count in step_cnt.items():
            total_weighted_steps += step * count
            total_counts += count
        if total_counts == 0:
            return None
        return total_weighted_steps / total_counts

    def get_conditional_step_counts(self, type_id: str, cur_step: int) -> Counter:
        """Return historical endpoints still possible at the current prefix."""
        step_info = self.map.get(type_id)
        if step_info is None:
            return Counter()
        return Counter(
            {
                step: count
                for step, count in step_info.step_cnt.items()
                if step >= cur_step and count > 0
            }
        )

    def get_conditional_state_counts(
        self,
        type_id: str,
        cur_step: int,
        prefix_k: int,
        state_bucket: str,
        previous_state: Optional[str] = None,
    ) -> Counter:
        """Return endpoints for one state cell, excluding already impossible T."""
        step_info = self.map.get(type_id)
        if step_info is None:
            return Counter()
        if previous_state is None:
            counts = step_info.state_step_cnt.get((prefix_k, state_bucket), Counter())
        else:
            counts = step_info.transition_step_cnt.get(
                (prefix_k, previous_state, state_bucket),
                Counter(),
            )
        return Counter(
            {
                step: count
                for step, count in counts.items()
                if step >= cur_step and count > 0
            }
        )

    @staticmethod
    def normalize_agent_type(agent_type: object) -> str:
        value = str(agent_type or "").strip().lower().replace("_", "-")
        aliases = {
            "a": "A",
            "type-a": "A",
            "planning": "A",
            "plan": "A",
            "b": "B",
            "type-b": "B",
            "macro-round": "B",
            "macro": "B",
            "c": "C",
            "type-c": "C",
            "micro-step": "C",
            "micro": "C",
        }
        return aliases.get(value, "unknown")

    @staticmethod
    def conditional_mode(step_cnt: Counter) -> Optional[float]:
        if not step_cnt:
            return None
        highest = max(step_cnt.values())
        return float(min(step for step, count in step_cnt.items() if count == highest))

    @staticmethod
    def conditional_median(step_cnt: Counter) -> Optional[float]:
        if not step_cnt:
            return None
        values: list[int] = []
        for step, count in step_cnt.items():
            values.extend([step] * count)
        return float(statistics.median(values))

    def predict_remaining_steps(
        self,
        type_id: str,
        cur_step: int,
        *,
        agent_type: str = "unknown",
        num_plan_steps: Optional[int] = None,
        prefix_k: Optional[int] = None,
        state_bucket: Optional[str] = None,
        previous_state: Optional[str] = None,
    ) -> Optional[float]:
        """Predict T - k with the estimator selected by Type-A/B/C."""
        normalized_type = self.normalize_agent_type(agent_type)
        if normalized_type == "A":
            if num_plan_steps is None or num_plan_steps <= 0:
                return None
            return float(max(0, num_plan_steps - cur_step))

        if normalized_type == "B":
            step_cnt = Counter()
            if prefix_k is not None and state_bucket:
                candidate = self.get_conditional_state_counts(
                    type_id,
                    cur_step,
                    prefix_k,
                    state_bucket,
                )
                if sum(candidate.values()) >= self.MIN_STATE_SAMPLES:
                    step_cnt = candidate
            if not step_cnt:
                step_cnt = self.get_conditional_step_counts(type_id, cur_step)
            end_step = self.conditional_mode(step_cnt)
        elif normalized_type == "C":
            step_cnt = Counter()
            if prefix_k is not None and state_bucket and previous_state:
                candidate = self.get_conditional_state_counts(
                    type_id,
                    cur_step,
                    prefix_k,
                    state_bucket,
                    previous_state,
                )
                if sum(candidate.values()) >= self.MIN_STATE_SAMPLES:
                    step_cnt = candidate
            if not step_cnt and prefix_k is not None and state_bucket:
                candidate = self.get_conditional_state_counts(
                    type_id,
                    cur_step,
                    prefix_k,
                    state_bucket,
                )
                if sum(candidate.values()) >= self.MIN_STATE_SAMPLES:
                    step_cnt = candidate
            if not step_cnt:
                step_cnt = self.get_conditional_step_counts(type_id, cur_step)
            end_step = self.conditional_median(step_cnt)
        else:
            step_cnt = self.get_conditional_step_counts(type_id, cur_step)
            end_step = self.get_max_prob_step(type_id, cur_step)
        if end_step is None:
            return None
        return max(0.0, end_step - cur_step)

    def update_step_time(self, type_id, last_step_time, backend_url):
        if type_id not in self.map:
            self.map[type_id] = StepInfo()
        step_info = self.map[type_id]
        step_time = step_info.step_time
        step_time.setdefault(backend_url, deque(maxlen=10)).append(last_step_time)
        # logger.info(f"[DEBUG] update_step_time! {step_info.step_time}")

    def get_avg_step_time(self, type_id, backend_url):
        if type_id not in self.map:
            return None
        step_info = self.map[type_id]
        if backend_url not in step_info.step_time:
            return None
        step_time_list = step_info.step_time[backend_url]
        if not step_time_list or len(step_time_list) == 0:
            return None
        return sum(step_time_list) / len(step_time_list)

    def predict_remaining_time(
        self,
        type_id,
        cur_step,
        backend_urls,
        *,
        agent_type="unknown",
        num_plan_steps=None,
        prefix_k=None,
        state_bucket=None,
        previous_state=None,
    ):
        predicted_steps = self.predict_remaining_steps(
            type_id,
            cur_step,
            agent_type=agent_type,
            num_plan_steps=num_plan_steps,
            prefix_k=prefix_k,
            state_bucket=state_bucket,
            previous_state=previous_state,
        )
        if predicted_steps is None:
            return None
        # Time prediction is made before the current request executes, so it
        # includes that request even though the SRTF priority uses T - k.
        remaining_steps = predicted_steps + 1

        backend_data = {}
        for backend_url in backend_urls:
            avg_time = self.get_avg_step_time(type_id, backend_url)
            if avg_time is not None:
                # 将 avg_time 和计算出的 remaining_time 一起存入
                backend_data[backend_url] = {
                    "avg_time": avg_time,
                    "remaining_time": remaining_steps * avg_time
                }

        if not backend_data:
            return None

        # logger.info(f"[DEBUG] remaining_steps:{remaining_steps}, backend_data:{backend_data}")

        return {
            "remaining_steps": remaining_steps,
            "backends": backend_data
        }
