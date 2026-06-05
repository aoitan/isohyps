from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class QueryClient(Protocol):
    def query(self, prompt: str) -> str:
        ...


SAFE_BUILTINS = {
    "Exception": Exception,
    "False": False,
    "None": None,
    "RuntimeError": RuntimeError,
    "True": True,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


@dataclass
class ExecutionObservation:
    stdout: str
    error: str | None
    state: dict[str, str]
    finished: bool
    result: Any

    def to_prompt(self) -> str:
        state_lines = "\n".join(f"- {name}: {value}" for name, value in sorted(self.state.items()))
        return (
            f"stdout:\n{self.stdout or '(empty)'}\n\n"
            f"error:\n{self.error or '(none)'}\n\n"
            f"state:\n{state_lines or '(empty)'}\n\n"
            f"finished: {self.finished}\n"
            f"result: {self.result!r}"
        )


@dataclass
class ControllerResult:
    result: Any
    steps: list[ExecutionObservation]


class LocalREPL:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.finished = False
        self.result: Any = None
        self._globals: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        self._helper_names = {"list_dir", "read_text", "finish"}
        self._globals.update(
            {
                "list_dir": self.list_dir,
                "read_text": self.read_text,
                "finish": self.finish,
            }
        )

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = (self.root / Path(path)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes root: {path}")
        return candidate

    def list_dir(self, path: str = ".") -> list[str]:
        target = self._resolve_path(path)
        return sorted(item.name for item in target.iterdir())

    def read_text(self, path: str, limit: int = 2000) -> str:
        target = self._resolve_path(path)
        return target.read_text(encoding="utf-8", errors="ignore")[:limit]

    def finish(self, value: Any) -> None:
        self.finished = True
        self.result = value

    def _summarize_value(self, value: Any) -> str:
        rendered = repr(value)
        if len(rendered) > 120:
            rendered = rendered[:117] + "..."
        return f"{type(value).__name__} {rendered}"

    def snapshot_state(self) -> dict[str, str]:
        state = {}
        for name, value in self._globals.items():
            if name.startswith("_") or name == "__builtins__" or name in self._helper_names:
                continue
            state[name] = self._summarize_value(value)
        return state

    def execute(self, code: str) -> ExecutionObservation:
        stream = io.StringIO()
        error = None
        with redirect_stdout(stream):
            try:
                exec(code, self._globals, self._globals)
            except Exception:
                error = traceback.format_exc()
        return ExecutionObservation(
            stdout=stream.getvalue().strip(),
            error=error,
            state=self.snapshot_state(),
            finished=self.finished,
            result=self.result,
        )


class RLMPrototypeController:
    SYSTEM_PROMPT = (
        "You are testing a minimal Recursive Language Model runtime.\n"
        "Write only Python code, no prose.\n"
        "State persists between steps.\n"
        "Use these helpers:\n"
        "- list_dir(path='.') -> list[str]\n"
        "- read_text(path, limit=2000) -> str\n"
        "- finish(value) -> stop the controller\n"
        "Do not import modules. Inspect the repository via helpers and call finish when done."
    )

    def __init__(self, client: QueryClient, repl: LocalREPL, max_steps: int = 6):
        self.client = client
        self.repl = repl
        self.max_steps = max_steps

    def _extract_code(self, response: str) -> str:
        stripped = response.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    def _build_prompt(self, goal: str, step: int, previous: str) -> str:
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Goal: {goal}\n"
            f"Current step: {step}/{self.max_steps}\n\n"
            f"Previous observation:\n{previous}\n"
        )

    def run(self, goal: str) -> ControllerResult:
        previous = "No previous observation."
        observations: list[ExecutionObservation] = []
        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(goal, step, previous)
            code = self._extract_code(self.client.query(prompt))
            observation = self.repl.execute(code)
            observations.append(observation)
            if observation.finished:
                return ControllerResult(result=observation.result, steps=observations)
            previous = observation.to_prompt()
        raise RuntimeError(f"Controller stopped after reaching max_steps={self.max_steps} without finish().")
