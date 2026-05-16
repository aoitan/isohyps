from __future__ import annotations

import ast
import io
import math
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol, TypedDict

from analysis_helpers import detect_language, extract_symbols, read_text_excerpt

_DEBUG = bool(os.getenv("DEBUG"))


def _dbg(tag: str, msg: str) -> None:
    if _DEBUG:
        print(f"[DBG {tag}] {msg}", file=sys.stderr, flush=True)


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
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
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
    "zip": zip,
}


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _cap_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _summarize_value(value: Any, limit: int) -> str:
    rendered = repr(value)
    if len(rendered) > limit:
        rendered = rendered[: limit - 3] + "..."
    return f"{type(value).__name__} {rendered}"


@dataclass
class BudgetLimits:
    max_steps: int = 8
    max_depth: int = 2
    max_total_tokens: int = 30000
    step_timeout_seconds: float = 15.0
    max_stdout_chars: int = 2000
    max_state_items: int = 20
    max_state_value_chars: int = 160


@dataclass
class BudgetSnapshot:
    steps_used: int
    llm_calls: int
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class RunContext:
    limits: BudgetLimits
    steps_used: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0

    def reserve_step(self) -> None:
        if self.steps_used >= self.limits.max_steps:
            raise BudgetExceededError(f"max_steps={self.limits.max_steps} reached")
        self.steps_used += 1
        _dbg("budget", f"step reserved: {self.steps_used}/{self.limits.max_steps}")

    def ensure_depth(self, depth: int) -> None:
        if depth > self.limits.max_depth:
            raise BudgetExceededError(f"max_depth={self.limits.max_depth} reached")

    def record_query(self, prompt: str, response: str) -> None:
        self.llm_calls += 1
        self.prompt_tokens += _approx_tokens(prompt)
        self.response_tokens += _approx_tokens(response)
        _dbg("budget", f"llm_call #{self.llm_calls}: total_tokens={self.total_tokens}/{self.limits.max_total_tokens} (prompt_cumul={self.prompt_tokens} resp_cumul={self.response_tokens})")
        if self.total_tokens > self.limits.max_total_tokens:
            raise BudgetExceededError(f"max_total_tokens={self.limits.max_total_tokens} reached")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps_used=self.steps_used,
            llm_calls=self.llm_calls,
            prompt_tokens=self.prompt_tokens,
            response_tokens=self.response_tokens,
            total_tokens=self.total_tokens,
        )


@dataclass
class ExecutionObservation:
    kind: str
    stdout: str
    error: str | None
    state: dict[str, str]
    finished: bool
    result: Any

    def to_prompt(self) -> str:
        state_lines = "\n".join(f"- {name}: {value}" for name, value in sorted(self.state.items()))
        return (
            f"kind: {self.kind}\n"
            f"stdout:\n{self.stdout or '(empty)'}\n\n"
            f"error:\n{self.error or '(none)'}\n\n"
            f"state:\n{state_lines or '(empty)'}\n\n"
            f"finished: {self.finished}\n"
            f"result: {_summarize_value(self.result, 120)}"
        )

    @classmethod
    def issue(cls, kind: str, error: str) -> "ExecutionObservation":
        return cls(kind=kind, stdout="", error=error, state={}, finished=False, result=None)


@dataclass
class ControllerResult:
    status: str
    result: Any
    steps: list[ExecutionObservation]
    error: str | None
    budget: BudgetSnapshot
    final_state: dict[str, str]


@dataclass
class ValidatedCode:
    kind: str
    code: str | None
    error: str | None


class AnalysisDocumentDict(TypedDict, total=False):
    path: str
    title: str
    content: str


class _AnalysisResultOptionalDict(TypedDict, total=False):
    documents: list[AnalysisDocumentDict]


class AnalysisResultDict(_AnalysisResultOptionalDict):
    summary: str


ANALYSIS_RESULT_SCHEMA_TEXT = (
    "  {\n"
    "    'summary': str,        # High-level overview of the findings\n"
    "    'documents': [         # Optional list; omit it when no detailed docs are needed\n"
    "      {\n"
    "        'path': str,       # Optional relative path (e.g., 'auth.md')\n"
    "        'title': str,      # Optional document title\n"
    "        'content': str     # Optional Markdown content\n"
    "      }, ...\n"
    "    ]\n"
    "  }\n"
)

DEFAULT_GOAL_TEMPLATE = (
    "Analyze the project rooted at '{root_name}'. "
    "Inspect the repository using helpers, use llm_query only for focused subproblems, "
    "and finish with a dict containing a string 'summary'. "
    "Optionally include a 'documents' list; document path, title, and content fields are optional strings."
)


def validate_analysis_result(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [f"Expected finish(value) to receive a dict, got {type(value).__name__}."]

    errors: list[str] = []
    if "summary" not in value:
        errors.append("Missing required key 'summary'.")
    elif not isinstance(value["summary"], str):
        errors.append(f"Expected 'summary' to be str, got {type(value['summary']).__name__}.")

    documents = value.get("documents")
    if documents is None:
        return errors
    if not isinstance(documents, list):
        errors.append(f"Expected 'documents' to be list, got {type(documents).__name__}.")
        return errors

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            errors.append(f"Expected documents[{index}] to be dict, got {type(document).__name__}.")
            continue
        for field in ("path", "title", "content"):
            field_value = document.get(field)
            if field_value is not None and not isinstance(field_value, str):
                errors.append(
                    f"Expected documents[{index}]['{field}'] to be str, got {type(field_value).__name__}."
                )
    return errors


def format_analysis_result_errors(errors: list[str]) -> str:
    detail = " ".join(errors)
    return (
        f"Invalid analysis result format: {detail} "
        "Call finish() with a dict containing at least a string 'summary' and, if provided, "
        "a 'documents' list of dicts with string 'path', 'title', and 'content' fields."
    )


class CodeResponseValidator:
    MODEL_ERROR_PREFIXES = ("[Gemini Error:", "[Ollama Error:", "[Error:")

    def normalize(self, response: str) -> ValidatedCode:
        stripped = response.strip()
        if not stripped:
            return ValidatedCode(kind="invalid_code", code=None, error="Model returned an empty response.")
        if stripped.startswith(self.MODEL_ERROR_PREFIXES):
            return ValidatedCode(kind="model_error", code=None, error=stripped)

        code = stripped
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines).strip()

        if not code:
            return ValidatedCode(kind="invalid_code", code=None, error="No Python code remained after normalization.")

        try:
            ast.parse(code)
        except SyntaxError as exc:
            return ValidatedCode(kind="invalid_code", code=None, error=f"SyntaxError: {exc}")

        return ValidatedCode(kind="code", code=code, error=None)


class PromptBuilder:
    SYSTEM_PROMPT = (
        "You are operating a Recursive Language Model runtime.\n"
        "Return only Python code and no prose.\n"
        "State persists across steps inside a Python sandbox.\n"
        "Use helpers instead of imports or direct OS access.\n"
        "Available helpers:\n"
        "- list_dir(path='.') -> list[str]\n"
        "- read_text(path, limit=2000) -> str\n"
        "- extract_symbols(path) -> dict(language, symbols, fallback_excerpt, error)\n"
        "- llm_query(prompt, context=None) -> child result value\n"
        "- finish(value) -> immediately end the run. For top-level project analysis, the value MUST be a dict matching:\n"
        f"{ANALYSIS_RESULT_SCHEMA_TEXT}"
        "Rules:\n"
        "- Do not import modules.\n"
        "- Do not attempt network, subprocess, or filesystem mutation.\n"
        "- Prefer helper calls over large string constants.\n"
        "- When you are done, call finish(value).\n"
    )

    def build(self, goal: str, step: int, max_steps: int, previous: str, parent_context: Any | None) -> str:
        context_line = "(none)" if parent_context is None else _summarize_value(parent_context, 400)
        return (
            f"{self.SYSTEM_PROMPT}\n"
            f"Goal: {goal}\n"
            f"Current step: {step}/{max_steps}\n"
            f"Parent context: {context_line}\n\n"
            f"Previous observation:\n{previous}\n"
        )


class _FinishSignal(Exception):
    def __init__(self, value: Any):
        super().__init__("finish")
        self.value = value


class _CappedWriter(io.StringIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self._size = 0
        self._truncated = False

    def write(self, s: str) -> int:
        if self._size >= self.limit:
            self._truncated = True
            return len(s)
        remaining = self.limit - self._size
        chunk = s[:remaining]
        self._size += len(chunk)
        if len(chunk) < len(s):
            self._truncated = True
        return super().write(chunk)

    def get_capped_value(self) -> str:
        value = self.getvalue().strip()
        if self._truncated:
            value = f"{value}\n...[stdout truncated]".strip()
        return value


def _sandbox_worker(connection: Connection, root: str, limits: BudgetLimits) -> None:
    root_path = Path(root).resolve()
    helper_names = {"list_dir", "read_text", "extract_symbols", "llm_query", "finish"}
    globals_dict: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}

    def resolve_path(path: str | Path) -> Path:
        candidate = (root_path / Path(path)).resolve()
        if candidate != root_path and root_path not in candidate.parents:
            raise ValueError(f"path escapes root: {path}")
        return candidate

    def list_dir(path: str = ".") -> list[str]:
        _dbg("sandbox", f"list_dir({path!r})")
        target = resolve_path(path)
        return sorted(item.name for item in target.iterdir())

    def read_text(path: str, limit: int = 2000) -> str:
        _dbg("sandbox", f"read_text({path!r}, limit={limit})")
        target = resolve_path(path)
        return read_text_excerpt(target, limit=min(limit, 2000))

    def extract_symbols_helper(path: str) -> dict[str, Any]:
        _dbg("sandbox", f"extract_symbols({path!r})")
        target = resolve_path(path)
        info = extract_symbols(target, lang=detect_language(target))
        info["path"] = str(target.relative_to(root_path))
        return info

    def llm_query(prompt: str, context: Any = None) -> Any:
        _dbg("sandbox", f"llm_query {len(prompt)} chars: {prompt[:80]!r}...")
        connection.send({"type": "llm_query", "prompt": prompt, "context": context})
        while True:
            message = connection.recv()
            if message["type"] == "llm_query_result":
                _dbg("sandbox", f"llm_query result: {str(message['value'])[:80]!r}")
                return message["value"]
            if message["type"] == "llm_query_error":
                raise RuntimeError(message["error"])

    def finish(value: Any) -> None:
        _dbg("sandbox", f"finish({type(value).__name__}): {repr(value)[:80]!r}")
        raise _FinishSignal(value)

    globals_dict.update(
        {
            "list_dir": list_dir,
            "read_text": read_text,
            "extract_symbols": extract_symbols_helper,
            "llm_query": llm_query,
            "finish": finish,
        }
    )

    def snapshot_state() -> dict[str, str]:
        state: dict[str, str] = {}
        for name, value in globals_dict.items():
            if name.startswith("_") or name == "__builtins__" or name in helper_names:
                continue
            state[name] = _summarize_value(value, limits.max_state_value_chars)
            if len(state) >= limits.max_state_items:
                state["..."] = f"state truncated at {limits.max_state_items} items"
                break
        return state

    while True:
        try:
            command = connection.recv()
        except EOFError:
            break

        if command["type"] == "shutdown":
            break
        if command["type"] != "exec":
            connection.send({"type": "result", "kind": "execution_error", "stdout": "", "error": "Unknown command.", "state": snapshot_state(), "finished": False, "result": None})
            continue

        stream = _CappedWriter(limits.max_stdout_chars)
        error = None
        finished = False
        result = None
        kind = "ok"
        with redirect_stdout(stream):
            try:
                exec(command["code"], globals_dict, globals_dict)
            except _FinishSignal as finish_signal:
                finished = True
                result = finish_signal.value
                kind = "finished"
            except Exception:
                error = _cap_text(traceback.format_exc(), limits.max_stdout_chars)
                kind = "execution_error"
        connection.send(
            {
                "type": "result",
                "kind": kind,
                "stdout": stream.get_capped_value(),
                "error": error,
                "state": snapshot_state(),
                "finished": finished,
                "result": result,
            }
        )


class IsolatedREPL:
    def __init__(self, root: Path, limits: BudgetLimits):
        self.root = root.resolve()
        self.limits = limits
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive() and self._connection is not None:
            return
        if self._connection is not None:
            self._connection.close()
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(
            target=_sandbox_worker,
            args=(child_conn, str(self.root), self.limits),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._connection = parent_conn
        _dbg("repl", f"sandbox worker started pid={process.pid} root={self.root}")

    def _restart_worker(self) -> None:
        _dbg("repl", "restarting sandbox worker")
        self.close()
        self._ensure_worker()

    def execute(self, code: str, llm_query_handler, finish_validator=None) -> ExecutionObservation:
        self._ensure_worker()
        assert self._connection is not None
        assert self._process is not None

        first_line = code.split("\n")[0][:60]
        _dbg("repl", f"execute ({len(code)} chars): {first_line!r}...")
        self._connection.send({"type": "exec", "code": code})
        deadline = time.monotonic() + self.limits.step_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _dbg("repl", f"sandbox step timed out after {self.limits.step_timeout_seconds:.1f}s — restarting worker")
                self._restart_worker()
                return ExecutionObservation.issue(
                    kind="execution_error",
                    error=f"Sandbox step timed out after {self.limits.step_timeout_seconds:.1f}s; sandbox state was reset.",
                )
            if not self._connection.poll(min(remaining, 0.1)):
                continue
            message = self._connection.recv()
            if message["type"] == "llm_query":
                _dbg("repl", f"sandbox requests llm_query: {message['prompt'][:80]!r}...")
                try:
                    value = llm_query_handler(message["prompt"], message.get("context"))
                    _dbg("repl", f"llm_query_handler returned: {str(value)[:80]!r}")
                    self._connection.send({"type": "llm_query_result", "value": value})
                except Exception as exc:
                    _dbg("repl", f"llm_query_handler error: {exc}")
                    self._connection.send({"type": "llm_query_error", "error": str(exc)})
                continue
            if message["type"] == "result":
                obs = ExecutionObservation(
                    kind=message["kind"],
                    stdout=message["stdout"],
                    error=message["error"],
                    state=message["state"],
                    finished=message["finished"],
                    result=message["result"],
                )
                _dbg("repl", f"result kind={obs.kind} finished={obs.finished} stdout={len(obs.stdout)}chars state_keys={list(obs.state.keys())[:5]}")
                if obs.error:
                    _dbg("repl", f"result error: {obs.error[:120]!r}")
                if obs.finished and finish_validator is not None:
                    validation_errors = finish_validator(obs.result)
                    if validation_errors:
                        validation_error = format_analysis_result_errors(validation_errors)
                        _dbg("repl", f"finish validation failed: {validation_error[:120]!r}")
                        return ExecutionObservation(
                            kind="execution_error",
                            stdout=obs.stdout,
                            error=validation_error,
                            state=obs.state,
                            finished=False,
                            result=None,
                        )
                return obs

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.send({"type": "shutdown"})
            except Exception:
                pass
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.2)
            self._process = None

    def __enter__(self) -> "IsolatedREPL":
        self._ensure_worker()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class AnalysisDocument:
    path: str
    title: str
    content: str


@dataclass
class StructuredAnalysis:
    summary: str
    documents: list[AnalysisDocument]


def _normalize_structured_analysis(root_name: str, result: Any) -> StructuredAnalysis:
    if isinstance(result, dict):
        summary = str(result.get("summary") or result.get("result") or "")
        raw_documents = result.get("documents") or []
    else:
        summary = str(result)
        raw_documents = []

    documents: list[AnalysisDocument] = []
    for item in raw_documents:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "index.md")
        title = str(item.get("title") or Path(path).stem or root_name)
        content = str(item.get("content") or "")
        documents.append(AnalysisDocument(path=path, title=title, content=content))

    if not any(doc.path == "index.md" for doc in documents):
        documents.insert(0, AnalysisDocument(path="index.md", title=f"Directory: {root_name}", content=summary))

    return StructuredAnalysis(summary=summary, documents=documents)


def write_analysis_docs(
    output_dir: Path,
    root_path: Path,
    controller_result: ControllerResult,
    backend: str,
    model: str,
) -> StructuredAnalysis:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structured = _normalize_structured_analysis(root_path.name, controller_result.result)

    if controller_result.status != "finished":
        detail = controller_result.error or structured.summary or str(controller_result.result)
        guidance = []
        if controller_result.status == "budget_exceeded":
            guidance.append("Try increasing --max-total-tokens or --max-steps.")
        guidance.append("If the controller keeps failing, retry with --runtime legacy.")
        failure_summary = f"[{controller_result.status}] {detail} {' '.join(guidance)}".strip()
        structured = StructuredAnalysis(
            summary=failure_summary,
            documents=[
                AnalysisDocument(
                    path="index.md",
                    title=f"Analysis stopped: {controller_result.status}",
                    content=failure_summary,
                )
            ],
        )

    def resolve_document_target(document_path: str) -> Path:
        candidate = (output_dir / Path(document_path)).resolve()
        if candidate == output_dir or output_dir not in candidate.parents:
            fallback_name = Path(document_path).name or "index.md"
            candidate = (output_dir / fallback_name).resolve()
        return candidate

    for document in structured.documents:
        target = resolve_document_target(document.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = document.content
        if body and not body.lstrip().startswith("#"):
            body = f"# {document.title}\n\n{body}"
        target.write_text(body or f"# {document.title}\n", encoding="utf-8")

    report_path = output_dir / "analysis_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Project Analysis Report: {root_path.name}",
                "",
                f"**Root Directory:** `{root_path}`  ",
                f"**Backend:** {backend} ({model})  ",
                f"**Runtime:** controller  ",
                f"**Status:** {controller_result.status}  ",
                f"**Steps Used:** {controller_result.budget.steps_used}  ",
                f"**Approx Tokens:** {controller_result.budget.total_tokens}  ",
                "",
                "## Executive Summary",
                "",
                structured.summary or str(controller_result.result),
                "",
                "## Final State",
                "",
                "\n".join(f"- {name}: {value}" for name, value in sorted(controller_result.final_state.items())) or "- (empty)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return structured


class RLMController:
    def __init__(
        self,
        client: QueryClient,
        root: Path,
        run_context: RunContext,
        prompt_builder: PromptBuilder | None = None,
        validator: CodeResponseValidator | None = None,
    ):
        self.client = client
        self.root = root.resolve()
        self.run_context = run_context
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.validator = validator or CodeResponseValidator()

    def run(
        self,
        goal: str,
        depth: int = 0,
        parent_context: Any | None = None,
        require_structured_finish: bool = False,
    ) -> ControllerResult:
        _dbg("ctrl", f"run start depth={depth} goal={goal[:80]!r}")
        steps: list[ExecutionObservation] = []
        previous = "No previous observation."
        final_state: dict[str, str] = {}

        try:
            self.run_context.ensure_depth(depth)
        except BudgetExceededError as exc:
            _dbg("ctrl", f"depth budget exceeded: {exc}")
            return ControllerResult(
                status="budget_exceeded",
                result=None,
                steps=steps,
                error=str(exc),
                budget=self.run_context.snapshot(),
                final_state=final_state,
            )

        with IsolatedREPL(self.root, self.run_context.limits) as repl:
            while True:
                try:
                    self.run_context.reserve_step()
                except BudgetExceededError as exc:
                    _dbg("ctrl", f"step budget exceeded: {exc}")
                    return ControllerResult(
                        status="budget_exceeded",
                        result=None,
                        steps=steps,
                        error=str(exc),
                        budget=self.run_context.snapshot(),
                        final_state=final_state,
                    )

                prompt = self.prompt_builder.build(
                    goal=goal,
                    step=self.run_context.steps_used,
                    max_steps=self.run_context.limits.max_steps,
                    previous=previous,
                    parent_context=parent_context,
                )
                _dbg("ctrl", f"step {self.run_context.steps_used}/{self.run_context.limits.max_steps}: sending prompt ({len(prompt)} chars)")

                try:
                    response = self.client.query(prompt)
                    _dbg("ctrl", f"llm response ({len(response)} chars): {response[:100]!r}...")
                    self.run_context.record_query(prompt, response)
                except BudgetExceededError as exc:
                    _dbg("ctrl", f"token budget exceeded during query: {exc}")
                    observation = ExecutionObservation.issue("model_error", str(exc))
                except Exception as exc:
                    _dbg("ctrl", f"llm query error: {type(exc).__name__}: {exc}")
                    observation = ExecutionObservation.issue("model_error", f"{type(exc).__name__}: {exc}")
                else:
                    validation = self.validator.normalize(response)
                    _dbg("ctrl", f"validation kind={validation.kind}" + (f" error={validation.error!r}" if validation.error else ""))
                    if validation.kind != "code":
                        observation = ExecutionObservation.issue(validation.kind, validation.error or validation.kind)
                    else:
                        observation = repl.execute(
                            validation.code,
                            llm_query_handler=lambda child_goal, child_context: self._run_subquery(
                                child_goal,
                                child_context,
                                depth=depth + 1,
                            ),
                            finish_validator=validate_analysis_result if require_structured_finish else None,
                        )

                steps.append(observation)
                final_state = observation.state or final_state
                _dbg("ctrl", f"observation: kind={observation.kind} finished={observation.finished} steps_so_far={len(steps)}")
                if observation.finished:
                    _dbg("ctrl", f"finish at step {self.run_context.steps_used}: result={type(observation.result).__name__}")
                    return ControllerResult(
                        status="finished",
                        result=observation.result,
                        steps=steps,
                        error=None,
                        budget=self.run_context.snapshot(),
                        final_state=final_state,
                    )
                previous = observation.to_prompt()

    def _run_subquery(self, goal: str, child_context: Any, depth: int) -> Any:
        _dbg("ctrl", f"subquery depth={depth} goal={goal[:80]!r}")
        child_result = self.run(goal=goal, depth=depth, parent_context=child_context)
        _dbg("ctrl", f"subquery done status={child_result.status} steps_used={child_result.budget.steps_used} tokens={child_result.budget.total_tokens}")
        if child_result.status != "finished":
            raise RuntimeError(child_result.error or f"Child query stopped with status={child_result.status}")
        return child_result.result
