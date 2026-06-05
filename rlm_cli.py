from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from isohyps.rlm_runtime import (
    BudgetLimits,
    ChildQueryConfig,
    ControllerResult,
    PartialBudgetLimits,
    RLMController,
    RunContext,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False


load_dotenv()


class GeminiClient:
    def __init__(self, model_name: str):
        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise ValueError("google-genai package is required for the Gemini backend.") from exc
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def query(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return (response.text or "").strip()


class OllamaClient:
    def __init__(self, model_name: str, base_url: str | None, num_ctx: int):
        try:
            import ollama
        except ModuleNotFoundError as exc:
            raise ValueError("ollama package is required for the Ollama backend.") from exc
        self.model_name = model_name
        self.client = ollama.Client(host=base_url) if base_url else ollama
        self.options = {
            "num_ctx": num_ctx,
            "temperature": 0.2,
        }

    def query(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options=self.options,
        )
        return response["message"]["content"].strip()


class ScriptedClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def query(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("No more scripted responses are available.")
        return self.responses.pop(0)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def result_to_dict(result: ControllerResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "result": _jsonable(result.result),
        "error": result.error,
        "budget": {
            "steps_used": result.budget.steps_used,
            "llm_calls": result.budget.llm_calls,
            "prompt_tokens": result.budget.prompt_tokens,
            "response_tokens": result.budget.response_tokens,
            "total_tokens": result.budget.total_tokens,
        },
        "final_state": result.final_state,
        "steps": [
            {
                "kind": step.kind,
                "stdout": step.stdout,
                "error": step.error,
                "state": step.state,
                "finished": step.finished,
                "result": _jsonable(step.result),
            }
            for step in result.steps
        ],
    }


def _read_goal(args: argparse.Namespace) -> str:
    provided = [args.goal is not None, args.goal_file is not None, args.goal_stdin]
    if sum(bool(item) for item in provided) != 1:
        raise ValueError("Specify exactly one of --goal, --goal-file, or --goal-stdin.")
    if args.goal is not None:
        return args.goal
    if args.goal_file is not None:
        return Path(args.goal_file).read_text(encoding="utf-8")
    return sys.stdin.read()


def _make_client(args: argparse.Namespace):
    if args.backend == "scripted":
        if not args.scripted_response:
            raise ValueError("--backend scripted requires at least one --scripted-response.")
        return ScriptedClient(args.scripted_response)
    if args.backend == "gemini":
        return GeminiClient(args.model or "gemini-1.5-flash")
    if args.backend == "ollama":
        return OllamaClient(args.model or "llama3", args.ollama_url, args.num_ctx)
    raise ValueError(f"Unsupported backend: {args.backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the generic RLM controller without project-analysis output.")
    parser.add_argument("root", nargs="?", default=".", help="Sandbox root directory for helper access")

    goal_group = parser.add_mutually_exclusive_group(required=True)
    goal_group.add_argument("--goal", help="Goal text passed to the RLM controller")
    goal_group.add_argument("--goal-file", help="Read goal text from a UTF-8 file")
    goal_group.add_argument("--goal-stdin", action="store_true", help="Read goal text from stdin")

    parser.add_argument("--backend", choices=["gemini", "ollama", "scripted"], default="gemini")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--ollama-url", help="Base URL for Ollama API")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context size for Ollama")
    parser.add_argument(
        "--scripted-response",
        action="append",
        default=[],
        help="Scripted Python-code response for --backend scripted; repeat for multiple controller/child calls",
    )

    parser.add_argument("--max-steps", type=int, default=8, help="Max controller steps")
    parser.add_argument("--depth", type=int, default=2, help="Max child-query depth")
    parser.add_argument("--max-total-tokens", type=int, default=30000, help="Approximate shared token budget")
    parser.add_argument("--step-timeout", type=float, default=15.0, help="Per-step sandbox timeout in seconds")
    parser.add_argument("--llm-timeout", type=float, default=120.0, help="Per-query model timeout in seconds")
    parser.add_argument("--max-stdout-chars", type=int, default=2000, help="Max captured stdout per step")
    parser.add_argument("--max-state-items", type=int, default=20, help="Max state items shown in observations")
    parser.add_argument("--max-state-value-chars", type=int, default=160, help="Max chars per state value summary")
    parser.add_argument("--child-max-steps", type=int, help="Override child-query max steps")
    parser.add_argument("--child-max-total-tokens", type=int, help="Override child-query token budget")

    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    parser.add_argument("--fail-on-nonfinished", action="store_true", help="Exit nonzero unless status is finished")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    goal = _read_goal(args)
    client = _make_client(args)
    limits = BudgetLimits(
        max_steps=args.max_steps,
        max_depth=args.depth,
        max_total_tokens=args.max_total_tokens,
        step_timeout_seconds=args.step_timeout,
        llm_timeout_seconds=args.llm_timeout,
        max_stdout_chars=args.max_stdout_chars,
        max_state_items=args.max_state_items,
        max_state_value_chars=args.max_state_value_chars,
    )

    child_limits = None
    if args.child_max_steps is not None or args.child_max_total_tokens is not None:
        child_limits = PartialBudgetLimits(
            max_steps=args.child_max_steps,
            max_total_tokens=args.child_max_total_tokens,
        )

    controller = RLMController(
        client=client,
        root=Path(args.root).resolve(),
        run_context=RunContext(limits=limits),
        child_config=ChildQueryConfig(limits=child_limits),
    )
    return result_to_dict(controller.run(goal=goal))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    indent = None if args.compact else 2
    print(json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_nonfinished and payload["status"] != "finished":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
