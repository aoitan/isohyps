"""Shared test utilities for RLM runtime tests."""


class ScriptedClient:
    """A test-only LLM client that returns pre-configured responses in order.

    If a response is an Exception instance, it is raised instead of returned.
    Raises AssertionError when all responses have been consumed.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def query(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError(
                f"No more scripted responses left. Prompt was: {prompt[:100]}..."
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
