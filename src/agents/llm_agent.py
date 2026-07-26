#!/usr/bin/env python3
"""
Simple LLM Agent using OpenAI API for PNNL
"""
import os
import re
import json
import openai
from typing import List, Dict, Any


class SimpleLLMClient:
    """Wrapper around OpenAI client for PNNL API."""

    def __init__(self, model: str = "claude-opus-4-8-project"):
        api_key = os.getenv("PNNL_API_KEY")
        if not api_key:
            raise ValueError("PNNL_API_KEY environment variable not set")
        self.client = openai.OpenAI(
            api_key  = api_key,
            base_url = "https://ai-incubator-api.pnnl.gov",
        )
        self.model = model
        # reproducibility knobs (None = provider default, preserves old behavior)
        self.temperature = None
        self.seed = None
        self.max_tokens = None
        self.last_response_model = None      # provider-reported model version

    def ask(self,
            messages:       List[Dict[str, str]],
            system_message: str = None) -> str:
        """Send messages to LLM and get response."""
        if system_message:
            messages = [{"role": "system",
                         "content": system_message}] + messages
        kwargs = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            # some gateway backends reject optional sampling params — retry bare
            if "temperature" in kwargs or "seed" in kwargs:
                kwargs.pop("temperature", None)
                kwargs.pop("seed", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise
        self.last_response_model = getattr(response, "model", None) or self.model
        u = getattr(response, "usage", None)   # exact token accounting when the
        self.last_usage = ({"prompt_tokens": u.prompt_tokens,      # gateway
                            "completion_tokens": u.completion_tokens}
                           if u else None)     # returns it; None otherwise
        return response.choices[0].message.content


class LLMAgent:
    """Simple LLM-based agent."""

    def __init__(self,
                 name:           str,
                 system_message: str,
                 model:          str = "claude-opus-4-8-project"):
        self.name           = name
        self.system_message = system_message
        self.llm            = SimpleLLMClient(model=model)
        self.conversation_history = []

    def respond(self, user_message: str) -> str:
        """Get agent response to user message."""
        self.conversation_history.append({
            "role":    "user",
            "content": user_message
        })
        response = self.llm.ask(
            messages       = self.conversation_history,
            system_message = self.system_message
        )
        self.conversation_history.append({
            "role":    "assistant",
            "content": response
        })
        return response

    def ask_with_system(self,
                        user_message:   str,
                        system_message: str) -> str:
        """
        One-shot call with a custom system message.
        Does NOT affect conversation history — stateless.
        Use this instead of swapping self.system_message.
        """
        return self.llm.ask(
            messages       = [{"role": "user", "content": user_message}],
            system_message = system_message
        )

    @staticmethod
    def _strip_json_comments(s: str) -> str:
        """Remove // line comments that sit OUTSIDE string literals.

        The prompt schemas are annotated with `//` comments (see
        reception_agentic.txt), so models regularly echo them into their
        output. Naive regex stripping would corrupt URLs ("http://…") and any
        legitimate slashes inside strings, so track string state properly.
        """
        out, in_str, esc, i, n = [], False, False, 0, len(s)
        while i < n:
            c = s[i]
            if in_str:
                out.append(c)
                if esc:            esc = False
                elif c == "\\":    esc = True
                elif c == '"':     in_str = False
                i += 1
                continue
            if c == '"':
                in_str = True
                out.append(c)
                i += 1
                continue
            if c == "/" and i + 1 < n and s[i + 1] == "/":
                while i < n and s[i] != "\n":      # drop to end of line
                    i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

    @staticmethod
    def _escape_raw_newlines(s: str) -> str:
        """Escape literal newlines/tabs that appear INSIDE string literals."""
        out, in_str, esc = [], False, False
        for c in s:
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                if not esc and c in "\n\r\t" and in_str:
                    out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
                    continue
            elif c == '"':
                in_str = True
            out.append(c)
        return "".join(out)

    def parse_json(self, response: str) -> Dict[str, Any]:
        """
        Shared utility — extract JSON from LLM response.
        Handles markdown fences and leading/trailing text.
        Inherited by all agents — do not duplicate in subclasses.

        Applies a cascade of repairs for the LLM slips seen in practice:
        markdown fences, `//` comments echoed from the prompt schemas,
        trailing commas, and raw newlines inside strings. Each repair is
        additive, so the most-repaired form is tried last.
        """
        text = re.sub(r"```json\s*", "", response)
        text = re.sub(r"```\s*$",    "", text).strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON found in response: {text[:100]}")

        frag       = text[start:end + 1]
        no_comment = self._strip_json_comments(frag)
        no_trail   = re.sub(r",\s*([}\]])", r"\1", no_comment)
        escaped    = self._escape_raw_newlines(no_trail)

        last = None
        for candidate in (frag, no_comment, no_trail, escaped):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                last = e
        # Surface the offending region — "line 77 column 6" alone is useless
        # when the caller never sees the raw response.
        pos  = getattr(last, "pos", 0) or 0
        near = escaped[max(0, pos - 120): pos + 120]
        raise ValueError(
            f"Could not parse JSON after repairs ({last}). Near: …{near}…"
        ) from last

    def parse_json_resilient(self, response: str) -> Dict[str, Any]:
        """parse_json(), plus ONE LLM self-repair round if that fails.

        The deterministic repairs in parse_json() cover fences, echoed `//`
        comments, trailing commas and raw newlines. They deliberately do NOT
        guess at structural slips (a missing comma between elements), because
        inserting punctuation heuristically can silently change the data.
        For those, ask the model to fix its own output — the same approach
        tools/create_slides.py already uses.

        Only the JSON is regenerated, not the reasoning, so this is cheap and
        cannot change the scientific content of a valid response.
        """
        try:
            return self.parse_json(response)
        except Exception as first:
            print(f"   ⚠️  JSON parse failed ({str(first)[:90]}…) — "
                  f"asking the model to repair it")
            try:
                fixed = self.llm.ask(
                    [{"role": "user",
                      "content": "The following was meant to be a single JSON "
                                 "object but does not parse. Return ONLY the "
                                 "corrected JSON — identical content, no "
                                 "commentary, no markdown fence.\n\n" + response}],
                    system_message="You repair malformed JSON. Output valid "
                                   "JSON only, preserving every value exactly.",
                )
                return self.parse_json(fixed)
            except Exception as second:
                raise ValueError(
                    f"JSON unparseable even after LLM repair. "
                    f"original={first} ; repair={second}"
                ) from first

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []

    def __str__(self):
        return f"Agent({self.name})"