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

    def __init__(self, model: str = "gemini-2.5-flash-project"):
        api_key = os.getenv("PNNL_API_KEY")
        if not api_key:
            raise ValueError("PNNL_API_KEY environment variable not set")
        self.client = openai.OpenAI(
            api_key  = api_key,
            base_url = "https://ai-incubator-api.pnnl.gov",
        )
        self.model = model

    def ask(self,
            messages:       List[Dict[str, str]],
            system_message: str = None) -> str:
        """Send messages to LLM and get response."""
        if system_message:
            messages = [{"role": "system",
                         "content": system_message}] + messages
        response = self.client.chat.completions.create(
            model    = self.model,
            messages = messages,
        )
        return response.choices[0].message.content


class LLMAgent:
    """Simple LLM-based agent."""

    def __init__(self,
                 name:           str,
                 system_message: str,
                 model:          str = "gemini-2.5-flash-project"):
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

    def parse_json(self, response: str) -> Dict[str, Any]:
        """
        Shared utility — extract JSON from LLM response.
        Handles markdown fences and leading/trailing text.
        Inherited by all agents — do not duplicate in subclasses.
        """
        text = re.sub(r"```json\s*", "", response)
        text = re.sub(r"```\s*$",    "", text).strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"No JSON found in response: {text[:100]}")

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []

    def __str__(self):
        return f"Agent({self.name})"