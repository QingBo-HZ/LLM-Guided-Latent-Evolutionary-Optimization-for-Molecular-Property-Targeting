"""LLM API configuration.

Set credentials through environment variables before running generation scripts:

    export OPENAI_BASE_URL="https://api.openai.com/v1"
    export OPENAI_API_KEY="..."
"""

import os


class LLMConfig:
    def __init__(self):
        self.OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
        if not self.OPENAI_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for LLM generation scripts")
