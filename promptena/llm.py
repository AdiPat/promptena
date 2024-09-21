import uuid
import openai
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMOptions(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.5)
    verbose: bool = Field(default=False)
    llm_id: str = Field(default=str(uuid.uuid4()))


class LLM:

    def __init__(self, options: LLMOptions):
        self.verbose = options.verbose
        self.options = options

        openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.Client(api_key=openai_key)
