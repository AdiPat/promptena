import uuid
import openai
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from typing import Type

load_dotenv()


class LLMOptions(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.5)
    verbose: bool = Field(default=False)
    llm_id: str = Field(default=str(uuid.uuid4()))


class GenerateOptions(BaseModel):
    system_prompt: str = Field(
        default="You are a helpful AI assistant. You will be given a prompt, generate an output for it. "
    )
    prompt: str
    response_schema: Type[BaseModel] = Field(default=None)
    max_tokens: int = Field(default=100)
    temperature: float = Field(default=0.5)
    verbose: bool = Field(default=False)


class LLM:

    def __init__(self, options: LLMOptions):
        self.verbose = options.verbose
        self.options = options

        openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.Client(api_key=openai_key)

    def generate_text(self, options: GenerateOptions) -> str:
        try:
            if (
                options.response_schema
            ):  # we absolutely want the user to see this and hence it's not under verbose
                logging.warning("LLM: Schema is not supported in generate text.")

            response = self.client.chat.completions.create(
                model=self.options.model,
                messages=[
                    {"role": "system", "content": options.system_prompt},
                    {"role": "user", "content": options.prompt},
                ],
                temperature=options.temperature,
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            logging.error(
                "LLM: API Error occurred while generating text.",
                extra={"error": e, "options": options.model_dump()},
            )
            raise e

    def generate_object(self, options: GenerateOptions) -> BaseModel:
        try:
            if not options.response_schema:
                raise ValueError("LLM: Schema is required for generating object.")

            response = self.client.beta.chat.completions.parse(
                model=self.options.model,
                messages=[
                    {"role": "system", "content": options.system_prompt},
                    {"role": "user", "content": options.prompt},
                ],
                temperature=options.temperature,
                response_format=options.response_schema,
            )
            return response.choices[0].message.parsed
        except openai.APIError as e:
            logging.error(
                "LLM: API Error occurred while generating text.",
                extra={"error": e, "options": options.model_dump()},
            )
            raise e
