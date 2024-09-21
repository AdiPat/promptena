from promptena.llm import LLM, LLMOptions, GenerateOptions
import uuid
import pytest
import openai
from pydantic import BaseModel


def test_default_options():
    options = LLMOptions()
    assert options.model == "gpt-4o-mini"
    assert options.temperature == 0.5
    assert not options.verbose
    assert isinstance(uuid.UUID(options.llm_id), uuid.UUID)


def test_custom_options():
    custom_id = str(uuid.uuid4())
    options = LLMOptions(
        model="custom-model", temperature=0.7, verbose=True, llm_id=custom_id
    )
    assert options.model == "custom-model"
    assert options.temperature == 0.7
    assert options.verbose
    assert options.llm_id == custom_id


def test_llm_initialization():
    options = LLMOptions()
    llm = LLM(options)
    assert llm.options == options
    assert llm.client


def test_generate_text():
    options = GenerateOptions(prompt="Generate 1 line of Lorem Ipsum text.")
    llm_options = LLMOptions()
    llm = LLM(llm_options)
    result = llm.generate_text(options=options)
    assert len(result) > 0


def test_invalid_model():
    options = LLMOptions(model="invalid-model")
    llm = LLM(options)
    with pytest.raises(openai.APIError):
        llm.generate_text(
            options=GenerateOptions(prompt="Generate 1 line of Lorem Ipsum text.")
        )


class TestResponse(BaseModel):
    text: str
    author: str


def test_generate_object():
    options = GenerateOptions(
        prompt="Generate 1 line of Lorem Ipsum text.",
        response_schema=TestResponse,
    )
    llm_options = LLMOptions()
    llm = LLM(llm_options)
    result = llm.generate_object(options=options)
    assert result.text
    assert result.author
    assert len(result.text) > 0
    assert len(result.author) > 0
