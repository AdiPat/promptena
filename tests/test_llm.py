from promptena.llm import LLM, LLMOptions
import uuid


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


# class TestLLMOptions(unittest.TestCase):
#     def test_default_options(self):
#         options = LLMOptions()
#         self.assertEqual(options.model, "gpt-4o-mini")
#         self.assertEqual(options.temperature, 0.5)
#         self.assertFalse(options.verbose)
#         self.assertIsInstance(uuid.UUID(options.llm_id), uuid.UUID)

#     def test_custom_options(self):
#         custom_id = str(uuid.uuid4())
#         options = LLMOptions(
#             model="custom-model", temperature=0.7, verbose=True, llm_id=custom_id
#         )
#         self.assertEqual(options.model, "custom-model")
#         self.assertEqual(options.temperature, 0.7)
#         self.assertTrue(options.verbose)
#         self.assertEqual(options.llm_id, custom_id)


# class TestLLM(unittest.TestCase):
#     @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
#     @patch("openai.Client")
#     def test_llm_initialization(self, mock_openai_client):
#         options = LLMOptions()
#         llm = LLM(options)
#         self.assertEqual(llm.options, options)
#         self.assertTrue(llm.client)
#         mock_openai_client.assert_called_once_with(api_key="test_key")

#     @patch.dict(os.environ, {}, clear=True)
#     def test_llm_initialization_without_api_key(self):
#         options = LLMOptions()
#         with self.assertRaises(ValueError) as context:
#             LLM(options)
#         self.assertEqual(
#             str(context.exception), "OPENAI_API_KEY environment variable not set"
#         )


# if __name__ == "__main__":
#     unittest.main()
