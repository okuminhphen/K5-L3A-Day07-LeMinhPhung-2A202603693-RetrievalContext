"""Ollama reports usage even when its generated content cannot be parsed."""
import io
import json
import unittest
from unittest.mock import patch

from src.ollama import chat_json


class OllamaTests(unittest.TestCase):
    def test_token_counts_survive_invalid_model_json(self):
        payload = {
            "prompt_eval_count": 81,
            "eval_count": 17,
            "message": {"content": "{"},
        }
        usage = []
        with patch("src.ollama.urlopen", return_value=io.StringIO(json.dumps(payload))):
            with self.assertRaises(json.JSONDecodeError):
                chat_json("test-model", [], {}, usage=usage)
        self.assertEqual(usage, [{"prompt_tokens": 81, "output_tokens": 17}])
