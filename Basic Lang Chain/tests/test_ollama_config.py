import unittest
from unittest.mock import patch

import main


class TestOllamaConfig(unittest.TestCase):
    @patch("main.urlopen")
    def test_remote_is_used_when_available(self, mock_urlopen):
        response = type("Response", (), {"status": 200})()
        mock_urlopen.return_value.__enter__.return_value = response

        with patch.dict(
            "os.environ",
            {
                "OLLAMA_BASE_URL": "http://192.168.1.3:11434",
                "OLLAMA_LOCAL_BASE_URL": "http://localhost:11434",
            },
            clear=False,
        ):
            self.assertEqual(main.resolve_ollama_base_url(), "http://192.168.1.3:11434")

    @patch("main.urlopen")
    def test_local_is_used_when_remote_unreachable(self, mock_urlopen):
        mock_urlopen.side_effect = [
            Exception("remote unavailable"),
            type("Response", (), {"status": 200})(),
        ]

        with patch.dict(
            "os.environ",
            {
                "OLLAMA_BASE_URL": "http://192.168.1.3:11434",
                "OLLAMA_LOCAL_BASE_URL": "http://localhost:11434",
            },
            clear=False,
        ):
            self.assertEqual(main.resolve_ollama_base_url(), "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
