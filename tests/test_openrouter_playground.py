import ast
from pathlib import Path
import unittest


class OpenRouterPlaygroundTests(unittest.TestCase):
    def test_saved_conversation_json_is_written_as_utf8(self) -> None:
        notebook = Path(__file__).resolve().parents[1] / "notebooks/openrouter_playground.py"
        tree = ast.parse(notebook.read_text(encoding="utf-8"))

        save_writes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "save_path"
        ]

        self.assertEqual(len(save_writes), 1)
        encoding = next(
            (kw.value for kw in save_writes[0].keywords if kw.arg == "encoding"),
            None,
        )
        self.assertIsInstance(encoding, ast.Constant)
        self.assertEqual(encoding.value, "utf-8")


if __name__ == "__main__":
    unittest.main()
