"""Run from backend/:  .venv\\Scripts\\python.exe -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_text import CODE_BLOCK_REPLACEMENT, to_speakable  # noqa: E402


class ToSpeakableTests(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(to_speakable("Done. Anything else?"), "Done. Anything else?")

    def test_bold_and_italic_markers_removed(self):
        self.assertEqual(
            to_speakable("This is **very** important and *also* _this_."),
            "This is very important and also this.",
        )

    def test_headings_become_sentences(self):
        self.assertEqual(to_speakable("## Summary\nAll tests pass"), "Summary. All tests pass.")

    def test_bullets_and_numbered_lists(self):
        self.assertEqual(
            to_speakable("Two things:\n- first item\n- second item\n1. third"),
            "Two things: first item. second item. third.",
        )

    def test_code_fence_replaced(self):
        out = to_speakable("Here you go:\n```python\nprint('hi')\n```\nThat's it.")
        self.assertEqual(out, f"Here you go: {CODE_BLOCK_REPLACEMENT} That's it.")
        self.assertNotIn("print", out)

    def test_unclosed_code_fence_replaced(self):
        self.assertNotIn("print", to_speakable("Start:\n```\nprint('hi')"))

    def test_inline_code_keeps_content(self):
        self.assertEqual(to_speakable("Run `npm install` first."), "Run npm install first.")

    def test_links_keep_label_urls_dropped(self):
        self.assertEqual(
            to_speakable("See [the docs](https://example.com/x) or https://foo.dev/y."),
            "See the docs or a link.",
        )

    def test_table_rows_read_as_lists(self):
        out = to_speakable("| Name | Size |\n|---|---|\n| a.py | 2kb |")
        self.assertEqual(out, "Name, Size. a.py, 2kb.")

    def test_snake_case_and_arrows(self):
        self.assertEqual(
            to_speakable("mute_for_playback -> run_tts"),
            "mute for playback to run tts.",
        )

    def test_hyphenated_words_untouched(self):
        self.assertEqual(to_speakable("Duck-e-bugger is up."), "Duck-e-bugger is up.")

    def test_only_symbols_gives_empty(self):
        self.assertEqual(to_speakable("---\n***\n"), "")
        self.assertEqual(to_speakable("   "), "")


if __name__ == "__main__":
    unittest.main()
