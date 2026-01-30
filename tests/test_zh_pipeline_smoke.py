import os
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text

from chatbot.retrieval.normalize import detect_lang, normalize_text
from chatbot.retrieval.entity_resolver import resolve_entity


class TestZhPipelineSmoke(unittest.TestCase):
    def test_detect_lang_basic(self):
        self.assertEqual(detect_lang("hello world"), "en")
        self.assertEqual(detect_lang("月亮代表我的心"), "zh")
        self.assertEqual(detect_lang("Back In Black 专辑 曲目"), "mixed")

    def test_normalize_text_preserves_cjk(self):
        s = "月亮\u200b代表\u200c我的\u200d心"
        out = normalize_text(s, lang="zh")
        self.assertIn("月亮代表我的心", out)

    def test_zh_resolver_does_not_call_fts_search(self):
        # Build a tiny index DB with an entity_fts table; the zh resolver path uses LIKE.
        with tempfile.TemporaryDirectory() as td:
            idx_path = os.path.join(td, "entity_fts.sqlite")
            idx_uri = f"sqlite:///{idx_path}"
            engine = create_engine(idx_uri)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE TABLE entity_fts (entity_type TEXT, entity_id TEXT, name TEXT, extra TEXT)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO entity_fts(entity_type, entity_id, name, extra) "
                        "VALUES ('Track','1','月亮代表我的心','')"
                    )
                )

            with patch("chatbot.retrieval.entity_resolver._derive_index_db_uri", return_value=idx_uri), patch(
                "chatbot.retrieval.entity_resolver.ensure_fts_index", return_value=idx_uri
            ), patch(
                "chatbot.retrieval.entity_resolver.fts_search",
                side_effect=AssertionError("fts_search must not be called for zh/mixed queries"),
            ):
                res = resolve_entity(
                    db_uri="sqlite:///dummy_source.sqlite",
                    query="月亮代表我的心",
                    preferred_tables=["Track"],
                    limit=10,
                    debug=False,
                )
                hits = res.get("hits", []) or []
                self.assertTrue(hits, "Expected at least one LIKE-seeded hit")
                self.assertEqual(hits[0].get("name"), "月亮代表我的心")


if __name__ == "__main__":
    unittest.main()

