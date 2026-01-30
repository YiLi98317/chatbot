import os
import unittest
from pathlib import Path

from chatbot.retrieval.entity_resolver import resolve_entity


class TestExactMatchOverride(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[2]  # project root
        db_path = base / "data" / "chinook" / "Chinook_Sqlite.sqlite"
        self.sqlite_uri = f"sqlite:///{db_path}" if db_path.exists() else None

    def test_exact_match_promotes_high_decision(self):
        if not self.sqlite_uri:
            self.skipTest("Chinook SQLite DB not found; skipping integration test.")
        query = "fly me to the moon"
        res = resolve_entity(
            db_uri=self.sqlite_uri,
            query=query,
            normalized_query=query,
            fts_query_primary=query,
            fts_query_fallback="fly me to moon",
            fuzzy_query=query,
            limit=25,
            debug=False,
        )
        self.assertIn(res.get("decision"), {"high", "medium"})
        hits = res.get("hits", [])
        self.assertTrue(hits, "Expected at least one hit")
        top_name = hits[0].get("name", "").lower()
        # Should prefer exact match if present
        self.assertTrue("fly me to the moon" in top_name)
        # Ensure 1045 (Fly Me To The Moon) ranks above 2932 (The Fly) when both appear
        ids = [h.get("entity_id") for h in hits]
        if 1045 in ids and 2932 in ids:
            self.assertLess(ids.index(1045), ids.index(2932))
        # If exact title exists, it must be at top-1
        self.assertEqual(hits[0].get("entity_id"), 1045)


if __name__ == "__main__":
    unittest.main()

