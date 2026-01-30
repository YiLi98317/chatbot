import unittest

from chatbot.retrieval.entity_resolver import fuzzy_rerank


class TestFuzzyScoring(unittest.TestCase):
    def test_fuzzy_ordering_prefers_exact(self):
        query = "fly me to the moon"
        # Simulate FTS rows with bm25 ranks (lower is better)
        rows = [
            {"entity_type": "Track", "entity_id": 1045, "name": "Fly Me To The Moon", "extra": "", "rank": -11.350},
            {"entity_type": "Track", "entity_id": 2932, "name": "The Fly", "extra": "", "rank": -6.947},
            {"entity_type": "Track", "entity_id": 1049, "name": "Come Fly With Me", "extra": "", "rank": -8.115},
            {"entity_type": "Track", "entity_id": 2692, "name": "Sparks Will Fly", "extra": "", "rank": -6.910},
        ]
        ranked = fuzzy_rerank(query, rows)
        # Find candidates by name
        by_name = {c.name: c for c in ranked}
        self.assertGreater(by_name["Fly Me To The Moon"].fuzzy, by_name["Sparks Will Fly"].fuzzy)
        self.assertGreater(by_name["Fly Me To The Moon"].fuzzy, by_name["Come Fly With Me"].fuzzy)
        self.assertGreater(by_name["Fly Me To The Moon"].fuzzy, by_name["The Fly"].fuzzy)
        # The Fly should not get a perfect score
        self.assertLess(by_name["The Fly"].fuzzy, 100.0)

    def test_final_differs_when_bm25_differs(self):
        # With empty query, fuzzy should be comparable; bm25_norm should drive ordering
        query = ""
        rows = [
            {"entity_type": "Track", "entity_id": 1, "name": "A", "extra": "", "rank": -11.350},
            {"entity_type": "Track", "entity_id": 2, "name": "B", "extra": "", "rank": -8.115},
        ]
        ranked = fuzzy_rerank(query, rows)
        # Sort for inspection
        ranked_sorted = sorted(ranked, key=lambda x: x.final, reverse=True)
        self.assertGreater(ranked_sorted[0].final, ranked_sorted[1].final)
        # And bm25_norm should reflect stronger match for lower bm25 (more negative)
        self.assertGreater(ranked_sorted[0].bm25_norm, ranked_sorted[1].bm25_norm)


if __name__ == "__main__":
    unittest.main()

