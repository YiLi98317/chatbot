import unittest

from chatbot.retrieval.query_planner import deterministic_fallback_plan, _post_process_plan, QueryPlan


class TestPlannerRawPreservation(unittest.TestCase):
    def test_raw_query_preserved(self):
        raw = "fly me to the moon"
        plan = deterministic_fallback_plan(raw)
        self.assertEqual(plan.raw_query, raw)
        self.assertEqual(plan.user_input_raw, raw)

    def test_post_process_adds_variants(self):
        raw = "fly me to the moon"
        plan = deterministic_fallback_plan(raw)
        plan2 = _post_process_plan(plan)
        self.assertTrue(plan2.query_normalized)
        self.assertTrue(plan2.query_for_fts_primary)
        self.assertTrue(plan2.query_for_fuzzy)
        # For entity_lookup (title-like), primary/lexical should be derived from literal input, not LLM variants
        self.assertEqual(plan2.query_for_fts_primary, plan2.query_normalized)
        self.assertEqual(plan2.lexical_query, plan2.query_for_fts_fallback)


if __name__ == "__main__":
    unittest.main()

