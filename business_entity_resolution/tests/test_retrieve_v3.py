import unittest

from business_entity_resolution.src.retrieve_v3 import (
    address_parts, enhanced_keys, enhanced_cap, pair_view, pair_features, pair_rank, route_pool,
)


class EnhancedRetrievalTests(unittest.TestCase):
    def test_folded_keys_bridge_accents_without_removing_originals(self):
        left = dict(enhanced_keys("École Française SAS", "0141 Rue du Marché"))
        right = dict(enhanced_keys("Ecole Francaise", "141 Rue du Marche"))
        self.assertIn("f:t:ecole", left.keys() & right.keys())
        self.assertIn("an:marche|141", left.keys() & right.keys())
        self.assertIn("n:141", left)
        self.assertEqual(len(enhanced_keys("Acme Acme", "Market Market 141 141")),
                         len(dict(enhanced_keys("Acme Acme", "Market Market 141 141"))))

    def test_address_abbreviation_and_numbers(self):
        self.assertEqual(address_parts("0141 Longstreet Rd"), address_parts("141 Longstreet Road"))
        self.assertNotEqual(address_parts("142 Longstreet Road"), address_parts("141 Longstreet Road"))
        self.assertEqual(address_parts(None), (frozenset(), frozenset()))

    def test_address_ranking_disambiguates_equal_names(self):
        query = pair_view("Acme Bakery", "0141 Longstreet Road")
        good = pair_features(query, pair_view("Acme Bakery LLC", "141 Longstreet Rd"))
        bad = pair_features(query, pair_view("Acme Bakery LLC", "999 Elsewhere Avenue"))
        self.assertEqual(pair_rank(good, False), pair_rank(bad, False))
        self.assertGreater(pair_rank(good, True), pair_rank(bad, True))

    def test_empty_names_do_not_match_exactly(self):
        features = pair_features(pair_view("", ""), pair_view("", ""))
        self.assertEqual(features["name_exact"], 0)
        self.assertEqual(pair_rank(features), 0)

    def test_route_pool_reserves_address_candidates(self):
        ranked = list(range(100))
        routes = {candidate: {"t"} if candidate < 90 else {"a2"} for candidate in ranked}
        pool = route_pool(ranked, routes, 12)
        self.assertEqual(len(pool), 12)
        self.assertEqual(len(set(pool)), 12)
        self.assertIn(90, pool)
        self.assertIn(91, pool)
        self.assertEqual(pool, sorted(pool))

    def test_extra_keys_are_capped(self):
        for key, _ in enhanced_keys("Krishna Exports", "147 Preet Vihar Longstreet"):
            self.assertLessEqual(enhanced_cap(key), 5000)

    def test_telugu_name_has_a_shared_skeleton_route(self):
        latin = dict(enhanced_keys("Lotus Infotech", ""))
        telugu = dict(enhanced_keys("లోటస్ ఇన్ఫోటెక్", ""))
        self.assertTrue(any(key.startswith("sg:") for key in latin.keys() & telugu.keys()))


if __name__ == "__main__":
    unittest.main()
