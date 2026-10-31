import unittest
from merger_negotiation_game import minimax, alpha_beta
from merger_negotiation_batch_experiments import build_scenarios, build_game_from_scenario


class ExactSearchTests(unittest.TestCase):
    def test_exact_values_match_on_roots_and_successors(self):
        for scenario in build_scenarios():
            for ordered in (False, True):
                game = build_game_from_scenario(scenario, use_move_ordering=ordered)
                states = [game.initial_state]
                states.extend(game.result(game.initial_state, a)
                              for a in game.actions(game.initial_state))
                for state in states:
                    with self.subTest(scenario=scenario.scenario_id,
                                      ordered=ordered, state=state):
                        self.assertAlmostEqual(minimax(game, state)[0],
                                               alpha_beta(game, state)[0])



class ModelRegressionTests(unittest.TestCase):
    def game(self):
        return build_game_from_scenario(build_scenarios()[0])

    def test_buyer_can_spend_remaining_budget(self):
        game = self.game()
        state = game.result(game.initial_state, ('offer', 80))
        state = game.result(state, ('offer', 100))
        self.assertIn(('offer', 90), game.actions(state))
        state = game.result(state, ('offer', 90))
        self.assertEqual(state.buyer_remaining_concession, 0)

    def test_seller_can_spend_remaining_budget(self):
        game = self.game()
        state = game.result(game.initial_state, ('offer', 70))
        state = game.result(state, ('offer', 100))
        state = game.result(state, ('offer', 70))
        self.assertIn(('offer', 80), game.actions(state))

    def test_illegal_action_rejected(self):
        with self.assertRaises(ValueError):
            self.game().result(self.game().initial_state, ('accept', 999))

    def test_pressure_heuristic_can_differ(self):
        game = self.game()
        from dataclasses import replace
        state = replace(game.initial_state, round_no=3)
        self.assertLess(game.h3(state), game.h2(state))

    def test_invalid_step_rejected(self):
        from merger_negotiation_game import BilateralMergerNegotiation
        with self.assertRaises(ValueError):
            BilateralMergerNegotiation(100, 70, 10, 0, 20, 30, 3, offer_step=0)

if __name__ == '__main__':
    unittest.main()
