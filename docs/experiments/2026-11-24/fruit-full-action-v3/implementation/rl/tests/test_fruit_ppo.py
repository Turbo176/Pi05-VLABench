"""CPU regressions for reward accounting and effective actor/critic updates."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from train_fruit_ppo import (CORRECTED_REWARD, GAMMA, LEGACY_REWARD, check_replay, finish_returns,
                            fruit_reward, optimize, parameter_groups, ppo_terms, replay_scores,
                            value_metrics)


class RewardTests(unittest.TestCase):
    def test_legacy_equation_is_unchanged_at_terminal(self):
        reward = fruit_reward(-0.8, -0.2, True, True, 0.5, LEGACY_REWARD)
        self.assertAlmostEqual(reward['reward'], 1 + 0.5 * (-0.2 + 0.8))
        self.assertEqual(reward['effective_potential_after'], -0.2)

    def test_discounted_terminal_identity_for_random_complete_episodes(self):
        rng = np.random.default_rng(20261123)
        for _ in range(100):
            count = int(rng.integers(1, 121))
            potentials = rng.uniform(-1.2, 0, count + 1)
            solved = bool(rng.integers(0, 2))
            shaped = sparse = 0.0
            for index in range(count):
                reward = fruit_reward(potentials[index], potentials[index + 1],
                                      solved and index == count - 1, index == count - 1,
                                      0.5, CORRECTED_REWARD)
                shaped += GAMMA ** index * reward['reward']
                sparse += GAMMA ** index * reward['official_reward']
            self.assertAlmostEqual(shaped - sparse, -0.5 * potentials[0], places=12)

    def test_success_always_sets_terminal_potential_to_zero(self):
        reward = fruit_reward(-0.5, -0.1, True, False, 0.5, CORRECTED_REWARD)
        self.assertTrue(reward['terminal'])
        self.assertEqual(reward['effective_potential_after'], 0)
        self.assertAlmostEqual(reward['reward'], 1.25)

    def test_zero_shaping_and_invalid_inputs(self):
        self.assertEqual(fruit_reward(-0.8, -0.1, False, True, 0, CORRECTED_REWARD)['reward'], 0)
        with self.assertRaises(ValueError):
            fruit_reward(float('nan'), 0, False, False, 0.5, CORRECTED_REWARD)
        with self.assertRaises(ValueError):
            fruit_reward(0, 0, False, False, 0.5, 'unknown')

    def test_full_chunk_and_one_control_tail_use_one_chunk_discount(self):
        episode = [dict(reward=0.2, value=0.3, terminal=False, executed=5),
                   dict(reward=1., value=0.4, terminal=True, executed=1)]
        finish_returns(episode)
        self.assertAlmostEqual(episode[0]['mc_return'], 0.2 + GAMMA)
        self.assertAlmostEqual(episode[-1]['return'], 1.)
        expected = 0.2 + GAMMA * 0.4 - 0.3 + GAMMA * 0.95 * (1 - 0.4)
        self.assertAlmostEqual(episode[0]['advantage'], expected)
        with self.assertRaises(ValueError):
            finish_returns([dict(reward=0., value=0., terminal=False)])

    def test_clip_and_kl_are_calculated_from_same_prefix_score(self):
        positive = ppo_terms(torch.tensor(math.log(1.5)), 0., torch.tensor(0.), 1., 1.)
        negative = ppo_terms(torch.tensor(math.log(0.5)), 0., torch.tensor(0.), 1., -1.)
        self.assertAlmostEqual(float(positive['actor_loss']), -1.2, places=6)
        self.assertAlmostEqual(float(negative['actor_loss']), 0.8, places=6)
        self.assertEqual(float(positive['clip_fraction']), 1.)
        self.assertGreater(float(positive['approx_kl']), 0.)
        self.assertGreater(float(negative['approx_kl']), 0.)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = torch.nn.Linear(2, 1)
        self.value_head = torch.nn.Linear(2, 1)

    def replay(self, transition):
        features = transition['features']
        mean = self.action_head(features).reshape(())
        logp = -0.5 * (mean - transition['action']) ** 2
        # Deliberately includes actor features: the warmup must still freeze them.
        value = self.value_head(features + mean).reshape(())
        return logp, value


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.model = TinyPolicy()
        actor, critic = parameter_groups(self.model)
        self.optimizer = torch.optim.AdamW([dict(params=actor, lr=0.001),
                                            dict(params=critic, lr=0.01)], weight_decay=0)
        self.transitions = []
        for index in range(8):
            transition = dict(features=torch.tensor([index / 8., 1.]), action=0.8 - index / 10.,
                              executed=1 if index == 7 else 5, action_dim=7,
                              advantage=float(index - 3), mc_return=1.0, reward=1.0, **{'return': 1.0})
            with torch.no_grad():
                logp, value = self.model.replay(transition)
            transition.update(old_logp=float(logp), value=float(value))
            self.transitions.append(transition)

    def snapshot(self):
        return {name: value.detach().clone() for name, value in self.model.named_parameters()}

    def test_critic_warmup_preserves_actor_and_optimizer_state(self):
        before = self.snapshot()
        history = optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                           np.random.default_rng(1), critic_only=True, epochs=2)
        for name, parameter in self.model.named_parameters():
            if name.startswith('action_head.'):
                self.assertTrue(torch.equal(parameter, before[name]))
                self.assertTrue(parameter.requires_grad)
                self.assertNotIn(parameter, self.optimizer.state)
                self.assertIsNone(parameter.grad)
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in
                            self.model.named_parameters() if name.startswith('value_head.')))
        self.assertTrue(all(row['actor_grad_l2'] == 0 and row['actor_max_abs_delta_this_step'] == 0
                            for row in history))

    def test_ppo_changes_actor_and_records_finite_diagnostics(self):
        before = self.snapshot()
        history = optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                           np.random.default_rng(1), minibatch=4)
        self.assertTrue(any(not torch.equal(parameter, before[name]) for name, parameter in
                            self.model.named_parameters() if name.startswith('action_head.')))
        self.assertGreater(history[0]['actor_grad_l2'], 0)
        self.assertGreater(history[0]['actor_max_abs_delta_this_step'], 0)
        self.assertAlmostEqual(history[0]['approx_kl'], 0., places=6)
        self.assertEqual(sum(sum(row['executed_lengths'].values()) for row in history), 8)
        json.dumps(history, allow_nan=False)

    def test_kl_guard_discards_update_and_clears_gradients(self):
        before = self.snapshot()
        for transition in self.transitions:
            transition['old_logp'] -= 1.
        history = optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                           np.random.default_rng(1), minibatch=8)
        self.assertTrue(history[0]['stopped_for_kl'])
        self.assertFalse(history[0]['optimizer_step_applied'])
        self.assertFalse(self.optimizer.state)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
            self.assertIsNone(parameter.grad)

    def test_nonfinite_ratio_does_not_step_or_hide_overflow(self):
        for transition in self.transitions:
            transition['old_logp'] = -1000.
        with self.assertRaises(FloatingPointError):
            optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                     np.random.default_rng(1))
        self.assertFalse(self.optimizer.state)

    def test_large_last_step_rolls_back_parameters_and_optimizer_state(self):
        self.optimizer.param_groups[0]['lr'] = 1.0
        before = self.snapshot()
        history = optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                           np.random.default_rng(1), minibatch=8)
        self.assertTrue(history[0]['poststep_reverted_for_kl'])
        self.assertFalse(history[0]['optimizer_step_applied'])
        self.assertGreater(history[0]['postupdate_approx_kl'], 0.02)
        self.assertFalse(self.optimizer.state)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_zero_reward_only_fits_critic_and_cannot_claim_actor_update(self):
        before = self.snapshot()
        for transition in self.transitions:
            transition.update(reward=0., mc_return=0., **{'return': 0.})
        history = optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                           np.random.default_rng(1))
        self.assertEqual(history[0]['actor_update_skipped_reason'], 'all_chunk_rewards_zero')
        for name, parameter in self.model.named_parameters():
            if name.startswith('action_head.'):
                self.assertTrue(torch.equal(parameter, before[name]))
                self.assertNotIn(parameter, self.optimizer.state)

    def test_poststep_replay_exception_restores_existing_adamw_state(self):
        optimize(self.model, self.optimizer, self.transitions, self.model.replay,
                 np.random.default_rng(1), critic_only=True)
        before, state = self.snapshot(), copy.deepcopy(self.optimizer.state_dict())
        records, calls = [], 0

        def invalid_after_step(transition):
            nonlocal calls
            calls += 1
            if calls > len(self.transitions):
                raise ValueError('nonfinite post-step fixture')
            return self.model.replay(transition)

        with self.assertRaisesRegex(RuntimeError, 'were rolled back'):
            optimize(self.model, self.optimizer, self.transitions, invalid_after_step,
                     np.random.default_rng(1), record=records.append)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
            self.assertIsNone(parameter.grad)
        restored = self.optimizer.state_dict()
        self.assertEqual(restored['param_groups'], state['param_groups'])
        self.assertEqual(set(restored['state']), set(state['state']))
        for key, expected in state['state'].items():
            for field, value in expected.items():
                torch.testing.assert_close(restored['state'][key][field], value, rtol=0, atol=0)
        self.assertTrue(records[0]['poststep_reverted_for_kl'])

    def test_actor_flags_restored_after_failed_warmup(self):
        def invalid_replay(transition):
            raise RuntimeError('fixture')

        with self.assertRaises(RuntimeError):
            optimize(self.model, self.optimizer, self.transitions, invalid_replay,
                     np.random.default_rng(1), critic_only=True)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

    def test_constant_value_targets_have_no_invented_explained_variance(self):
        result = value_metrics(self.transitions)
        self.assertIsNone(result['mc_explained_variance'])
        self.assertGreater(result['mc_value_mse'], 0)


class ReplayTests(unittest.TestCase):
    def test_opposite_coordinate_errors_are_logged_before_raising(self):
        transition = dict(old_logprobs=torch.zeros(1, 2), old_logp=0.)
        records = []
        with self.assertRaisesRegex(RuntimeError, 'coordinate=0.02'):
            check_replay([transition], lambda t: torch.tensor([[0.02, -0.02]]), records.append)
        self.assertFalse(records[0]['passed'])
        self.assertEqual(records[0]['aggregate_max_abs_error'], 0.)
        self.assertGreater(records[0]['coordinate_max_abs_error'], 0.005)

    def test_replay_preserves_coordinate_shape_and_partial_chunk(self):
        class Observation:
            state = torch.zeros(1, 7)
            tokenized_prompt = torch.zeros(1, 1, dtype=torch.long)
            tokenized_prompt_mask = torch.ones(1, 1, dtype=torch.bool)
            images = {'base_0_rgb': torch.zeros(1, 3, 2, 2)}
            image_masks = {'base_0_rgb': torch.ones(1, dtype=torch.bool)}

        class Encoding:
            def encode(self, row):
                return Observation(), None

        class Model:
            def default_forward(self, inputs):
                return dict(logprobs=torch.arange(35.).reshape(1, 5, 7), values=torch.tensor([[0.2]]))

        transition = dict(row={}, chain=torch.zeros(1, 2, 5, 7), denoise_inds=torch.zeros(1, 1),
                          executed=1, action_dim=7)
        scores, value = replay_scores(Model(), Encoding(), transition)
        self.assertEqual(scores.shape, (1, 7))
        self.assertEqual(float(scores.sum()), 21.)
        self.assertAlmostEqual(float(value), 0.2)
        transition['action_dim'] = 6
        with self.assertRaises(ValueError):
            replay_scores(Model(), Encoding(), transition)


if __name__ == '__main__':
    unittest.main(verbosity=2)

