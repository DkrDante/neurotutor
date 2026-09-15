import random
import pytest
from adaptive.rl_policy import ACTIONS, QLearningPolicy

def test_decide_returns_one_of_the_defined_actions():
    policy = QLearningPolicy(rng=random.Random(0))
    action = policy.decide("Focused", confidence=0.9, correct=True)
    assert action in ACTIONS

def test_unknown_state_raises():
    policy = QLearningPolicy(rng=random.Random(0))
    with pytest.raises(ValueError):
        policy.decide("NotAState", confidence=0.9, correct=True)

def test_q_table_starts_empty_and_grows_as_states_are_visited():
    policy = QLearningPolicy(rng=random.Random(0))
    assert policy.q_table == {}
    policy.decide("Focused", confidence=0.9, correct=True)
    assert ("Focused", "high") in policy.q_table

def test_repeated_reward_increases_the_chosen_actions_q_value():
    # epsilon=0 (pure greedy) with a fixed seed makes this deterministic: the
    # same action gets picked every time a state repeats, so its Q-value must
    # strictly increase as long as the reward keeps being positive.
    policy = QLearningPolicy(epsilon=0.0, rng=random.Random(0))
    policy.decide("Focused", confidence=0.9, correct=True)  # establishes the pending (state, action)
    key = ("Focused", "high")
    q_before = list(policy.q_table[key])

    policy.decide("Focused", confidence=0.9, correct=True)  # rewards the previous action with +1
    q_after = policy.q_table[key]

    chosen_idx = max(range(len(q_before)), key=lambda i: q_before[i])
    assert q_after[chosen_idx] > q_before[chosen_idx]

def test_negative_reward_decreases_the_chosen_actions_q_value():
    policy = QLearningPolicy(epsilon=0.0, rng=random.Random(0))
    policy.decide("Confused", confidence=0.9, correct=True)
    key = ("Confused", "high")
    q_before = list(policy.q_table[key])

    policy.decide("Confused", confidence=0.9, correct=False)  # penalizes the previous action
    q_after = policy.q_table[key]

    chosen_idx = max(range(len(q_before)), key=lambda i: q_before[i])
    assert q_after[chosen_idx] < q_before[chosen_idx]

def test_epsilon_zero_is_deterministic_once_a_best_action_is_learned():
    policy = QLearningPolicy(epsilon=0.0, rng=random.Random(0))
    key = ("Engaged", "high")
    policy.q_table[key] = [0.0, 0.0, 0.0, 5.0, 0.0]  # action index 3 is clearly best
    action = policy.decide("Engaged", confidence=0.9, correct=True)
    assert action == ACTIONS[3]

def test_state_round_trips_through_to_state_and_from_state():
    policy = QLearningPolicy(alpha=0.5, gamma=0.9, epsilon=0.2, rng=random.Random(0))
    policy.decide("Focused", confidence=0.9, correct=True)
    policy.decide("Focused", confidence=0.9, correct=True)
    policy.decide("Fatigued", confidence=0.5, correct=False)

    restored = QLearningPolicy.from_state(policy.to_state())

    assert restored.alpha == 0.5
    assert restored.gamma == 0.9
    assert restored.epsilon == 0.2
    assert restored.q_table == policy.q_table

def test_from_state_with_empty_data_produces_a_fresh_policy():
    policy = QLearningPolicy.from_state({})
    assert policy.q_table == {}
    assert policy.alpha == 0.3
