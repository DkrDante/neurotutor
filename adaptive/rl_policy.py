"""RL-based adaptive policy — the design spec's "RL-based adaptive policy" and
"multi-session personalization" longer-term extensions, now implemented as a
tabular Q-learning agent.

Why tabular Q-learning rather than deep RL: the state space here is
deliberately small (5 cognitive states x 3 confidence buckets = 15 states)
and the action space is a handful of meaningful difficulty/hint/pacing
choices — exactly the regime where a Q-table converges fast, stays fully
inspectable (you can print the whole learned policy), and needs no training
infrastructure (GPU, replay buffer, function approximation) to work online,
from a single learner's live session. A deep network would be substantially
more machinery for a problem this small.

Learning signal: decide() is called once per attempt with that attempt's own
`correct` flag — which is exactly the reward for the PREVIOUS decision (did
the difficulty/hint/pacing choice made last time lead to a good outcome this
time?). So each call first updates the Q-value for the last (state, action)
pair using this attempt's `correct` as the reward, then picks the next
action — no separate reward callback needed, and the Policy interface stays
identical to RuleBasedPolicy's.

Multi-session personalization: to_state() / from_state() (de)serialize the
whole learned Q-table to a JSON-safe structure, so web/server.py can persist
it per learner (storage.db.SessionStore's learner_profiles table) and reload
it on the learner's next visit — the policy keeps improving across sessions
instead of starting from a blank slate each time.
"""
from __future__ import annotations
import random
from adaptive.policy import Action, Policy
from common.states import STATES

# Same five meta-actions RuleBasedPolicy's STATE_RULES offers (a hint+pacing
# response paired with a difficulty push) — the RL agent's job is to LEARN
# which of these to pick given the situation, rather than following a fixed
# per-state lookup table.
ACTIONS: list[Action] = [
    Action(difficulty_delta=-150.0, show_hint=True, pacing_delay=3.0),
    Action(difficulty_delta=-75.0, show_hint=True, pacing_delay=1.5),
    Action(difficulty_delta=0.0, show_hint=False, pacing_delay=0.0),
    Action(difficulty_delta=75.0, show_hint=False, pacing_delay=0.0),
    Action(difficulty_delta=150.0, show_hint=False, pacing_delay=0.0),
]

def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.4:
        return "low"
    if confidence < 0.7:
        return "medium"
    return "high"

class QLearningPolicy(Policy):
    def __init__(
        self, alpha: float = 0.3, gamma: float = 0.5, epsilon: float = 0.15,
        q_table: dict[tuple[str, str], list[float]] | None = None, rng: random.Random | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[tuple[str, str], list[float]] = q_table if q_table is not None else {}
        self._rng = rng or random.Random()
        # The (state_key, action_index) this policy is still waiting to learn
        # a reward for — set by the previous decide() call, consumed by the next.
        self._pending: tuple[tuple[str, str], int] | None = None

    def _state_key(self, state: str, confidence: float) -> tuple[str, str]:
        return (state, _confidence_bucket(confidence))

    def _q_row(self, key: tuple[str, str]) -> list[float]:
        return self.q_table.setdefault(key, [0.0] * len(ACTIONS))

    def decide(self, state: str, confidence: float, correct: bool) -> Action:
        if state not in STATES:
            raise ValueError(f"Unknown state: {state}")

        if self._pending is not None:
            self._learn_from_outcome(current_state_key=self._state_key(state, confidence), reward=1.0 if correct else -1.0)

        key = self._state_key(state, confidence)
        row = self._q_row(key)
        if self._rng.random() < self.epsilon:
            action_idx = self._rng.randrange(len(ACTIONS))
        else:
            action_idx = max(range(len(row)), key=lambda i: row[i])

        self._pending = (key, action_idx)
        return ACTIONS[action_idx]

    def _learn_from_outcome(self, current_state_key: tuple[str, str], reward: float) -> None:
        assert self._pending is not None  # only called when decide() has just checked this
        prev_key, prev_action_idx = self._pending
        prev_row = self._q_row(prev_key)
        best_next_value = max(self._q_row(current_state_key))
        # Standard Q-learning update: Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        prev_row[prev_action_idx] += self.alpha * (reward + self.gamma * best_next_value - prev_row[prev_action_idx])

    def to_state(self) -> dict:
        """A JSON-safe snapshot of everything needed to resume this policy
        later — the learned Q-table plus its hyperparameters."""
        return {
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
            "q_table": [{"state": list(key), "values": values} for key, values in self.q_table.items()],
        }

    @classmethod
    def from_state(cls, data: dict) -> "QLearningPolicy":
        q_table = {tuple(row["state"]): list(row["values"]) for row in data.get("q_table", [])}
        return cls(
            alpha=data.get("alpha", 0.3), gamma=data.get("gamma", 0.5),
            epsilon=data.get("epsilon", 0.15), q_table=q_table,
        )
