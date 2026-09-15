"""Full-game chess mode: play an entire game against an adaptive-strength AI
opponent, instead of solving discrete tactics puzzles. The design spec
(docs/superpowers/specs/2026-09-10-chess-eeg-tutor-design.md) scoped
PuzzleTaskEngine as the only implementation and left full-game mode as an
interface stub ("Full-game mode: interface defined (TaskEngine base), not
implemented") — this is that implementation.

FullGameEngine implements TaskEngine's exact two methods so it plugs into the
existing TutorSession/server orchestration unchanged, but the fit isn't a
literal 1:1 rename of "puzzle" concepts:
  - There is no fixed, curated "solution move". get_puzzle() packages the
    CURRENT live position plus the evaluator's own best move for it — that
    best move is a strong suggestion, not the one right answer a tactics
    puzzle has.
  - submit_move() grades "correct" by how close the played move's eval_loss
    is to that best move (within CORRECT_EVAL_LOSS_THRESHOLD centipawns),
    not by exact match — an ordinary game move is rarely THE single best
    move, and grading it as "wrong" whenever it isn't would be meaningless.
  - Unlike puzzle mode, nothing about this engine resets a rejected move or
    scripts the opponent's reply: web/server.py's game-mode loop pushes
    whatever the player actually played (final, as in a real game) and then
    asks pick_opponent_move() for a live, unscripted reply.
"""
from __future__ import annotations
import random
import chess
from chess_task.base import BehaviorEvent, Puzzle, TaskEngine
from chess_task.evaluator import MoveEvaluator

# A played move within this many centipawns of the evaluator's own best move
# counts as "correct" — full games don't have one right answer per move the
# way a tactics puzzle does, so exact-match grading isn't meaningful here.
CORRECT_EVAL_LOSS_THRESHOLD = 30.0

# Difficulty range the AI opponent's strength is scaled across — 400 is
# TutorSession's difficulty floor (web/session.py), 2600 covers the puzzle
# set's real rating ceiling (see chess_task/puzzle_data/sample_puzzles.csv).
_STRENGTH_FLOOR = 400.0
_STRENGTH_SPAN = 2200.0

# How long the AI opponent "thinks" before its move is revealed, in seconds —
# scales with the same strength signal that governs how it plays, so a
# stronger opponent visibly takes a bit longer, like a deeper engine search.
_MIN_THINKING_SECONDS = 0.4
_MAX_THINKING_SECONDS = 1.6

class FullGameEngine(TaskEngine):
    def __init__(self, evaluator: MoveEvaluator, starting_fen: str | None = None, rng: random.Random | None = None):
        self.evaluator = evaluator
        self.board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self._ply = 0
        self._rng = rng or random.Random()

    def get_puzzle(self, difficulty: float, rating_min: float | None = None, rating_max: float | None = None) -> Puzzle:
        """Packages the CURRENT live position — not a fixed puzzle — plus the
        evaluator's best move for it, so the existing hint/explain-suboptimal-
        move code (which reads puzzle.solution_move) keeps working unchanged.
        rating_min/rating_max (targeted practice on a weak rating band) don't
        apply here — a full game has no puzzle pool to filter."""
        best = self.evaluator.best_move(self.board)
        return Puzzle(
            puzzle_id=f"game-ply-{self._ply}", fen=self.board.fen(),
            solution_move=best.uci(), rating=int(difficulty),
        )

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        """Grades the move already known to be legal (server.py checks legality
        before calling this, same as puzzle mode) — does NOT push it onto
        self.board; the caller controls exactly when that happens (see
        push_player_move), same separation PuzzleTaskEngine.submit_move keeps."""
        played = chess.Move.from_uci(move_uci)
        best = chess.Move.from_uci(puzzle.solution_move)
        eval_loss = 0.0 if played == best else self.evaluator.eval_loss(self.board, played, best)
        correct = eval_loss <= CORRECT_EVAL_LOSS_THRESHOLD
        return BehaviorEvent(
            puzzle_id=puzzle.puzzle_id, correct=correct, time_to_move=time_to_move,
            eval_loss=eval_loss, puzzle_rating=puzzle.rating,
        )

    def push_player_move(self, move_uci: str) -> None:
        self.board.push(chess.Move.from_uci(move_uci))
        self._ply += 1

    @staticmethod
    def _strength(difficulty: float) -> float:
        """[0, 1], linear in difficulty — not a calibrated Elo mapping, just a
        simple, real, inspectable scale shared by move choice and think time."""
        return max(0.0, min(1.0, (difficulty - _STRENGTH_FLOOR) / _STRENGTH_SPAN))

    def pick_opponent_move(self, difficulty: float) -> chess.Move:
        """The AI opponent's reply, strength-scaled to the SAME adaptive
        difficulty signal that already governs puzzle rating/hints/pacing —
        the harder the session gets, the more often the opponent plays its
        real best move instead of a random legal one."""
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]
        if self._rng.random() < self._strength(difficulty):
            return self.evaluator.best_move(self.board)
        return self._rng.choice(legal_moves)

    def thinking_time(self, difficulty: float) -> float:
        """Seconds to pause before revealing the opponent's move — purely a UX
        pacing device (the move above is already computed instantly), scaled
        by the same strength signal so a "stronger" opponent visibly takes a
        little longer, evoking a deeper search rather than an instant reflex."""
        return _MIN_THINKING_SECONDS + self._strength(difficulty) * (_MAX_THINKING_SECONDS - _MIN_THINKING_SECONDS)

    def push_opponent_move(self, move: chess.Move) -> None:
        self.board.push(move)
        self._ply += 1
