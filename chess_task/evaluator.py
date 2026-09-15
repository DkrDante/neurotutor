from __future__ import annotations
from abc import ABC, abstractmethod
import math
import random
import shutil
import chess
import chess.engine

# Standard piece values (centipawns) for the material-only position estimate
# used when no engine is available — real and computed, just not engine-strength.
MATERIAL_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900,
}

class MoveEvaluator(ABC):
    @abstractmethod
    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        raise NotImplementedError

    def evaluate_position(self, board: chess.Board) -> float:
        """Centipawn evaluation of the CURRENT position from White's perspective
        (positive favors White). Base implementation: pure material count — a real,
        computed number, just not engine-strength. StockfishEvaluator overrides this
        with an actual engine score."""
        score = 0
        for piece_type, value in MATERIAL_VALUES.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        return float(score)

    def best_move(self, board: chess.Board) -> chess.Move:
        """The engine's own top move for this position — used both to grade a
        played move's quality (eval_loss) and, in full-game mode, as the AI
        opponent's strongest possible reply. Base implementation: prefer the
        highest-value capture if one exists, else a random legal move — a real,
        computed choice, just not engine-strength (see StockfishEvaluator)."""
        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures:
            def capture_value(move: chess.Move) -> int:
                if board.is_en_passant(move):
                    return MATERIAL_VALUES[chess.PAWN]
                captured = board.piece_at(move.to_square)
                return MATERIAL_VALUES.get(captured.piece_type, 0) if captured else 0
            return max(captures, key=capture_value)
        return random.choice(legal_moves)

    def close(self) -> None:
        """Release any resources (e.g. engine subprocesses). No-op by default."""
        return None

def win_probability(eval_cp: float) -> float:
    """Maps a centipawn evaluation (White's perspective) to White's win probability
    in [0, 1], using the same logistic approximation lichess uses for its eval bar:
    P(White wins) = 1 / (1 + exp(-0.00368208 * cp)). This is a display heuristic,
    not a calibrated probability — it just gives a smooth, intuitive percentage
    instead of a raw, unbounded centipawn number."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * eval_cp))

class StockfishEvaluator(MoveEvaluator):
    def __init__(self, binary_path: str = "stockfish", depth: int = 10):
        if shutil.which(binary_path) is None:
            raise FileNotFoundError(f"Stockfish binary not found: {binary_path}")
        self.binary_path = binary_path
        self.depth = depth
        # One long-lived engine subprocess for the evaluator's lifetime: spawning a
        # fresh process per _score() call dominated the cost of every incorrect move.
        self._engine = chess.engine.SimpleEngine.popen_uci(binary_path)

    def _score(self, board: chess.Board, move: chess.Move) -> int:
        board = board.copy()
        board.push(move)
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].pov(not board.turn)
        return score.score(mate_score=10000)

    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        best_score = self._score(board, best_move)
        played_score = self._score(board, played_move)
        return float(max(0, best_score - played_score))

    def evaluate_position(self, board: chess.Board) -> float:
        # A shallower depth than eval_loss's move comparison: this runs on every
        # puzzle load and every move (for the live win-probability bar), not just
        # on a wrong answer, so it needs to stay fast.
        info = self._engine.analyse(board, chess.engine.Limit(depth=min(self.depth, 8)))
        score = info["score"].pov(chess.WHITE)
        return float(score.score(mate_score=10000))

    def best_move(self, board: chess.Board) -> chess.Move:
        result = self._engine.play(board, chess.engine.Limit(depth=self.depth))
        assert result.move is not None  # only None for an already-game-over board
        return result.move

    def close(self) -> None:
        self._engine.quit()
