from __future__ import annotations
from abc import ABC, abstractmethod
import shutil
import chess
import chess.engine

class MoveEvaluator(ABC):
    @abstractmethod
    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        raise NotImplementedError

class StockfishEvaluator(MoveEvaluator):
    def __init__(self, binary_path: str = "stockfish", depth: int = 10):
        if shutil.which(binary_path) is None:
            raise FileNotFoundError(f"Stockfish binary not found: {binary_path}")
        self.binary_path = binary_path
        self.depth = depth

    def _score(self, board: chess.Board, move: chess.Move) -> int:
        board = board.copy()
        board.push(move)
        with chess.engine.SimpleEngine.popen_uci(self.binary_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].pov(not board.turn)
            return score.score(mate_score=10000)

    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        best_score = self._score(board, best_move)
        played_score = self._score(board, played_move)
        return float(max(0, best_score - played_score))
