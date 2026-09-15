"""Tactical/mate-pattern tagging for the curated puzzle set — every puzzle in
chess_task/puzzle_data/sample_puzzles.csv delivers checkmate in its final
move, so "motif" here means the real, computable mate PATTERN (back-rank,
smothered, discovered/double check, or which piece actually delivers mate),
not a general tactical-motif classifier for mid-game positions. Every label
is derived from the real post-move board via python-chess — nothing here is
guessed or looked up from an external dataset.
"""
from __future__ import annotations
import chess

MOTIFS = [
    "smothered", "back_rank", "double_check", "discovered_check",
    "queen_mate", "rook_mate", "knight_mate", "bishop_mate", "pawn_mate", "other",
]


def _is_back_rank_mate(board: chess.Board, king_square: int) -> bool:
    """The mated king sits on its own back rank (rank 1 for White, rank 8
    for Black) with every square it could otherwise step to along that rank
    blocked by its own pawns — the textbook back-rank pattern."""
    rank = chess.square_rank(king_square)
    back_rank = 0 if board.color_at(king_square) == chess.WHITE else 7
    if rank != back_rank:
        return False
    file = chess.square_file(king_square)
    escape_files = [f for f in (file - 1, file + 1) if 0 <= f <= 7]
    if not escape_files:
        return False
    for escape_file in escape_files:
        square_in_front = chess.square(escape_file, rank + (1 if back_rank == 0 else -1))
        piece = board.piece_at(square_in_front)
        if piece is None or piece.color != board.color_at(king_square):
            return False
    return True


def _is_smothered_mate(board: chess.Board, king_square: int) -> bool:
    """Every square adjacent to the mated king is occupied by one of the
    king's OWN pieces — the classic smothered-mate signature, almost always
    delivered by a knight since only a knight can check from a square its
    own king can't otherwise be reached from."""
    king_color = board.color_at(king_square)
    neighbors = [
        sq for sq in chess.SQUARES
        if sq != king_square and chess.square_distance(sq, king_square) == 1
    ]
    if not neighbors:
        return False
    for sq in neighbors:
        piece = board.piece_at(sq)
        if piece is None or piece.color != king_color:
            return False
    return True


def classify_mate_pattern(fen_before_final_move: str, final_move_uci: str) -> str:
    """The real classification: applies the puzzle's actual final move to the
    actual position and inspects the resulting board — not a lookup table."""
    board = chess.Board(fen_before_final_move)
    move = chess.Move.from_uci(final_move_uci)
    mating_piece_type = board.piece_type_at(move.from_square)
    board.push(move)

    if not board.is_checkmate():
        return "other"

    king_square = board.king(board.turn)
    checkers = board.checkers()

    if len(checkers) >= 2:
        return "double_check"
    if _is_smothered_mate(board, king_square):
        return "smothered"
    if _is_back_rank_mate(board, king_square):
        return "back_rank"
    if move.to_square not in checkers:
        # The piece that just moved isn't the one giving check — some other,
        # already-placed piece's line was opened by this move.
        return "discovered_check"

    return {
        chess.QUEEN: "queen_mate", chess.ROOK: "rook_mate", chess.KNIGHT: "knight_mate",
        chess.BISHOP: "bishop_mate", chess.PAWN: "pawn_mate",
    }.get(mating_piece_type, "other")


def classify_puzzle(fen: str, solution_moves: list[str]) -> str:
    """Classifies a full puzzle (possibly multi-move) by its FINAL move —
    the one that actually delivers mate. Earlier moves (solver move,
    scripted opponent reply, ...) are replayed first to reach the real
    position the mating move is played from."""
    board = chess.Board(fen)
    for move_uci in solution_moves[:-1]:
        board.push(chess.Move.from_uci(move_uci))
    return classify_mate_pattern(board.fen(), solution_moves[-1])
