"""Legal-move lookup and plain-English illegality explanations for the live tutor UI.

Used by web/server.py to answer two questions the frontend asks while a puzzle is
in progress (never advances the puzzle, never scores an attempt):
  - "what are the legal destinations for the piece on this square?" (legal_targets)
  - "why isn't this move legal?" (explain_illegal_move)
"""
from __future__ import annotations
import chess

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}

def legal_targets(board: chess.Board, square_name: str) -> list[str]:
    """Legal destination squares (as square names) for the piece on square_name,
    for whichever side is to move. Empty if there's no piece there, or it belongs
    to the side NOT to move, or square_name isn't a real square."""
    try:
        square = chess.parse_square(square_name)
    except ValueError:
        return []
    piece = board.piece_at(square)
    if piece is None or piece.color != board.turn:
        return []
    return sorted(chess.square_name(m.to_square) for m in board.legal_moves if m.from_square == square)

def explain_illegal_move(board: chess.Board, move_uci: str) -> str:
    """A short, human-readable reason move_uci is illegal on this board. Only
    meaningful to call when the move is already known not to be legal."""
    try:
        move = chess.Move.from_uci(move_uci)
    except (ValueError, IndexError):
        return "That doesn't look like a real move."

    piece = board.piece_at(move.from_square)
    if piece is None:
        return f"There's no piece on {chess.square_name(move.from_square)}."
    if piece.color != board.turn:
        return "That's not your piece to move."

    dest_piece = board.piece_at(move.to_square)
    if dest_piece is not None and dest_piece.color == piece.color:
        return "You already have a piece on that square."

    if board.is_pseudo_legal(move) and move not in board.legal_moves:
        return "That move would leave your king in check."

    if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        between = chess.SquareSet(chess.between(move.from_square, move.to_square))
        if any(board.piece_at(sq) is not None for sq in between):
            return "There's a piece blocking that path."

    name = PIECE_NAMES[piece.piece_type]
    return f"A {name} can't move like that."
