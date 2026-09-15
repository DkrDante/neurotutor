"""Legal-move lookup and plain-English move explanations for the live tutor UI.

Used by web/server.py to answer four questions the frontend asks while a puzzle
is in progress (never advances the puzzle, never scores an attempt unless noted):
  - "what are the legal destinations for the piece on this square?" (legal_targets)
  - "why isn't this move legal?" (explain_illegal_move)
  - "what did my move actually do?" (describe_move) — factual, not a verdict.
  - "why isn't this legal move the puzzle's solution, and how good was it
    anyway?" (score_move + explain_suboptimal_move) — called by server.py only
    on a scored, legal-but-wrong attempt (puzzle-mode "retry" or game-mode
    "incorrect"), never to decide legality.

explain_suboptimal_move's explanation scales with `tier` ("high"/"medium"/
"low" hand-holding — see web/server.py's hand_holding_tier(), which derives it
from the session's current adaptive difficulty): someone whose difficulty has
been dropping gets the fuller, more didactic version; someone cruising at a
high difficulty gets the terse one. "medium" matches this module's original,
single-tier behavior exactly, so existing callers that don't pass `tier` are
unaffected.
"""
from __future__ import annotations
import chess

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# (max eval_loss in centipawns this label covers, label) — first match wins,
# checked in ascending order. Mirrors the familiar lichess/chess.com move-
# quality vocabulary so it reads as a real chess score, not an invented scale.
MOVE_QUALITY_BANDS = [
    (0, "Best"),
    (20, "Excellent"),
    (60, "Good"),
    (150, "Inaccuracy"),
    (300, "Mistake"),
    (float("inf"), "Blunder"),
]


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


def describe_move(board: chess.Board, move_uci: str) -> str:
    """A factual, verdict-free one-liner for what the played move actually did
    — piece, from/to, capture, check — computed fresh from the board rather
    than reusing the SAN legend already shown client-side, so it reads as a
    sentence rather than a symbol-by-symbol breakdown."""
    try:
        move = chess.Move.from_uci(move_uci)
    except (ValueError, IndexError):
        return "That move couldn't be parsed."

    piece = board.piece_at(move.from_square)
    piece_name = PIECE_NAMES[piece.piece_type] if piece else "piece"
    from_sq, to_sq = chess.square_name(move.from_square), chess.square_name(move.to_square)

    captured_name = None
    if board.is_capture(move):
        if board.is_en_passant(move):
            captured_name = "pawn"
        else:
            captured = board.piece_at(move.to_square)
            captured_name = PIECE_NAMES[captured.piece_type] if captured else "piece"

    trial = board.copy()
    trial.push(move)
    gives_check = trial.is_check()

    sentence = f"You moved the {piece_name} from {from_sq} to {to_sq}"
    if captured_name:
        sentence += f", capturing the {captured_name}"
    if gives_check:
        sentence += ", giving check"
    return sentence + "."


def score_move(eval_loss: float) -> dict:
    """A 0-100 move-quality score plus a familiar label, from the same
    eval_loss (centipawns) already computed for every attempt. Score formula:
    100 - eval_loss / 5, floored at 0 — simple and stated once, here."""
    score = round(max(0.0, 100.0 - eval_loss / 5.0))
    label = next(label for threshold, label in MOVE_QUALITY_BANDS if eval_loss <= threshold)
    return {"score": score, "label": label}


def _san_safe(board: chess.Board, move_uci: str | None) -> str | None:
    if not move_uci:
        return None
    try:
        return board.san(chess.Move.from_uci(move_uci))
    except (ValueError, IndexError, AssertionError):
        return move_uci


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    """-1 if move isn't a capture, else the value of the piece it captures."""
    if not board.is_capture(move):
        return -1
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return PIECE_VALUES[captured.piece_type] if captured else -1


def _hangs_piece(board: chess.Board, move: chess.Move) -> chess.Square | None:
    """After playing move, is the moved piece left on a square the opponent attacks
    and the mover doesn't defend? Returns that square, else None."""
    trial = board.copy()
    trial.push(move)
    to_square = move.to_square
    piece = trial.piece_at(to_square)
    if piece is None or piece.piece_type == chess.KING:
        return None
    attackers = trial.attackers(not piece.color, to_square)
    defenders = trial.attackers(piece.color, to_square)
    return to_square if attackers and not defenders else None


def explain_suboptimal_move(
    board: chess.Board, played_uci: str, solution_uci: str, tier: str = "medium", reveal_solution: bool = True,
) -> str:
    """A human-readable reason a LEGAL move isn't the puzzle's solution, at one
    of three hand-holding levels ("high"/"medium"/"low" — see hand_holding_tier
    in web/server.py). Grounded in the actual position and the known correct
    move (not a full engine search): checks for a missed forced mate, a hung
    piece, a bigger capture left on the board, or a missed check, in that
    order, before falling back to a generic "there's a better move" message
    naming the real solution.

    reveal_solution=False (the caller's default is now to withhold it — see
    web/server.py's _run_puzzle_mode) redacts the solution's actual notation
    from the text, so a player only sees it after explicitly asking (the
    "Solution" button) or after auto-reveal kicks in (repeated misses, or a
    proactive-intervention trigger) — the WHY (hangs a piece, missed a check,
    etc.) is still explained either way; only the specific move is withheld.

    Only meaningful when played_uci is legal but != solution_uci — the caller
    (web/server.py) invokes this on a scored wrong attempt, never to decide
    legality.
    """
    try:
        played = chess.Move.from_uci(played_uci)
    except (ValueError, IndexError):
        return "That's not the strongest move here — try again."

    solution = None
    if solution_uci:
        try:
            solution = chess.Move.from_uci(solution_uci)
        except (ValueError, IndexError):
            solution = None
    # move_ref is what the text actually NAMES the solution with — the real
    # notation when reveal_solution is True, an opaque placeholder otherwise.
    # Redacting at the point of generation (rather than string-replacing
    # afterward) is what lets the "bigger capture" branch below also drop the
    # destination square it separately names — a plain substring replace of
    # the SAN alone would still leak that square.
    solution_san = _san_safe(board, solution_uci) if solution else None
    move_ref = solution_san if reveal_solution else "the solution move"

    if solution is not None:
        mate_trial = board.copy()
        mate_trial.push(solution)
        if mate_trial.is_checkmate():
            if tier == "high":
                return (
                    f"Big miss! {move_ref} would have delivered checkmate right here — always scan for a "
                    "forced checkmate before anything else, especially once the enemy king is short on escape squares."
                )
            if tier == "low":
                return f"Missed mate — {move_ref} was mate."
            return f"You missed a forced mate — {move_ref} would have delivered checkmate."

    hang_square = _hangs_piece(board, played)
    if hang_square is not None:
        # Describes the PLAYED move's own flaw, never the solution — nothing
        # here needs redacting regardless of reveal_solution.
        after = board.copy()
        after.push(played)
        hung_piece = after.piece_at(hang_square)
        name = PIECE_NAMES[hung_piece.piece_type] if hung_piece else "piece"
        square = chess.square_name(hang_square)
        if tier == "high":
            return (
                f"That move leaves your {name} on {square} completely undefended, so the opponent can just "
                f"capture it for free next turn — that's called 'hanging' a piece. Before you move, always ask: "
                "can anything take my piece afterward, and do I have something guarding that square?"
            )
        if tier == "low":
            return f"Hangs the {name} on {square}."
        return f"Bad move — that hangs your {name} on {square} for free."

    if solution is not None:
        played_value = _captured_value(board, played)
        solution_value = _captured_value(board, solution)
        if solution_value > played_value:
            target = board.piece_at(solution.to_square)
            target_name = PIECE_NAMES[target.piece_type] if target else "piece"
            # The destination square is itself part of "the correct answer" —
            # named only when reveal_solution allows it.
            target_square = chess.square_name(solution.to_square) if reveal_solution else None
            where = f" on {target_square}" if target_square else ""
            if tier == "high":
                return (
                    f"There was a bigger prize available: {move_ref} would have won the {target_name}{where}, "
                    "which is worth more than what you captured. When you have a choice of captures, always "
                    "compare what each one actually wins before picking one."
                )
            if tier == "low":
                return f"{move_ref} wins more material ({target_name})."
            return f"There's a better move — {move_ref} wins the {target_name}{where}."

        played_trial = board.copy()
        played_trial.push(played)
        solution_trial = board.copy()
        solution_trial.push(solution)
        if solution_trial.is_check() and not played_trial.is_check():
            if tier == "high":
                return (
                    f"{move_ref} would have given check, forcing your opponent to respond immediately and "
                    "keeping them on the back foot — checks are a great way to gain time and disrupt their plans."
                )
            if tier == "low":
                return f"{move_ref} keeps the pressure with check."
            return f"There's a better move — {move_ref} gives a check that keeps up the pressure."

    if solution_san:
        if tier == "high":
            return (
                f"This move is legal, but {move_ref} was the stronger choice here. Try comparing a few "
                "candidate moves before playing — look at checks, captures, and threats first, in that order."
            )
        if tier == "low":
            return f"{move_ref} was stronger."
        return f"That's legal, but there's a better move — {move_ref} was stronger here."

    if tier == "high":
        return (
            "That's a legal move, but it's not the best one here — take a moment to look for checks, captures, "
            "and threats before deciding."
        )
    if tier == "low":
        return "Not the best move here."
    return "That's legal, but there's a better move — try again."
