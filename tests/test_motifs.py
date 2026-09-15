from chess_task.motifs import MOTIFS, classify_mate_pattern, classify_puzzle
from chess_task.puzzles import DEFAULT_PUZZLE_CSV, PuzzleTaskEngine

def test_back_rank_mate_detected():
    # Rook delivers mate along the back rank; the king's own pawns block
    # every escape square in front of it — the textbook pattern.
    assert classify_mate_pattern("2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", "a1a8") == "back_rank"

def test_queen_mate_detected_when_not_back_rank():
    assert classify_mate_pattern("7k/8/6QK/8/8/8/8/8 w - - 0 1", "g6g7") == "queen_mate"

def test_non_mating_move_returns_other():
    # a1a2 doesn't deliver checkmate at all.
    assert classify_mate_pattern("2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", "a1a2") == "other"

def test_classify_puzzle_replays_earlier_moves_before_classifying_the_mate():
    # solver move, scripted opponent reply, solver move that mates — only
    # the FINAL move should be classified, using the position it's actually
    # played from (verified separately to genuinely be checkmate).
    result = classify_puzzle("2k5/1ppp3p/8/8/8/8/8/R6K w - - 0 1", ["h1h2", "h7h6", "a1a8"])
    assert result == "back_rank"

def test_every_real_puzzle_classifies_without_raising():
    # The strongest check: run the real classifier over the entire real
    # curated set, not a handful of hand-picked examples.
    puzzles = PuzzleTaskEngine._load_puzzles(DEFAULT_PUZZLE_CSV)
    seen_motifs = set()
    for p in puzzles:
        motif = classify_puzzle(p.fen, p.solution_moves)
        assert motif in MOTIFS
        seen_motifs.add(motif)
    # A real, curated set should exercise more than one or two categories.
    assert len(seen_motifs) >= 5
