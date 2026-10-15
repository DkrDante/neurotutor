"""Grow chess_task/puzzle_data/sample_puzzles.csv from the open Lichess puzzle database (CC0).

    curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst
    zstd -dc lichess_db_puzzle.csv.zst | python -m data_gen.build_puzzle_bank --per-bucket 120

Lichess rows are `PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,...`, where FEN is the
position *before* the opponent's move and Moves[0] is that move. We play Moves[0] so the learner sees the
position they actually have to solve, and keep the first solver move as the answer (the same single-move
format as the rest of the bank). Only well-tested puzzles are kept (high popularity, many plays, low rating
deviation) and they are sampled evenly per 100-point rating bucket, so every difficulty band has plenty of
distinct puzzles and a long session never has to repeat one.
"""
from __future__ import annotations
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
import chess

BANK = Path(__file__).resolve().parent.parent / "chess_task" / "puzzle_data" / "sample_puzzles.csv"


def read_candidates(stream, min_popularity: int, min_plays: int, max_deviation: int, lo: int, hi: int):
    reader = csv.reader(stream)
    next(reader, None)   # header
    for row in reader:
        try:
            pid, fen, moves, rating, dev, pop, plays = row[0], row[1], row[2].split(), int(row[3]), int(row[4]), int(row[5]), int(row[6])
        except (ValueError, IndexError):
            continue
        if not (lo <= rating <= hi) or pop < min_popularity or plays < min_plays or dev > max_deviation or len(moves) < 2:
            continue
        yield pid, fen, moves, rating


def to_puzzle(pid: str, fen: str, moves: list[str], rating: int):
    """Position after the opponent's first move, plus the solver's first move; None if anything is illegal."""
    try:
        board = chess.Board(fen)
        board.push_uci(moves[0])
        answer = chess.Move.from_uci(moves[1])
        if answer not in board.legal_moves:
            return None
        return {"puzzle_id": pid, "fen": board.fen(), "solution_move": moves[1], "rating": rating}
    except (ValueError, AssertionError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-bucket", type=int, default=120, help="puzzles to add per 100-point rating bucket")
    p.add_argument("--min-rating", type=int, default=600)
    p.add_argument("--max-rating", type=int, default=2400)
    p.add_argument("--min-popularity", type=int, default=90)
    p.add_argument("--min-plays", type=int, default=1000)
    p.add_argument("--max-deviation", type=int, default=90)
    p.add_argument("--seed", type=int, default=14)
    p.add_argument("--bank", type=Path, default=BANK)
    args = p.parse_args()

    with open(args.bank, newline="") as f:
        existing = list(csv.DictReader(f))
    have = {r["puzzle_id"] for r in existing}

    # Reservoir-sample per bucket while streaming, so memory stays small.
    rng = random.Random(args.seed)
    cap = args.per_bucket * 3          # oversample: some will fail the legality check
    pools: dict[int, list] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    for pid, fen, moves, rating in read_candidates(
            sys.stdin, args.min_popularity, args.min_plays, args.max_deviation, args.min_rating, args.max_rating):
        if pid in have:
            continue
        b = rating // 100
        seen[b] += 1
        if len(pools[b]) < cap:
            pools[b].append((pid, fen, moves, rating))
        else:
            j = rng.randrange(seen[b])
            if j < cap:
                pools[b][j] = (pid, fen, moves, rating)

    added = []
    for b in sorted(pools):
        good = [q for q in (to_puzzle(*c) for c in pools[b]) if q]
        rng.shuffle(good)
        added.extend(good[:args.per_bucket])
        print(f"{b * 100}-{b * 100 + 99}: {min(len(good), args.per_bucket)} added (of {seen[b]} eligible)", file=sys.stderr)

    with open(args.bank, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["puzzle_id", "fen", "solution_move", "rating"])
        w.writeheader()
        w.writerows(existing)
        w.writerows(sorted(added, key=lambda r: r["rating"]))
    print(f"bank now has {len(existing) + len(added)} puzzles ({len(added)} new)", file=sys.stderr)


if __name__ == "__main__":
    main()
