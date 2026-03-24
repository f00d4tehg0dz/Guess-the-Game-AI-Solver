#!/usr/bin/env python3
"""Scrape all puzzle answers from the GuessThe.Game API."""

from typing import Optional
import json
import os
import sys
import time
import requests

API_BASE = "https://api.guessthe.game/api"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ANSWERS_FILE = os.path.join(DATA_DIR, "answers.json")


def fetch_answer(puzzle_num: int) -> Optional[dict]:
    """Fetch the answer for a given puzzle number."""
    try:
        resp = requests.get(
            f"{API_BASE}/game_info/",
            params={"puzzle_num": puzzle_num, "puzzle_type": "gtg"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "ok":
                return {
                    "puzzle_num": puzzle_num,
                    "answer": data["answer"],
                    "release_year": data.get("release_year", ""),
                    "developer": data.get("developer", ""),
                }
    except Exception as e:
        print(f"  Error fetching puzzle {puzzle_num}: {e}")
    return None


def find_latest_puzzle() -> int:
    """Binary search for the latest puzzle number with images available."""
    lo, hi = 1, 2000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        resp = requests.head(
            f"https://images.guessthe.game/gtg_images/{mid}/1.webp?v=2.3.75",
            timeout=10,
        )
        if resp.status_code == 200:
            lo = mid
        else:
            hi = mid - 1
    return lo


def scrape_all_answers(start: int = 1, end: Optional[int] = None):
    """Scrape answers for all puzzles from start to end."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load existing answers
    answers = {}
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE) as f:
            existing = json.load(f)
            answers = {a["puzzle_num"]: a for a in existing}
        print(f"Loaded {len(answers)} existing answers")

    if end is None:
        print("Finding latest puzzle number...")
        end = find_latest_puzzle()
        print(f"Latest puzzle: {end}")

    # Find puzzles we haven't scraped yet
    to_fetch = [n for n in range(start, end + 1) if n not in answers]
    print(f"Fetching {len(to_fetch)} new answers (puzzles {start}-{end})...")

    for i, num in enumerate(to_fetch):
        result = fetch_answer(num)
        if result:
            answers[num] = result
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(to_fetch)}")
            # Save periodically
            _save(answers)
        time.sleep(0.05)  # Be gentle on the API

    _save(answers)
    print(f"Done! Total answers: {len(answers)}")
    return answers


def _save(answers: dict):
    sorted_answers = sorted(answers.values(), key=lambda x: x["puzzle_num"])
    with open(ANSWERS_FILE, "w") as f:
        json.dump(sorted_answers, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    scrape_all_answers(start, end)