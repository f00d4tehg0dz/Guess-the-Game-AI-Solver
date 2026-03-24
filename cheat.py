#!/usr/bin/env python3
"""
GuessThe.Game Ultimate Cheat Tool

Usage:
  python3 cheat.py              # Solve today's puzzle (show answer)
  python3 cheat.py --submit     # Solve and auto-submit on guess 1
  python3 cheat.py --images     # Download today's images for viewing
  python3 cheat.py --puzzle 1409  # Solve a specific puzzle
  python3 cheat.py --scrape     # Build/update the full answer database

The Exploit:
  The game_info API at https://api.guessthe.game/api/game_info/
  accepts any puzzle_num and returns the answer, developer, and year.
  No auth required. This works for past AND current puzzles.
"""

from typing import Optional, List
import argparse
import json
import os
import sys
import time
import requests

API_BASE = "https://api.guessthe.game/api"
IMAGES_BASE = "https://images.guessthe.game/gtg_images"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANSWERS_FILE = os.path.join(DATA_DIR, "answers.json")
VERSION = "2.3.75"

# Suppress SSL warnings for older Python
import warnings
warnings.filterwarnings("ignore")


def find_today_puzzle() -> int:
    """Binary search for the latest puzzle number with images."""
    lo, hi = 1, 2000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            resp = requests.head(
                f"{IMAGES_BASE}/{mid}/1.webp?v={VERSION}", timeout=10
            )
            if resp.status_code == 200:
                lo = mid
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
    return lo


def get_answer(puzzle_num: int) -> Optional[dict]:
    """Get answer from game_info API."""
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
        print(f"Error: {e}")
    return None


def submit_guess(puzzle_num: int, guess: str, guess_num: int = 1) -> dict:
    """Submit a guess via the API."""
    csrf_resp = requests.get(f"{API_BASE}/csrf/", timeout=10)
    csrf = csrf_resp.json()["csrfToken"]
    resp = requests.post(
        f"{API_BASE}/submit_guess/",
        json={
            "data": {
                "puzzle_num": puzzle_num,
                "guess_num": guess_num,
                "guess": guess,
                "puzzle_type": "gtg",
                "elapsed_time": 0,
            }
        },
        headers={"X-CSRFToken": csrf},
        timeout=10,
    )
    return resp.json()


def download_images(puzzle_num: int) -> List[str]:
    """Download all images for a puzzle."""
    out_dir = os.path.join(DATA_DIR, "images", str(puzzle_num))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(1, 6):
        path = os.path.join(out_dir, f"{i}.webp")
        if not os.path.exists(path):
            try:
                resp = requests.get(f"{IMAGES_BASE}/{puzzle_num}/{i}.webp?v={VERSION}", timeout=15)
                if resp.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(resp.content)
            except Exception:
                pass
        if os.path.exists(path):
            paths.append(path)

    # Try video
    video_path = os.path.join(out_dir, "6.webm")
    if not os.path.exists(video_path):
        try:
            resp = requests.get(f"{IMAGES_BASE}/{puzzle_num}/video/6.webm?v={VERSION}", timeout=15)
            if resp.status_code == 200:
                with open(video_path, "wb") as f:
                    f.write(resp.content)
                paths.append(video_path)
        except Exception:
            pass
    elif os.path.exists(video_path):
        paths.append(video_path)
    return paths


def scrape_all_answers():
    """Build the full answer database."""
    os.makedirs(DATA_DIR, exist_ok=True)
    answers = {}
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE) as f:
            answers = {a["puzzle_num"]: a for a in json.load(f)}
        print(f"Loaded {len(answers)} existing answers")

    latest = find_today_puzzle()
    print(f"Latest puzzle: #{latest}")

    to_fetch = [n for n in range(1, latest + 1) if n not in answers]
    if not to_fetch:
        print("Database is up to date!")
        return

    print(f"Fetching {len(to_fetch)} new answers...")
    for i, num in enumerate(to_fetch):
        result = get_answer(num)
        if result:
            answers[num] = result
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(to_fetch)}")
            _save_answers(answers)
        time.sleep(0.05)

    _save_answers(answers)
    print(f"Done! Total: {len(answers)} answers saved to {ANSWERS_FILE}")


def _save_answers(answers):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ANSWERS_FILE, "w") as f:
        json.dump(
            sorted(answers.values(), key=lambda x: x["puzzle_num"]),
            f, indent=2, ensure_ascii=False,
        )


def solve(puzzle_num: Optional[int] = None, do_submit: bool = False, do_images: bool = False):
    """Main solve flow."""
    if puzzle_num is None:
        print("Finding today's puzzle...")
        puzzle_num = find_today_puzzle()

    print(f"\n  Puzzle #{puzzle_num}")
    print(f"  {'='*40}")

    # Get the answer
    data = get_answer(puzzle_num)
    if data:
        print(f"  ANSWER:    {data['answer']}")
        print(f"  Year:      {data['release_year']}")
        print(f"  Developer: {data['developer']}")
        print(f"  {'='*40}\n")

        if do_images:
            print("Downloading images...")
            paths = download_images(puzzle_num)
            print(f"  Saved {len(paths)} files to data/images/{puzzle_num}/")
            for p in paths:
                print(f"    {os.path.basename(p)}")

        if do_submit:
            print("Submitting answer...")
            result = submit_guess(puzzle_num, data["answer"])
            print(f"  Result: {result}")

        # Image URLs for easy viewing
        print(f"\n  View images:")
        for i in range(1, 6):
            print(f"    {IMAGES_BASE}/{puzzle_num}/{i}.webp?v={VERSION}")
        print(f"    {IMAGES_BASE}/{puzzle_num}/video/6.webm?v={VERSION}")

        return data["answer"]
    else:
        print("  Could not fetch answer from API!")
        return None


def main():
    parser = argparse.ArgumentParser(description="GuessThe.Game Cheat Tool")
    parser.add_argument("--puzzle", "-p", type=int, help="Puzzle number (default: today)")
    parser.add_argument("--submit", "-s", action="store_true", help="Auto-submit the answer")
    parser.add_argument("--images", "-i", action="store_true", help="Download puzzle images")
    parser.add_argument("--scrape", action="store_true", help="Build/update full answer database")
    args = parser.parse_args()

    if args.scrape:
        scrape_all_answers()
    else:
        solve(args.puzzle, args.submit, args.images)


if __name__ == "__main__":
    main()