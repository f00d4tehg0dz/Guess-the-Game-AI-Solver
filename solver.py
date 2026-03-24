#!/usr/bin/env python3
"""
GuessThe.Game Solver — The Cheat Engine

Strategy:
1. Finds today's puzzle number
2. Checks if the answer is already in the game_info API (it usually is!)
3. If not, downloads images for AI vision analysis
4. Submits the correct answer

The key discovery: the game_info API exposes answers for ALL puzzles,
including today's. No vision analysis actually needed for the cheat —
but we include it as a fallback and for fun.
"""

from typing import Optional, List
import json
import os
import sys
import requests

API_BASE = "https://api.guessthe.game/api"
IMAGES_BASE = "https://images.guessthe.game/gtg_images"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VERSION = "2.3.75"


def find_today_puzzle() -> int:
    """Binary search for the latest puzzle number."""
    lo, hi = 1, 2000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        resp = requests.head(
            f"{IMAGES_BASE}/{mid}/1.webp?v={VERSION}", timeout=10
        )
        if resp.status_code == 200:
            lo = mid
        else:
            hi = mid - 1
    return lo


def get_answer_from_api(puzzle_num: int) -> Optional[dict]:
    """Get answer directly from the game_info API (the exploit)."""
    resp = requests.get(
        f"{API_BASE}/game_info/",
        params={"puzzle_num": puzzle_num, "puzzle_type": "gtg"},
        timeout=10,
    )
    if resp.status_code == 200:
        data = resp.json()
        if data.get("status") == "ok":
            return data
    return None


def get_csrf_token() -> str:
    """Get a CSRF token for submitting guesses."""
    resp = requests.get(f"{API_BASE}/csrf/", timeout=10)
    return resp.json()["csrfToken"]


def submit_guess(puzzle_num: int, guess: str, guess_num: int = 1) -> dict:
    """Submit a guess to the API."""
    csrf = get_csrf_token()
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
    """Download puzzle images for vision analysis."""
    out_dir = os.path.join(DATA_DIR, "images", str(puzzle_num))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(1, 6):
        path = os.path.join(out_dir, f"{i}.webp")
        if not os.path.exists(path):
            resp = requests.get(
                f"{IMAGES_BASE}/{puzzle_num}/{i}.webp?v={VERSION}", timeout=15
            )
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
        if os.path.exists(path):
            paths.append(path)
    return paths


def solve(puzzle_num: Optional[int] = None, auto_submit: bool = False):
    """Main solver flow."""
    if puzzle_num is None:
        print("Finding today's puzzle...")
        puzzle_num = find_today_puzzle()
    print(f"Puzzle #{puzzle_num}")

    # Step 1: Try the direct API exploit
    print("Checking game_info API...")
    answer_data = get_answer_from_api(puzzle_num)
    if answer_data:
        answer = answer_data["answer"]
        year = answer_data.get("release_year", "?")
        dev = answer_data.get("developer", "?")
        print(f"\n{'='*50}")
        print(f"  ANSWER: {answer}")
        print(f"  Year:   {year}")
        print(f"  Dev:    {dev}")
        print(f"{'='*50}\n")

        if auto_submit:
            print("Submitting guess...")
            result = submit_guess(puzzle_num, answer)
            print(f"Result: {result}")

        return answer

    # Step 2: Fallback — download images for manual/AI analysis
    print("API didn't have the answer — downloading images for analysis...")
    paths = download_images(puzzle_num)
    print(f"Downloaded {len(paths)} images to: {os.path.dirname(paths[0])}")
    print("Use Claude vision or another AI to analyze these screenshots.")
    return None


if __name__ == "__main__":
    pnum = int(sys.argv[1]) if len(sys.argv) > 1 else None
    auto = "--submit" in sys.argv
    solve(pnum, auto)