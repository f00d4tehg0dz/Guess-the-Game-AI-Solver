#!/usr/bin/env python3
"""Download ALL puzzle images (1 through latest) using parallel threads."""

import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGES_BASE = "https://images.guessthe.game/gtg_images"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "images")
VERSION = "2.3.75"
MAX_WORKERS = 20


def download_file(url, path):
    """Download a single file if it doesn't already exist."""
    if os.path.exists(path):
        return None  # already have it
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
    except Exception:
        pass
    return None


def build_download_tasks(start, end):
    """Build list of (url, path) tuples for all images."""
    tasks = []
    for num in range(start, end + 1):
        puzzle_dir = os.path.join(DATA_DIR, str(num))
        for i in range(1, 6):
            url = f"{IMAGES_BASE}/{num}/{i}.webp?v={VERSION}"
            path = os.path.join(puzzle_dir, f"{i}.webp")
            tasks.append((url, path))
        # video
        url = f"{IMAGES_BASE}/{num}/video/6.webm?v={VERSION}"
        path = os.path.join(puzzle_dir, "6.webm")
        tasks.append((url, path))
    return tasks


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 1420

    tasks = build_download_tasks(start, end)

    # Filter out already downloaded
    pending = [(url, path) for url, path in tasks if not os.path.exists(path)]
    already = len(tasks) - len(pending)

    print(f"Total files: {len(tasks)}")
    print(f"Already downloaded: {already}")
    print(f"To download: {len(pending)}")
    print(f"Using {MAX_WORKERS} parallel threads\n")

    if not pending:
        print("All images already downloaded!")
        return

    downloaded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_file, url, path): (url, path) for url, path in pending}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                downloaded += 1
            else:
                failed += 1
            if i % 200 == 0:
                print(f"  Progress: {i}/{len(pending)} (downloaded: {downloaded}, skipped/failed: {failed})")

    print(f"\nDone! Downloaded: {downloaded}, Skipped/Failed: {failed}")
    print(f"Images saved to: {DATA_DIR}/")


if __name__ == "__main__":
    main()