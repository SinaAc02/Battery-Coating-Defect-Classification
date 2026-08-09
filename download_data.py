from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import requests


FIGSHARE_URL = "https://ndownloader.figshare.com/files/55182476"
EXPECTED_MD5 = "7cbbf14725f9762a64a81a12b858a2b4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the CoatingVision dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "CoatingVision.zip"
    destination = args.output / "CoatingVision"

    if not archive.exists():
        print("Downloading CoatingVision from the authors' official Figshare record...")
        with requests.get(FIGSHARE_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)

    digest = hashlib.md5(archive.read_bytes()).hexdigest()  # Dataset-published checksum.
    if digest != EXPECTED_MD5:
        raise RuntimeError(f"Checksum mismatch: expected {EXPECTED_MD5}, got {digest}")

    if not destination.exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
    print(f"Dataset ready at {destination.resolve()}")
    print("Kaggle mirror: vigneshirtt/li-ion-battery-coating-defect-dataset")


if __name__ == "__main__":
    main()

