"""Download the GeoRC dataset from Hugging Face at a pinned revision.

Reads the token from HF_TOKEN if set (raises rate limits; not needed for this ungated dataset).
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "mohit-talreja/GeoRC"
REVISION = "c37c87ead9672acecf293fee8648f970bf8b0b94"  # main as of 2026-11-08
TEXT_PATTERNS = ["*/*.txt", "*/*.json", "LICENSE.txt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true", help="chains and metadata, no images")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    path = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.out,
        allow_patterns=TEXT_PATTERNS if args.text_only else None,
    )
    print(path)


if __name__ == "__main__":
    main()
