from __future__ import annotations

import argparse


def preload(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--sam3-no-huggingface",
        action="store_true",
        help="Don't use SAM3 checkpoints from Hugging Face; require local checkpoints instead.",
    )
