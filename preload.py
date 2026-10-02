from __future__ import annotations

import argparse


def preload(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--sam3-no-huggingface",
        action="store_true",
        help="Don't use SAM3 checkpoints from Hugging Face; require local checkpoints instead "
        "(without it, a missing default sam3.pt is downloaded from facebook/sam3).",
    )
    parser.add_argument(
        "--sam3-no-auto-install",
        action="store_true",
        help="forge_sam3_extension install.py: only report missing requirements.txt packages instead of "
        "installing them (same as SAM3_NO_AUTO_INSTALL=1). Put it in COMMANDLINE_ARGS.",
    )
