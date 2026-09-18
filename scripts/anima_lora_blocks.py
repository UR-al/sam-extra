"""Install sam-extra's ANIMA 28/40/52-block LoRA compatibility adapter."""

from __future__ import annotations

import sys
import traceback

from modules import script_callbacks

from sam3ext.anima_lora_blocks import (
    install_forge_lora_block_hook,
    uninstall_forge_lora_block_hook,
)


def _install_anima_lora_block_hook() -> None:
    try:
        if install_forge_lora_block_hook():
            print(
                "[sam-extra] ANIMA LoRA block compatibility installed "
                "(28/40/52 blocks)."
            )
    except Exception:
        print(
            "[-] sam-extra: failed to install ANIMA LoRA block compatibility:\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )


# Built-in extensions normally load first, so install immediately.  on_before_ui
# is a fallback for unusual extension orders and remains idempotent.
_install_anima_lora_block_hook()
script_callbacks.on_before_ui(_install_anima_lora_block_hook)

try:
    script_callbacks.on_script_unloaded(uninstall_forge_lora_block_hook)
except AttributeError:
    pass
