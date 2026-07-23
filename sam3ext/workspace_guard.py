from __future__ import annotations

import os
import sys
from functools import wraps
from typing import Any


def _normalized_path(path: Any) -> str:
    return os.path.normcase(str(path or "")).replace("\\", "/")


def _module_path(module: Any) -> str:
    return _normalized_path(getattr(module, "__file__", ""))


def _loaded_script_modules() -> tuple[Any, ...]:
    """Return Forge dynamically-loaded scripts plus normal Python modules."""

    modules: list[Any] = []
    seen: set[int] = set()

    try:
        from modules import script_loading

        forge_scripts = getattr(script_loading, "loaded_scripts", {})
        if isinstance(forge_scripts, dict):
            for module in forge_scripts.values():
                if module is not None and id(module) not in seen:
                    seen.add(id(module))
                    modules.append(module)
    except Exception:
        pass

    for module in tuple(sys.modules.values()):
        if module is not None and id(module) not in seen:
            seen.add(id(module))
            modules.append(module)

    return tuple(modules)


def _belongs_to_demo(demo: Any, component: Any) -> bool:
    blocks = getattr(demo, "blocks", None)
    component_id = getattr(component, "_id", None)
    if not isinstance(blocks, dict) or component_id is None:
        return False
    return blocks.get(component_id) is component


def _prune_registry_lists(
    demo: Any | None,
    rk_pairs: Any = None,
    tde_sliders: Any = None,
) -> dict[str, int]:
    removed = {"rk": 0, "tde": 0}

    if isinstance(rk_pairs, list):
        kept_pairs = [
            pair
            for pair in rk_pairs
            if isinstance(pair, (tuple, list))
            and len(pair) >= 2
            and _belongs_to_demo(demo, pair[0])
            and _belongs_to_demo(demo, pair[1])
        ]
        removed["rk"] = len(rk_pairs) - len(kept_pairs)
        rk_pairs[:] = kept_pairs

    if isinstance(tde_sliders, list):
        kept_sliders = [
            component
            for component in tde_sliders
            if _belongs_to_demo(demo, component)
        ]
        removed["tde"] = len(tde_sliders) - len(kept_sliders)
        tde_sliders[:] = kept_sliders

    return removed


def prune_sampler_callback_targets(demo: Any, callback: Any) -> dict[str, int]:
    """Prune registries retained directly by a dynamically-loaded callback."""

    callback_globals = getattr(callback, "__globals__", None)
    if not isinstance(callback_globals, dict):
        return {"rk": 0, "tde": 0}
    return _prune_registry_lists(
        demo,
        callback_globals.get("_rk_method_dropdowns"),
        callback_globals.get("_tde_max_steps_sliders"),
    )


def prune_stale_sampler_load_targets(demo: Any | None) -> dict[str, int]:
    """Remove throwaway RK/TDE components before their ``demo.load`` hooks.

    With ``--api``, Forge builds default script arguments in temporary
    ``gr.Blocks`` contexts before ``app_started``. RK Sampler and TDE Sampler
    append those throwaway components to module-level lists alongside the real
    UI components. Forge loads extension scripts through
    ``script_loading.loaded_scripts`` rather than registering them in
    ``sys.modules``, so both registries are inspected here. Passing a demo
    retains only components belonging to its live ``gr.Blocks`` instance.
    """

    removed = {"rk": 0, "tde": 0}
    seen_lists: set[int] = set()

    for module in _loaded_script_modules():
        path = _module_path(module)

        rk_pairs = getattr(module, "_rk_method_dropdowns", None)
        if (
            isinstance(rk_pairs, list)
            and id(rk_pairs) not in seen_lists
            and (path.endswith("/rk_sampler.py") or "sd-webui-rk-sampler" in path)
        ):
            seen_lists.add(id(rk_pairs))
            result = _prune_registry_lists(demo, rk_pairs=rk_pairs)
            removed["rk"] += result["rk"]

        tde_sliders = getattr(module, "_tde_max_steps_sliders", None)
        if (
            isinstance(tde_sliders, list)
            and id(tde_sliders) not in seen_lists
            and (path.endswith("/tde_sampler.py") or "sd-webui-tde-sampler" in path)
        ):
            seen_lists.add(id(tde_sliders))
            result = _prune_registry_lists(demo, tde_sliders=tde_sliders)
            removed["tde"] += result["tde"]

    return removed


def guard_sampler_app_started_callbacks(callback_map: Any) -> int:
    """Wrap RK/TDE app-start callbacks so filtering cannot lose an order race.

    Metadata ordering is advisory and can be overridden by callback reload
    state. Wrapping the sampler callbacks themselves guarantees their
    module-level component lists are filtered immediately before they register
    ``demo.load`` outputs, regardless of the surrounding callback order.
    """

    if not isinstance(callback_map, dict):
        return 0

    installed = 0
    callbacks = callback_map.get("callbacks_app_started", ())
    for registered in tuple(callbacks):
        script_path = _normalized_path(getattr(registered, "script", ""))
        is_sampler = (
            script_path.endswith("/rk_sampler.py")
            or "sd-webui-rk-sampler" in script_path
            or script_path.endswith("/tde_sampler.py")
            or "sd-webui-tde-sampler" in script_path
        )
        callback = getattr(registered, "callback", None)
        if (
            not is_sampler
            or not callable(callback)
            or getattr(callback, "_sam3_workspace_guarded", False)
        ):
            continue

        @wraps(callback)
        def guarded(demo: Any, app: Any, _callback=callback):
            prune_sampler_callback_targets(demo, _callback)
            return _callback(demo, app)

        guarded._sam3_workspace_guarded = True
        registered.callback = guarded
        installed += 1

    return installed
