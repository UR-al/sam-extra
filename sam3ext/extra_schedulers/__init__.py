"""Extra Schedulers — more entries for Forge's "Schedule type" list.

Cosine, CosineExponential blend, Phi, Laplace, Karras Dynamic, a safe ``custom`` scheduler
(an arithmetic expression or a sigma list), React Cosinusoidal DynSF, Flow Cosmos rho7 and Flow Cosmos Dynamic.
This is a clean
reimplementation: the first six labels are the ones specified for infotext compatibility with the
unlicensed aoleg/Neo_ExtraSchedulers (a fork of the equally unlicensed DenOfEquity/webUI_ExtraSchedulers),
but no code of those repositories was read or used — only their READMEs, for the user-facing names.
Laplace is ComfyUI's ``get_sigmas_laplace`` (GPL-3.0); React Cosinusoidal DynSF is reForge's label and
published formula (reForge is AGPL-3.0; its code was not copied); Flow Cosmos rho7 has the name and the
idea of KeithZ117/Comfyui-anima-sampler (MIT) and is written from the Karras/EDM formula (none of its code
copied); Flow Cosmos Dynamic is this extension's own composition of Karras Dynamic and Flow Cosmos rho7; the
other closed forms are public math (see ``schedulers.py``).

Modules
    ``expression``  — AST-whitelisted expression compiler/evaluator for the custom scheduler.
    ``sigma_list``  — strict sigma-list parser and Forge's log-linear interpolation.
    ``schedulers``  — the scheduler functions (Forge's scheduler signature).
    ``settings``    — the accordion's per-generation values and the state the schedulers read.
    ``registry``    — registration in Forge 2.29.2's ``modules.sd_schedulers``.

The Forge hooks (accordion, infotext, XYZ axes) are in ``scripts/anima_extra_schedulers.py`` and
``sam3ext/ui_extra_schedulers.py``.
"""
