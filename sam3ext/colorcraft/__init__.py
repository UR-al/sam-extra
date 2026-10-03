"""Colorcraft for sam-extra — latent colour grading during sampling.

A port of muerrilla/ComfyUI-Colorcraft (MIT, Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla)) at
commit d28ac6a4e997d0f8a2f1a60b7361b561c4a15bbf (the latest math), on the Forge Neo structure of the
fork aoleg/ComfyUI-Colorcraft (MIT, Oleg Afonin) at f00066c63c9d8f96abc119cada3b51b36688fcac, whose
Flux2 basis vectors and calibration it also carries. Each module's header names its origin lines and
the changes made here.

* ``schedule``  per-step strength curve and its sigma lookup (upstream, unchanged)
* ``vectors``   basis-vector edits (upstream, unchanged)
* ``masking``   masks read off the latent (upstream, unchanged)
* ``color``     contrast / colour shift and their anchors (upstream; ``to_model_space`` as in the fork)
* ``basis``     families krea2 / zimage (upstream) and flux2 (fork), calibration, vector loading
* ``engine``    the per-evaluation modifier chain (upstream's post-CFG body)
* ``spec``      script arguments, chain building, infotext (ours, upstream's and the fork's)
* ``panel_state`` the panel's script arguments: hidden state, editor overlay, paste, labels
* ``hook``      the Forge post-CFG hook on a cloned UNet
* ``debug``     axis-projection / mask-preview rendering (upstream, unchanged)
* ``debug_panel`` upstream's Forge Debug panel: capture during sampling, render on demand
* ``ui``        the Gradio panel
* ``data/``     the basis vectors: colorcraft-krea2 / zimage (upstream), colorcraft-flux2 (fork)

Nothing here imports Forge at module level; ``scripts/colorcraft.py`` is the WebUI entry point.
"""
