"""SPEED — Spectral Progressive Diffusion (arXiv 2605.18736) for Forge, as a sampler wrapper.

The early, noise-dominated steps run on a DCT-truncated coarse latent grid; at a transition sigma
the coarse DCT block is embedded in the full grid, the new high frequencies are filled with
sigma-scaled noise, the state is rescaled by kappa = r / (1 + (r - 1) * sigma) and sampling continues
at full size.

Modules (no Forge import at module level except ``forge_host``'s lazy ones):

* ``spectral``   — pure-torch DCT (cached orthonormal basis on the device), FFT and Haar expansion,
  kappa / timestep alignment;
* ``schedule``   — power-spectrum presets, sigma* derivation (delta tolerance, adaptive delta,
  neo_shift divisor), manual thresholds and the two transition plans ("transition" = official /
  aoleg, "respace" = sorryhyun);
* ``runner``     — the segmented sampler run (Forge-agnostic, driven through a small host interface);
* ``forge_host`` — the Forge glue: guards, per-seed noise, Brownian noise samplers, the
  ``sampling_sigmas`` hand-off and the ``p.sampler.func`` wrapper.

Upstream sources (all MIT): howardhx/speed@ca7801c9 (Copyright (c) 2026 Howard Xiao),
aoleg/ComfyUI-SPEED@a8873591 (Copyright (c) 2026 A. Izzuddin Al Faruq; Forge/SwarmUI/preset work
by Oleg Afonin), sorryhyun/ComfyUI-Spectrum-KSampler@b46a364a (Copyright (c) 2026 sorryhyun).
Each derived module carries the notices and its list of changes.
"""
