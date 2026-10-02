# Verbatim copy of upstream CNS Sampler Patch for the origin-parity tests (tests/test_cns_origin.py).
#
# origin: namemechan/comfyui-cns_sampler_patch@42278b138284f7a8685ef174af0a50fe03246dd0:cns_sampler_patch.py:1-623
# https://github.com/namemechan/comfyui-cns_sampler_patch — GPL-3.0 (upstream LICENSE), the same license
# as this extension. Only this header was added and CRLF line endings became LF; everything after
# the marker line is the upstream file unchanged (test_cns_origin.py pins its SHA-256, so an
# accidental edit fails). Test oracle only: the extension never imports it. It imports
# ``comfy.samplers`` at module level, so the test installs a stub first.
# ---- upstream cns_sampler_patch.py below (verbatim) ----
"""
CNS Sampler Patch — Colored Noise Sampling for ComfyUI
=======================================================

Patches any ancestral-type SAMPLER to replace uniform white noise injection
with frequency-aware colored noise via Haar wavelet subband energy weighting.

Paper: "Colored Noise Diffusion Sampling"
       Davidson et al., arXiv:2605.30332v1 (2026)

Wavelet utilities: adapted from dcw_node.py (Haar DWT, identical API)

─────────────────────────────────────────────────────────────────────────────
Architecture overview
─────────────────────────────────────────────────────────────────────────────

  SAMPLER → CNSSamplerPatch → SAMPLER (patched)

  The patch intercepts the sampler's noise_sampler callable and replaces it
  with a CNS colored-noise sampler.  A callback hook captures x_t each step
  so the wavelet-based γ proxy stays synchronized with the current state.

  CNS core (per step):
    1. Haar DWT on x_t  →  LL, LH, HL, HH subband energies
    2. γ proxy per band:  γ_f ≈ e_f / Σe  (energy fraction = "built-ness")
    3. deficit per band:  d_f = 1 − γ_f   (how unresolved that band still is)
    4. β_f  = d_f^gamma_power  (unnormalized noise scale weight)
    5. RMS-normalize β so that mean(β²) = 1
    6. Apply β to noise subbands, IDWT back to spatial domain
    7. Crop-energy correction: rescale std after [..,:H,:W] crop (odd-dim guard)
    8. Lerp with white noise by `strength` (0=off, 1=full CNS)

  Wavelet γ proxy vs. CNS original:
    CNS uses FFT + radial frequency bins + precomputed γ(f,t) ODE trajectory.
    This implementation approximates γ from the live x_t subband energies,
    requiring NO separate ODE prepass and no precomputed matrices.

    Early steps (x_t ≈ pure noise):
      All subband energies roughly equal → γ_f ≈ 0.25 for all f
      → all β ≈ 1 → white noise  (correct: CNS starts as white noise)

    Late steps (x_t ≈ clean image, natural 1/f² spectrum):
      LL energy >> HH energy
      → γ_LL high → d_LL low → β_LL low   (suppress resolved low-freq noise)
      → γ_HH low  → d_HH high → β_HH high (boost unresolved high-freq noise)

─────────────────────────────────────────────────────────────────────────────
Compatibility
─────────────────────────────────────────────────────────────────────────────
  Tested:  euler_ancestral, euler_ancestral_cfg_pp,
           dpm_2_ancestral, dpmpp_2s_ancestral, dpmpp_sde
  Deterministic ODEs (euler, dpm++2m, ddim …) do not inject noise →
    CNS patch is silently inert (noise_sampler never called).
"""

import inspect

import torch
import torch.nn.functional as F

import comfy.samplers


# ─────────────────────────────────────────────────────────────────────────────
# Debug Logging Switch
# ─────────────────────────────────────────────────────────────────────────────
# True로 바꾸면 매 스텝마다 [CNS DEBUG] x_t.shape / parity 정보를 콘솔에 출력합니다.
# (5D 텐서 발생 여부, H/W 홀짝 전환 시점 등을 진단할 때 사용)
# 평소에는 False로 두세요 — 매 스텝 print는 콘솔이 매우 시끄러워집니다.
DEBUG_LOGGING = False


# ─────────────────────────────────────────────────────────────────────────────
# Haar Wavelet Utilities  (API-identical to dcw_node.py)
# ─────────────────────────────────────────────────────────────────────────────

def _pad_even(x: torch.Tensor):
    """
    Pad H and W to even numbers (Haar DWT requires even spatial dims).
    Returns (padded_tensor, (original_H, original_W)).

    Works for any number of leading dimensions (4-D image, 5-D video, …).

    IMPORTANT: F.pad's reflect/replicate modes require the pad tuple length
    to match the input rank in a strict way:
        3-D / 4-D input → pad length 4  (pads last 2 dims)
        4-D / 5-D input → pad length 6  (pads last 3 dims)
    A 5-D video latent (B, C, T, H, W) therefore needs a 6-length pad tuple
    (0, pw, 0, ph, 0, 0) — the trailing (0, 0) leaves T untouched — or
    PyTorch raises "Padding size N is not supported for 5D input tensor".
    We build the pad tuple dynamically so both 4-D and 5-D inputs work.
    """
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = H % 2, W % 2
    if ph or pw:
        ndim = x.dim()
        if ndim >= 5:
            # Pad only the last 2 dims (H, W); leave dim -3 (e.g. T) alone.
            pad = (0, pw, 0, ph, 0, 0)
        else:
            pad = (0, pw, 0, ph)

        mode = "reflect" if (H >= 2 and W >= 2) else "replicate"
        try:
            x = F.pad(x, pad, mode=mode)
        except RuntimeError:
            # Final safety net for any unsupported rank/size combination
            # (e.g. H or W == 1, or ndim outside the reflect/replicate range).
            x = F.pad(x, pad, mode="constant", value=0.0)
    return x, (H, W)


def haar_dwt2d(x: torch.Tensor):
    """
    2-D Haar Discrete Wavelet Transform.
    Input : (B, C, H, W)  — H and W must be even.
    Output: four subbands, each (B, C, H//2, W//2)
        LL – coarse structure / global energy
        LH – horizontal edges
        HL – vertical edges
        HH – diagonal / fine texture
    """
    L  = (x[..., 0::2, :] + x[..., 1::2, :]) * 0.5
    Hi = (x[..., 0::2, :] - x[..., 1::2, :]) * 0.5

    LL = (L[...,  0::2] + L[...,  1::2]) * 0.5
    LH = (L[...,  0::2] - L[...,  1::2]) * 0.5
    HL = (Hi[..., 0::2] + Hi[..., 1::2]) * 0.5
    HH = (Hi[..., 0::2] - Hi[..., 1::2]) * 0.5

    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """
    2-D Haar Inverse Discrete Wavelet Transform.
    Handles arbitrary leading dims (4-D image, 5-D video).
    """
    *leading, h, w = LL.shape
    dev, dt = LL.device, LL.dtype

    L  = torch.empty(*leading, h, w * 2, device=dev, dtype=dt)
    Hi = torch.empty(*leading, h, w * 2, device=dev, dtype=dt)
    L[...,  0::2] = LL + LH
    L[...,  1::2] = LL - LH
    Hi[..., 0::2] = HL + HH
    Hi[..., 1::2] = HL - HH

    out = torch.empty(*leading, h * 2, w * 2, device=dev, dtype=dt)
    out[..., 0::2, :] = L + Hi
    out[..., 1::2, :] = L - Hi

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CNS Core
# ─────────────────────────────────────────────────────────────────────────────

def _subband_energy(sub: torch.Tensor) -> torch.Tensor:
    """
    Mean squared energy of a wavelet subband, averaged over C×H×W per batch item.
    Returns shape (B, 1, 1, 1) — broadcastable with (B, C, H, W) subband tensors.
    """
    dims = tuple(range(1, sub.dim()))   # all dims after B
    return sub.float().pow(2).mean(dim=dims, keepdim=True).clamp(min=1e-8)


def color_noise_wavelet(
    noise:       torch.Tensor,
    x_t:         torch.Tensor,
    strength:    float = 1.0,
    gamma_power: float = 0.5,
    gamma_scale: float = 2.0,
) -> torch.Tensor:
    """
    Apply CNS wavelet coloring to a white noise tensor.

    Parameters
    ----------
    noise        White noise to color, shape (B, C, H, W).
    x_t          Current noisy latent at this sampling step, same shape.
    strength     Lerp weight: 0.0 = white noise unchanged, 1.0 = full CNS.
    gamma_power  Exponent applied to the deficit before computing β.
                 0.5 (√) matches CNS paper default.
    gamma_scale  Divides γ proxy values to soften the routing effect.
                 Equivalent to "γ Divider" in CNS paper (default 1.73–25.0).
    Returns
    -------
    Colored noise tensor with identical std to input (variance-preserving).
    """
    if strength == 0.0:
        return noise

    # ── DEBUG LOGGING ──────────────────────────────────────────────────────
    # 상단의 DEBUG_LOGGING = True 로 바꾸면 활성화됩니다 (기본값: False).
    # 매 스텝 x_t의 shape / 차원 / H,W 패리티를 출력하고, 직전 스텝과 모양이
    # 달라진 경우 "← 변경됨" 표시를 붙여 줍니다 (progressive upscale 전환
    # 시점이나 5D 텐서 발생 시점을 한눈에 찾을 수 있습니다).
    if DEBUG_LOGGING:
        shape_now = tuple(x_t.shape)
        prev_shape = getattr(color_noise_wavelet, "_prev_shape", None)
        step_i = getattr(color_noise_wavelet, "_step_i", 0)

        changed_mark = ""
        if prev_shape is not None and prev_shape != shape_now:
            changed_mark = "  ← 변경됨! (이전 단계와 shape 다름)"

        print(
            f"[CNS DEBUG] step={step_i:04d} | x_t.shape={shape_now} "
            f"(dim={x_t.dim()}) | H,W parity=({x_t.shape[-2] % 2},{x_t.shape[-1] % 2})"
            f"{changed_mark}"
        )

        color_noise_wavelet._prev_shape = shape_now
        color_noise_wavelet._step_i = step_i + 1

    orig_dtype = noise.dtype
    noise_f    = noise.float()
    x_f        = x_t.float()

    # ── Step 1: Haar DWT on x_t → subband energies ───────────────────────────
    x_p, (H, W) = _pad_even(x_f)
    LL_x, LH_x, HL_x, HH_x = haar_dwt2d(x_p)

    e_LL = _subband_energy(LL_x)   # (B,1,1,1)
    e_LH = _subband_energy(LH_x)
    e_HL = _subband_energy(HL_x)
    e_HH = _subband_energy(HH_x)
    e_sum = e_LL + e_LH + e_HL + e_HH  # total energy per batch item

    # ── Step 2: γ proxy — energy fraction = "how built is this band" ─────────
    # Divided by gamma_scale to prevent premature over-routing (CNS Appendix C.2)
    g_LL = (e_LL / e_sum / gamma_scale).clamp(0.0, 1.0)
    g_LH = (e_LH / e_sum / gamma_scale).clamp(0.0, 1.0)
    g_HL = (e_HL / e_sum / gamma_scale).clamp(0.0, 1.0)
    g_HH = (e_HH / e_sum / gamma_scale).clamp(0.0, 1.0)

    # ── Step 3: deficit = 1 − γ  ("how unresolved") ──────────────────────────
    d_LL = (1.0 - g_LL).clamp(min=1e-8)
    d_LH = (1.0 - g_LH).clamp(min=1e-8)
    d_HL = (1.0 - g_HL).clamp(min=1e-8)
    d_HH = (1.0 - g_HH).clamp(min=1e-8)

    # ── Step 4: β weights — proportional to deficit, shaped by gamma_power ───
    b_LL = d_LL.pow(gamma_power)
    b_LH = d_LH.pow(gamma_power)
    b_HL = d_HL.pow(gamma_power)
    b_HH = d_HH.pow(gamma_power)

    # ── Step 5: RMS normalization — enforces mean(β²) = 1  (CNS Eq. 11) ─────
    # Shapes the relative noise distribution across subbands while keeping
    # the expected total injected energy constant.  β ratios are preserved;
    # only the overall scale is set.  This alone is exact when H and W are
    # both even (no padding needed → no crop → no energy leak).
    rms = ((b_LL**2 + b_LH**2 + b_HL**2 + b_HH**2) / 4.0).sqrt().clamp(min=1e-8)
    b_LL = b_LL / rms
    b_LH = b_LH / rms
    b_HL = b_HL / rms
    b_HH = b_HH / rms

    # ── Step 6: Apply β to noise subbands ────────────────────────────────────
    n_p, _ = _pad_even(noise_f)
    nLL, nLH, nHL, nHH = haar_dwt2d(n_p)

    # β shape (B,1,1,1) broadcasts over (B,C,h,w)
    nLL = nLL * b_LL
    nLH = nLH * b_LH
    nHL = nHL * b_HL
    nHH = nHH * b_HH

    n_colored = haar_idwt2d(nLL, nLH, nHL, nHH)[..., :H, :W]

    # ── Step 7: Crop-energy correction ───────────────────────────────────────
    # When H or W is odd, _pad_even added a reflect row/col before DWT, and
    # the [..,:H,:W] crop above discards the corresponding IDWT output row/col.
    # That crop introduces a small energy leak that Step 5 cannot anticipate.
    # A single scalar rescale restores the correct std without disturbing the
    # subband β ratios established in Step 5.
    # For even-sized inputs there is no crop, so this is a near-identity op.
    orig_std  = noise_f.std().clamp(min=1e-8)
    n_colored = n_colored * (orig_std / n_colored.std().clamp(min=1e-8))

    # ── Step 8: Strength lerp ─────────────────────────────────────────────────
    if strength < 1.0:
        n_colored = torch.lerp(noise_f, n_colored, strength)

    return n_colored.to(dtype=orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Noise Sampler Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cns_noise_sampler(
    x_initial:            torch.Tensor,
    strength:             float,
    gamma_power:          float,
    gamma_scale:          float,
    original_noise_sampler=None,
):
    """
    Build a CNS noise sampler callable and a mutable state dict.

    The state dict holds ``state["x_current"]``, which must be updated each
    step via a k-diffusion callback so the γ proxy reflects the live x_t.

    Returns
    -------
    (cns_noise_sampler_fn, state_dict)

    cns_noise_sampler_fn  has signature  fn(sigma, sigma_next) → noise tensor
    state_dict            caller should update ["x_current"] each step
    """
    state = {"x_current": x_initial.detach().clone()}

    # 새 샘플링 런이 시작될 때 디버그 카운터/이전 shape 기록을 리셋합니다.
    # (리셋하지 않으면 이전 생성의 마지막 shape이 남아 있어 새 런의 첫 스텝에
    #  불필요한 "← 변경됨" 표시가 잘못 뜰 수 있습니다.)
    if DEBUG_LOGGING:
        color_noise_wavelet._prev_shape = None
        color_noise_wavelet._step_i = 0

    def cns_noise_sampler(sigma, sigma_next):
        x_t = state["x_current"]

        # Generate base white noise
        if original_noise_sampler is not None:
            try:
                noise = original_noise_sampler(sigma, sigma_next)
            except Exception:
                noise = torch.randn_like(x_t)
        else:
            noise = torch.randn_like(x_t)

        try:
            return color_noise_wavelet(
                noise, x_t,
                strength=strength,
                gamma_power=gamma_power,
                gamma_scale=gamma_scale,
            )
        except Exception as exc:
            print(f"[CNS] Warning: coloring skipped at this step – {exc}")
            return noise

    return cns_noise_sampler, state


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI Node
# ─────────────────────────────────────────────────────────────────────────────

class CNSSamplerPatch:
    """
    CNS Sampler Patch node.

    Wraps any ancestral SAMPLER and replaces its white noise injection with
    frequency-aware colored noise (CNS).

    Usage in SamplerCustomAdvanced
    ──────────────────────────────
      SAMPLER (e.g. euler_ancestral_cfg_pp)
          │
          ▼
      CNSSamplerPatch
          │
          ▼
      SamplerCustomAdvanced ← MODEL (with DCW patch, optional)

    The DCWModelPatch and CNSSamplerPatch are orthogonal:
      • DCW corrects x0_pred in the post-cfg hook (model-level).
      • CNS colors the ancestral noise in the sampler step (sampler-level).
    Both can be active simultaneously with no hook conflicts.
    """

    RETURN_TYPES  = ("SAMPLER",)
    RETURN_NAMES  = ("sampler",)
    FUNCTION      = "patch"
    CATEGORY      = "sampling/custom_sampling/samplers"
    DESCRIPTION   = (
        "CNS Sampler Patch: replaces white ancestral noise with wavelet-colored noise.\n"
        "Low-frequency noise is suppressed once those structures are built; "
        "high-frequency noise is boosted while fine details remain unresolved.\n"
        "Variance is strictly preserved per step (CNS energy budget constraint).\n"
        "Compatible with euler_ancestral, euler_ancestral_cfg_pp, dpm_2_ancestral, "
        "dpmpp_2s_ancestral, dpmpp_sde. Inert on deterministic (ODE) samplers."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),

                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min":    0.0,
                    "max":    1.0,
                    "step":   0.05,
                    "round":  0.01,
                    "tooltip": (
                        "[CNS] Effect strength. Lerp between white noise (0.0) and "
                        "fully colored noise (1.0).\n"
                        "Start at 1.0 and reduce only if artifacts appear."
                    ),
                }),

                "gamma_power": ("FLOAT", {
                    "default": 0.5,
                    "min":    0.1,
                    "max":    2.0,
                    "step":   0.05,
                    "round":  0.01,
                    "tooltip": (
                        "[CNS] Exponent applied to the subband deficit before computing β.\n"
                        "0.5 (√) = CNS paper default — smooth, proportional routing.\n"
                        "Higher (0.75–1.0) → sharper frequency routing, more aggressive.\n"
                        "Lower (0.25–0.4)  → softer routing, closer to white noise."
                    ),
                }),

                "gamma_scale": ("FLOAT", {
                    "default": 2.0,
                    "min":    0.1,
                    "max":    25.0,
                    "step":   0.1,
                    "round":  0.01,
                    "tooltip": (
                        "[CNS] γ proxy divider. Scales down the subband energy ratios "
                        "before computing deficits, preventing premature over-routing.\n"
                        "Equivalent to 'γ Divider' in CNS paper (range 1.73–25.0).\n"
                        "1.0 = most aggressive (full energy ratio).\n"
                        "2.0–5.0 = recommended starting range for most models.\n"
                        "Increase if early steps show structural instability."
                    ),
                }),
            }
        }

    def patch(
        self,
        sampler,
        strength:      float,
        gamma_power:   float,
        gamma_scale:   float,
    ):
        original_fn = sampler.sampler_function

        # Capture params in closure
        _strength    = strength
        _gamma_power = gamma_power
        _gamma_scale = gamma_scale
        def _wrap_kd_fn(kd_fn):
            """
            Wrap a single k-diffusion ancestral sampler function with CNS.
            Deterministic samplers never call noise_sampler so CNS is inert.
            """
            import functools

            @functools.wraps(kd_fn)
            def wrapped(model, x, sigmas, **kwargs):
                orig_noise_sampler = kwargs.pop("noise_sampler", None)
                cns_ns, state = make_cns_noise_sampler(
                    x_initial=x,
                    strength=_strength,
                    gamma_power=_gamma_power,
                    gamma_scale=_gamma_scale,
                    original_noise_sampler=orig_noise_sampler,
                )
                user_callback = kwargs.pop("callback", None)

                def cns_callback(info: dict):
                    x_current = info.get("x")
                    if x_current is not None:
                        state["x_current"] = x_current.detach()
                    if user_callback is not None:
                        user_callback(info)

                kwargs["noise_sampler"] = cns_ns
                kwargs["callback"]      = cns_callback
                return kd_fn(model, x, sigmas, **kwargs)

            return wrapped

        def patched_sampler_fn(model, x, sigmas, **kwargs):
            """
            Top-level sampler function registered in KSAMPLER.

            Two mutually exclusive paths:

            A) Direct path — original_fn IS a kd sample_* function.
               CNS is injected once via kwargs["noise_sampler"].
               No monkey-patching needed; original_fn is called directly.

            B) Wrapper path — original_fn is an outer wrapper (e.g. SPEED)
               that internally looks up and calls kds.sample_* by name.
               CNS is injected by temporarily monkey-patching all ancestral
               sample_* functions in comfy.k_diffusion.sampling just before
               the call, then restoring them in a finally block.
               kwargs["noise_sampler"] is NOT touched here so the wrapper
               can forward it unmodified (or ignore it).

            The two paths are detected by checking whether original_fn's
            unwrapped identity matches any function already in kds.  If it
            does, we are in path A and skip monkey-patching entirely.
            """
            import comfy.k_diffusion.sampling as kds

            # ── Identify kd sample_* functions and their noise_sampler support ─
            kds_ancestral = {}   # attr_name → (original_fn_obj, accepts_ns)
            for attr in dir(kds):
                if not attr.startswith("sample_"):
                    continue
                fn = getattr(kds, attr, None)
                if fn is None or not callable(fn):
                    continue
                try:
                    unwrapped = inspect.unwrap(fn)
                    sig = inspect.signature(unwrapped)
                    has_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                    accepts_ns = has_var_kw or "noise_sampler" in sig.parameters
                except (ValueError, TypeError, StopIteration):
                    accepts_ns = True
                kds_ancestral[attr] = (fn, accepts_ns)

            # ── Detect which path applies ─────────────────────────────────────
            try:
                unwrapped_orig = inspect.unwrap(original_fn)
            except Exception:
                unwrapped_orig = original_fn

            is_direct_kd = any(
                inspect.unwrap(fn) is unwrapped_orig
                for fn, _ in kds_ancestral.values()
                if callable(fn)
            )

            orig_noise_sampler = kwargs.pop("noise_sampler", None)

            if is_direct_kd:
                # ── Path A: original_fn is itself a kd sampler ────────────────
                # Check if it accepts noise_sampler (should, but be safe).
                try:
                    sig_orig = inspect.signature(unwrapped_orig)
                    has_var_kw_orig = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig_orig.parameters.values()
                    )
                    orig_accepts_ns = has_var_kw_orig or "noise_sampler" in sig_orig.parameters
                except (ValueError, TypeError, StopIteration):
                    orig_accepts_ns = True

                if orig_accepts_ns:
                    cns_ns, state = make_cns_noise_sampler(
                        x_initial=x,
                        strength=_strength,
                        gamma_power=_gamma_power,
                        gamma_scale=_gamma_scale,
                        original_noise_sampler=orig_noise_sampler,
                    )
                    user_callback = kwargs.pop("callback", None)

                    def cns_callback(info: dict):
                        x_current = info.get("x")
                        if x_current is not None:
                            state["x_current"] = x_current.detach()
                        if user_callback is not None:
                            user_callback(info)

                    kwargs["noise_sampler"] = cns_ns
                    kwargs["callback"]      = cns_callback
                else:
                    if orig_noise_sampler is not None:
                        kwargs["noise_sampler"] = orig_noise_sampler

                return original_fn(model, x, sigmas, **kwargs)

            else:
                # ── Path B: wrapper — monkey-patch kds, then call ─────────────
                patched_attrs = {
                    attr: (fn, _wrap_kd_fn(fn))
                    for attr, (fn, accepts_ns) in kds_ancestral.items()
                    if accepts_ns
                }
                for attr, (_, wrapped) in patched_attrs.items():
                    setattr(kds, attr, wrapped)

                # Restore original noise_sampler into kwargs for the wrapper
                # to forward (it is NOT wrapped again here — CNS lives in the
                # monkey-patched kds functions only).
                if orig_noise_sampler is not None:
                    kwargs["noise_sampler"] = orig_noise_sampler

                try:
                    return original_fn(model, x, sigmas, **kwargs)
                finally:
                    for attr, (original, _) in patched_attrs.items():
                        setattr(kds, attr, original)

        new_sampler = comfy.samplers.KSAMPLER(
            patched_sampler_fn,
            extra_options=sampler.extra_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (new_sampler,)



# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "CNSSamplerPatch": CNSSamplerPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CNSSamplerPatch": "CNS Sampler Patch (Colored Noise)",
}
