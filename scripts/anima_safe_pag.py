"""Anima Safe PAG — standalone Perturbed Attention Guidance for Forge Neo.

A SELF-CONTAINED extension script (like the PiD / Anima-Ref PoC), completely
independent of the SAM3 machinery and touching no Forge core file.

What it does
------------
Official-style hard PAG and Gaussian-query SEG for Anima / Cosmos /
Predict2-style DiT models, with the earlier soft-PAG / uniform-SEG
implementation retained behind an opt-in compatibility toggle. The Anima
batching and block selection were originally ported from the ComfyUI node
``iljung1106/comfyui-anima-safe-pag``.

How it hooks into Forge Neo (no core edits)
-------------------------------------------
Forge Neo's real sampling path (``backend/sampling/sampling_function.py``)
does NOT invoke ``sampler_calc_cond_batch_function`` (the setter exists on the
patcher but the sampler calls ``calc_cond_uncond_batch`` directly). The two
hooks it DOES honour are:

  * ``model_options["model_function_wrapper"]`` — wraps ``model.apply_model``
    with ``{"input","timestep","c","cond_or_uncond"}`` (proven by the PoC).
  * ``post_cfg_function(args)`` — args carry ``denoised`` (= cfg_result),
    ``cond_denoised``, ``uncond_denoised``, ``sigma``, ``input`` …

So we:

  1. In the model-function wrapper, append a COPY of the *cond* rows to the
     batch and run a SINGLE ``apply_model`` (no separate forward → the extra
     PAG prediction rides along in the same kernel launch). Self-attention on
     the appended rows is perturbed (hard value-only PAG or query-blur SEG). We return the original
     rows untouched, so CFG proceeds exactly as normal, and stash the PAG x0
     prediction.
  2. ``model.apply_model`` already returns the predictor-converted denoised
     (x0) prediction in Forge. In ``post_cfg_function`` we therefore apply the
     original Safe-PAG correction directly:
     ``guidance = scale*(cond_denoised - pag_denoised)`` and
     ``result = cfg_result + rescale(guidance)``.

Attention perturbation
----------------------
Current Forge Neo routes Anima self-attention through
``SelfCrossAttention.torch_attention_op`` and then the selected
``backend.attention.attention_function`` backend. We patch that static method
once. Hard PAG substitutes the pre-projection output with the raw value path;
official SEG Gaussian-blurs the appended row's query over the real H/W axes
captured by the surrounding Anima block. Legacy mode restores the earlier
tunable value/uniform-output interpolation.

Safety
------
Everything is wrapped in try/except and ALWAYS falls back to the normal
``apply_model`` / ``denoised`` on any error, so enabling this can never break a
generation — worst case it logs and renders normally.

⚠️ 실험 기능 — Forge Neo 2.27 + `anima_baseV10`에서 True/False end-to-end
생성을 검증했습니다. 다른 Anima 파생/정밀 조합은 콘솔의 ``[AnimaSafePAG]``
첫 weak delta와 generation summary로 실제 적용 여부를 확인하세요.
"""
from __future__ import annotations

import gc
import math
import importlib
import re
import sys
import time
import traceback
import weakref
from functools import partial
from pathlib import Path

import gradio as gr

from modules import script_callbacks, scripts

from sam3ext import layout_lanes
try:
    from modules import shared
except ImportError:  # standalone/unit-test loader
    shared = None  # type: ignore
from sam3ext.guidance.cns import color_noise_wavelet
from sam3ext.guidance.cwm_smc import (
    SMC_PRESET_NAMES,
    apply_cwm_error,
    apply_smc_error,
    compose_cfg,
    normalize_smc_preset,
    resolve_smc_preset,
)
from sam3ext.guidance.dave import apply_dave
from sam3ext.guidance.dcw import apply_dcw
from sam3ext.guidance.modulation import (
    clear_modulation_caches,
    list_clip_l_models,
    prepare_block_modulations,
)
from sam3ext.guidance.runtime import GuidanceRuntime
try:
    from guidance_diagnostics import guidance_diagnostics_enabled
except ImportError:  # standalone/unit-test loader without extension root on sys.path
    def guidance_diagnostics_enabled() -> bool:
        return False

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


def _log(msg: str) -> None:
    """Print a diagnostic without ever being able to abort a generation.

    Several of these messages carry non-ASCII characters, and on a Windows
    console using a legacy code page (cp949, cp1252, ...) print raises
    UnicodeEncodeError. _log runs inside the model wrapper and the post-CFG
    hook — including from their except handlers — so an exception here would
    replace a handled fallback with a hard sampler failure.
    """
    text = f"[AnimaSafePAG] {msg}"
    try:
        print(text, file=sys.stderr)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stderr, "encoding", None) or "ascii"
            print(
                text.encode(encoding, "replace").decode(encoding, "replace"),
                file=sys.stderr,
            )
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared runtime state (single-threaded per generation; the sampler drives one
# denoise step at a time — wrapper then post_cfg — so a plain dict is enough).
# ---------------------------------------------------------------------------

_STATE: dict = {
    "on": False,          # any perturbation guidance active this generation
    # Attention-perturbation method: "pag" (value-only) or "seg" (query blur),
    # mutually exclusive; None → no attention perturbation (SLG only).
    "attn_method": "pag",
    "attn_scale": 4.0,    # guidance strength for the attention-perturbation term
    "strength": 0.75,     # perturbation blend (1=full official target)
    "legacy_attn": False, # False=official PAG value / Gaussian-query SEG
    "seg_sigma": 100.0,   # official SEG query-blur sigma (>9999=infinite)
    "head_spec": "",      # empty=all attention heads; supports 0,2,4-7
    "attn_targets": set(),  # block indices for attention perturbation
    # SLG (Skip Layer Guidance): skip whole blocks on a separate weak prediction.
    "slg_on": False,
    "slg_scale": 3.0,
    "slg_targets": set(),  # block indices to skip
    "rescale": 0.20,      # std-matching rescale factor
    "rescale_mode": "full",  # full=CFG+guidance, partial=cond+guidance std source
    "start": 0.0,         # start percent of sampling
    "end": 0.7,           # end percent of sampling
    "total": 20,          # total steps this pass
    "step": 0,            # current step counter
    # A sampling step may invoke the model wrapper more than once when Forge
    # cannot fit cond+uncond in one batch. post_cfg closes the step after all
    # of those calls have completed.
    "step_open": False,
    # Row ranges of the appended weak predictions in the enlarged batch.
    "attn_b0": None, "attn_b1": None,   # attention-perturbed cond rows
    "slg_b0": None, "slg_b1": None,     # layer-skipped cond rows
    "any_b0": None,       # min appended index (marks "inside enlarged forward")
    "active": 0,          # >0 while inside a targeted self_attn.forward
    "attn_spatial_shape": None,  # (T,H,W) captured by the surrounding Anima block
    "attn_hook_hits": 0,  # successful weak-row perturbations in this enlarged forward
    "attn_hook_hits_total": 0,  # cumulative successful hook calls this generation
    "attn_last_rel_delta": None,  # ||weak-cond|| / ||cond|| diagnostic
    "attn_diag_logged": False,    # emit one hook-health diagnostic per generation
    "attn_shape_warned": False,   # emit layout mismatch warning only once
    "attn_raw": None,     # stashed attention-perturbed denoised x0
    "slg_raw": None,      # stashed layer-skipped denoised x0
    "cond_raw": None,     # cond rows the weak predictions were derived from
    "apg_autooff_rescale": True,  # skip PAG rescale while APG is on (toggleable)
    "adg_skipped": False,  # current model evaluation used cond-only ADG
    "requested_start": 0.0,
    "requested_end": 0.7,
    "range_mode": "continuous",
    # Per-generation diagnostics. These make a silent no-op distinguishable
    # from a genuinely active XYZ True cell in the WebUI console.
    "wrapper_calls": 0,
    "weak_steps": 0,
    "applied_steps": 0,
    "apg_steps": 0,
    "adg_skipped_steps": 0,
    "combined_calls": 0,
    "split_cond_calls": 0,
    "split_uncond_calls": 0,
    "control_blocked_calls": 0,
    "wrapper_fallbacks": 0,
    "requested_pert": False,
    "requested_apg": False,
    "requested_adg": False,
    "requested_cfg_mode": "preserve",
    "requested_cfg_stack": False,
    "requested_smc": False,
    "requested_cwm": False,
    "requested_dcw": False,
    "requested_rdc": False,
    "requested_dave": False,
    "requested_cns": False,
    "requested_modulation": False,
    "requested_method": None,
    "engine": "?",
    "diag_started_at": None,
    "delta_logged": False,
    # 확장 배치가 OOM 으로 실패하면 그 생성(같은 p 의 남은 패스·배치)의
    # perturbation 을 끈다. p 는 약한 참조로만 들고 postprocess 에서 푼다.
    "pass_owner": None,
    "pert_oom_owner": None,
    # 앞쪽 블록 중복 제거(Forge 설정 OPT_PREFIX_DEDUP, 생성마다 읽음). 첫 target
    # 블록 이전에는 weak 행이 cond 행과 입력·연산이 같으므로 원래 배치 행만
    # forward 하고 weak 자리에는 cond 행 출력을 복사한다.
    "prefix_dedup": True,
    # 공식 SEG blur 를 가로·세로 1D depthwise 두 번으로(Forge 설정 OPT_SEG_SEPARABLE).
    "seg_separable": True,
    # 확장 forward 한 번 동안만 유효한 표식(_clear_markers 가 지운다).
    "dedup_until": None,  # 첫 target 블록 번호. None = 이번 forward 에서 중복 제거 안 함
    "dedup_src": None,    # 붙인 weak 행의 원본 cond 행 인덱스(원래 배치 기준)
    "dedup_next": 0,      # 다음에 와야 하는 블록 번호(연속·순서 확인)
    # 생성 단위 진단/안전장치.
    "dedup_blocks": 0,     # 중복 제거로 돈 블록 호출 수
    "dedup_fallbacks": 0,  # 조건이 안 맞아 예전 경로(전체 배치)로 돈 횟수
    "dedup_warned": False,
    "dedup_disabled": False,  # 잘린 배치 forward 가 예외를 내면 이 생성은 끈다
}

# Forge 설정(섹션 'SAM Extra Guidance'). infotext 이름은 붙여넣기 때 같은 설정으로
# 되돌아가도록 OptionInfo(infotext=...) 로도 등록한다(값은 "True"/"False").
OPT_PREFIX_DEDUP = "sam3_guidance_pag_prefix_dedup"
OPT_SEG_SEPARABLE = "sam3_guidance_seg_separable_blur"
INFOTEXT_PREFIX_DEDUP = "Anima PAG prefix dedup"
INFOTEXT_SEG_SEPARABLE = "Anima SEG separable blur"

_EXTRA_GENERATION_PARAM_KEYS = (
    "Anima Perturbation Guidance",
    "Anima APG",
    "Anima Adaptive Guidance",
    "Anima CFG Orchestrator",
    "Anima DCW",
    "Anima RDC",
    "Anima DAVE",
    "Anima CNS Wavelet Noise",
    "Anima Modulation Guidance",
    INFOTEXT_PREFIX_DEDUP,
    INFOTEXT_SEG_SEPARABLE,
)


def _read_bool_option(name: str, default: bool = True) -> bool:
    """Forge 설정값. 설정이 없거나 Forge 밖(테스트)이면 ``default``."""
    try:
        opts = getattr(shared, "opts", None)
        if opts is None:
            return default
        return bool(getattr(opts, name, default))
    except Exception:
        return default


def _on_ui_settings() -> None:
    section = ("sam3_guidance", "SAM Extra Guidance")
    shared.opts.add_option(
        OPT_PREFIX_DEDUP,
        shared.OptionInfo(
            True,
            "PAG/SEG/SLG: 첫 target 블록 이전의 weak 행 중복 계산 건너뛰기",
            gr.Checkbox,
            section=section,
            infotext=INFOTEXT_PREFIX_DEDUP,
        ).info(
            "켜면(기본) 첫 target 블록(기본 18) 이전 블록은 원래 cond/uncond 행만 돌리고 weak 행 자리에는 "
            "cond 행 출력을 복사합니다 — 그 블록들에서 weak 행은 cond 행과 입력·연산이 같습니다. "
            "기본 블록(18)에서 PAG 한 장에 약 9% 빨라지고 앞쪽 블록의 활성 VRAM 도 줄어듭니다(추정). 행 수가 안 맞거나 "
            "모르는 블록 인자가 있으면 그 블록부터 예전 경로(전체 배치)로 돕니다. 행렬 곱의 배치 크기가 "
            "달라져 결과가 아주 미세하게 달라질 수 있어 infotext 에 남깁니다. 끄면 예전 경로 그대로입니다."
        ),
    )
    shared.opts.add_option(
        OPT_SEG_SEPARABLE,
        shared.OptionInfo(
            True,
            "SEG(공식): query Gaussian blur 를 가로·세로 1D 두 번으로 계산",
            gr.Checkbox,
            section=section,
            infotext=INFOTEXT_SEG_SEPARABLE,
        ).info(
            "켜면(기본) 같은 Gaussian 커널을 2D 한 번 대신 가로·세로 1D depthwise conv 두 번으로 적용합니다 — "
            "커널 크기 k 에서 픽셀당 곱셈이 k² 에서 2k 로 줄어듭니다. 수학적으로 같은 blur 지만 반올림 순서가 "
            "달라 결과가 아주 미세하게 달라질 수 있어 infotext 에 남깁니다. 끄면 예전 2D conv 그대로입니다."
        ),
    )


def _register_settings_hook() -> None:
    register = getattr(script_callbacks, "on_ui_settings", None)
    if register is not None and shared is not None:
        try:
            register(_on_ui_settings)
        except Exception:
            pass


_register_settings_hook()


# ---------------------------------------------------------------------------
# APG (Adaptive Projected Guidance) — a separate, model-agnostic post-CFG
# guidance variant. It REPLACES the CFG-combine result with a projected one so
# high guidance no longer oversaturates. Runs on any engine (not just Anima);
# unlike PAG it needs no attention patching, only the post_cfg args.
#
# With eta=1, norm_threshold=0, momentum=0 it reduces EXACTLY to standard CFG,
# so it is safe as a default-on-but-neutral toggle.
# ---------------------------------------------------------------------------

_APG: dict = {
    "on": False,
    "eta": 0.0,           # weight of the cond-parallel guidance component
    "norm_threshold": 15.0,  # clamp guidance L2 norm to this (0 = disabled)
    "momentum": 0.0,      # running-average coefficient across steps
    "avg": None,          # momentum buffer (per-generation)
    "last_sigma": None,   # to detect a new sampling pass (sigma goes up → reset)
}


# ---------------------------------------------------------------------------
# Adaptive Guidance — skip the uncond (negative) forward in the later steps,
# where it barely affects the result, for a near-lossless speedup. Implemented
# in the model_function_wrapper by running cond-only and setting the uncond
# output equal to cond (→ CFG collapses to the cond prediction that step).
# Model-agnostic; shares the wrapper with perturbation guidance.
# ---------------------------------------------------------------------------

_ADG: dict = {
    "on": False,
    "start": 0.5,     # skip uncond once this fraction of steps has elapsed
    "interval": 0,    # 0 = skip every step after start; N>0 = still keep uncond every Nth step
}


# ---------------------------------------------------------------------------
# Unified post-CFG, per-block, and sampler-noise feature configuration.
# All defaults are neutral; merely installing/updating the extension cannot
# alter a generation.
# ---------------------------------------------------------------------------

_CFG: dict = {
    # Independent toggles, always applied in the order SMC -> APG -> CWM.
    "smc_on": False,
    "apg_on": False,
    "cwm_on": False,
    # Legacy mutually-exclusive radio + stack checkbox, kept so saved infotext,
    # API calls and XYZ grids from before the split still resolve. Both are
    # OR-ed into the toggles above by _cfg_base_flags().
    "mode": "preserve",  # preserve | apg | cwm | smc | smc+cwm
    "experimental_stack": False,
    "alpha_low": 0.30,
    "alpha_high": 0.15,
    "smc_preset": "Off",
    "smc_resolved_preset": "Off",
    "smc_lambda": 6.0,
    "smc_k": 0.10,
    "steps": 0,
    "fit_error": None,
    "effective_scale": None,
    "external_cfg_detected": False,
    "warned": False,
    # CFG 1 guard: Forge skips the uncond pass at cond_scale≈1, so there is no
    # CFG error for SMC/APG/CWM to reshape; log that skip once per generation.
    "cfg1_warned": False,
    # 이번 post-CFG 호출에서 CFG 1 가드로 SMC/APG/CWM 을 건너뛰었는가(스텝마다
    # _apply_cfg_base 가 다시 씀). 건너뛴 스텝은 APG 가 돌지 않았으므로 PAG
    # rescale 자동 끄기(apg_autooff_rescale)도 적용하지 않는다.
    "base_skipped": False,
    # fit_error 는 경고 한 줄과 [VERIFY] 요약에만 쓰인다. 진단이 꺼져 있으면
    # 패스마다 첫 평가에서만 재고(동기화 1회), 이후 스텝은 건너뛴다.
    "fit_checked": False,
}

_DCW: dict = {
    "on": False,
    "dcw_on": False,
    "rdc_on": False,
    "lambda_low": 0.10,
    "lambda_high": 0.02,
    "steps": 0,
    "dcw_steps": 0,
    "rdc_steps": 0,
    "rdc_tau": 0.0,
    "rdc_alpha_ll": 0.03,
    "rdc_alpha_hh": 0.0,
}

_DAVE: dict = {
    "on": False,
    "strength": 0.30,
    "tau": 0.10,
    "targets": set(),
    "steps": 0,
}

_CNS: dict = {
    "on": False,
    "strength": 1.0,
    "gamma_power": 0.5,
    "gamma_scale": 3.0,
    "warned": False,
}

_MOD: dict = {
    "on": False,
    "weight": 3.0,
    "targets": set(),
    # CPU float32 [blocks, 3*model_channels]. Only selected rows are copied to
    # the active model device and cached as tiny per-block vectors.
    "block_modulations": None,
    "typed": {},
    "hits": 0,
    "warned": False,
    "clip_model": "",
    "adapter_path": "",
    "start_layer": 0,
    "end_layer": -1,
    "base_prompt": "",
    "positive_prompt": "",
    "negative_prompt": "",
}

_RUNTIME = GuidanceRuntime(
    state=_STATE,
    apg=_APG,
    adg=_ADG,
    cfg=_CFG,
    dcw=_DCW,
    dave=_DAVE,
    cns=_CNS,
)

# Original (unpatched) current Forge Neo Anima attention op.
_ORIG_ANIMA_ATTN_OP = None
_ORIG_CNS_DEFAULT_FACTORY = None
_ORIG_CNS_BROWNIAN_CALL = None
_PATCH_OWNER = object()

# What we patched, tracked so _teardown_global_patches() can restore the
# originals cleanly (e.g. on extension reload). The attention op is a
# class-level global and the CNS sources are k-diffusion module globals — those
# outlive the model and are the real cross-extension/model leak surface. Block /
# self_attn wraps live on the model instance (recreated on reload), tracked
# weakly so we never keep a dead model alive.
_PATCHED_ATTN_OWNER = None
_PATCHED_BLOCKS: "weakref.WeakSet" = weakref.WeakSet()
_PATCHED_SELFATTN: "weakref.WeakSet" = weakref.WeakSet()
_PATCHED_CNS_TARGETS: list = []  # [(sampling_module, brownian_cls_or_None)]


# ---------------------------------------------------------------------------
# Attention perturbation — patch the active Anima attention path once.
# ---------------------------------------------------------------------------


def _gaussian_blur_2d(img, sigma: float, separable: bool = True):
    """Official SEG separable Gaussian kernel, applied depthwise over H/W.

    ``separable=True`` 는 같은 1D 커널을 가로(1×k)·세로(k×1) depthwise conv 로
    두 번 적용한다(픽셀당 곱셈 k² → 2k). 반사 패딩을 먼저 한 번에 하므로 2D
    conv 와 수학적으로 같고, 반올림 순서만 달라 결과가 아주 미세하게 다를 수
    있다. ``False`` 는 예전 2D 외적 커널 conv 그대로다(Forge 설정으로 고른다)."""
    if F is None or sigma <= 0:
        return img
    height, width = int(img.shape[-2]), int(img.shape[-1])
    limit = min(height, width)
    if limit < 2:
        return img
    requested = math.ceil(6.0 * sigma)
    kernel_size = requested + 1 - requested % 2
    # reflect padding requires pad < both spatial dimensions. This is the
    # official SEG clamp generalized from square SDXL maps to rectangular maps.
    max_odd = limit if limit % 2 else limit - 1
    kernel_size = max(1, min(kernel_size, max_odd))
    if kernel_size <= 1:
        return img
    half = (kernel_size - 1) * 0.5
    x = torch.linspace(-half, half, steps=kernel_size, device=img.device, dtype=torch.float32)
    pdf = torch.exp(-0.5 * (x / float(sigma)).pow(2))
    kernel_1d = (pdf / pdf.sum()).to(dtype=img.dtype)
    if separable:
        channels = img.shape[-3]
        img = F.pad(img, [kernel_size // 2] * 4, mode="reflect")
        horizontal = kernel_1d.reshape(1, 1, 1, kernel_size).expand(
            channels, 1, 1, kernel_size
        )
        vertical = kernel_1d.reshape(1, 1, kernel_size, 1).expand(
            channels, 1, kernel_size, 1
        )
        img = F.conv2d(img, horizontal, groups=channels)
        return F.conv2d(img, vertical, groups=channels)
    kernel_2d = torch.mm(kernel_1d[:, None], kernel_1d[None, :])
    kernel_2d = kernel_2d.expand(img.shape[-3], 1, kernel_size, kernel_size)
    img = F.pad(img, [kernel_size // 2] * 4, mode="reflect")
    return F.conv2d(img, kernel_2d, groups=img.shape[-3])


def _parse_attention_heads(spec: str, n: int) -> set:
    """Parse an optional attention-head list; an empty value selects all."""
    if n <= 0:
        return set()
    spec = (spec or "").strip()
    if not spec:
        return set(range(n))
    out: set = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                start, end = int(a), int(b)
                if start <= end:
                    out.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return {i for i in out if 0 <= i < n}


def _head_index(heads: list, device):
    """head 축 인덱스를 장치 위 LongTensor 로(캐시) 돌려준다.

    파이썬 리스트로 인덱싱하면 PyTorch 가 매번 CPU LongTensor 를 만들어 GPU 로
    올린다(forward 중간의 동기 H2D, 인덱싱마다 1회). 같은 값의 장치 텐서로
    인덱싱하면 같은 index/index_put 커널이 같은 출력 배치로 돌므로 결과는
    비트 단위로 같다. slice 로 바꾸면 lerp 입력의 메모리 배치가 달라져 CPU 에서
    마지막 비트가 달라지는 경우가 있어(FMA 벡터 경로) 쓰지 않는다."""
    return _index_tensor(heads, device)


def _official_seg_query(
    query,
    a0: int,
    a1: int,
    sigma: float,
    spatial_shape=None,
    strength: float = 1.0,
    head_spec: str = "",
    separable: bool = True,
):
    """Blur only appended weak-row queries over Anima's real spatial axes.

    Current Forge Neo supplies ``[B,S,heads,dim]`` after the surrounding block
    has flattened ``[T,H,W]`` into ``S``. The block wrapper records those axes
    so this function never has to assume a square image. A 6-D layout remains
    accepted for compatible forks and focused tests.
    """
    weak = query[a0:a1]
    if weak.ndim == 6:  # B,T,H,W,N,D (compatible fork/test layout)
        b, t, h, w, n, d = weak.shape
        spatial = weak.permute(0, 1, 4, 5, 2, 3).reshape(b * t, n * d, h, w)
        if sigma > 9999.0:
            spatial = spatial.mean(dim=(-2, -1), keepdim=True).expand_as(spatial)
        else:
            spatial = _gaussian_blur_2d(spatial, sigma, separable)
        perturbed = spatial.reshape(b, t, n, d, h, w).permute(0, 1, 4, 5, 2, 3)
    elif weak.ndim == 4:  # B,S,N,D (current Forge Neo)
        if not spatial_shape or len(spatial_shape) != 3:
            raise ValueError("official SEG needs the surrounding block's T/H/W shape")
        t, h, w = (int(v) for v in spatial_shape)
        b, seq, n, d = weak.shape
        if t <= 0 or h <= 0 or w <= 0 or seq != t * h * w:
            raise ValueError(
                f"official SEG shape mismatch: seq={seq}, T/H/W={(t, h, w)}"
            )
        weak_6d = weak.reshape(b, t, h, w, n, d)
        spatial = weak_6d.permute(0, 1, 4, 5, 2, 3).reshape(
            b * t, n * d, h, w
        )
        if sigma > 9999.0:
            spatial = spatial.mean(dim=(-2, -1), keepdim=True).expand_as(spatial)
        else:
            spatial = _gaussian_blur_2d(spatial, sigma, separable)
        perturbed = (
            spatial.reshape(b, t, n, d, h, w)
            .permute(0, 1, 4, 5, 2, 3)
            .reshape(b, seq, n, d)
        )
    else:
        raise ValueError(f"unsupported query layout {tuple(query.shape)}")
    heads = sorted(_parse_attention_heads(head_spec, int(weak.shape[-2])))
    if not heads:
        return query
    strength = min(1.0, max(0.0, float(strength)))
    blended = weak.clone()
    head_idx = _head_index(heads, weak.device)
    blended[..., head_idx, :] = torch.lerp(
        weak[..., head_idx, :],
        perturbed[..., head_idx, :],
        strength,
    )
    result = query.clone()
    result[a0:a1] = blended
    return result


def _patched_anima_attention_op(query, key, value, *args, **kwargs):
    """Drop-in for current Forge Neo's ``SelfCrossAttention.torch_attention_op``.

    The inputs are ``[B, seq, heads, dim]`` and the original output is
    ``[B, seq, heads*dim]``. Perturbing here keeps the operation before
    ``output_proj``, matching the older SDPA patch semantically.
    """
    original = _ORIG_ANIMA_ATTN_OP
    if not callable(original):
        raise RuntimeError("original Anima attention op is unavailable")
    if _STATE["active"] <= 0 or _STATE["attn_b0"] is None:
        return original(query, key, value, *args, **kwargs)
    original_query = query
    out = None
    try:
        a0, a1 = _STATE["attn_b0"], _STATE["attn_b1"]
        method = _STATE["attn_method"]
        legacy = bool(_STATE["legacy_attn"])
        if not method or a1 is None or a1 > query.shape[0]:
            return original(query, key, value, *args, **kwargs)

        if method == "seg" and not legacy:
            query = _official_seg_query(
                query,
                a0,
                a1,
                float(_STATE["seg_sigma"]),
                _STATE.get("attn_spatial_shape"),
                float(_STATE["strength"]),
                str(_STATE.get("head_spec", "")),
                bool(_STATE.get("seg_separable", True)),
            )
        out = original(query, key, value, *args, **kwargs)
        if a1 > out.shape[0]:
            return out
        if method == "seg" and not legacy:
            _STATE["attn_hook_hits"] += 1
            _STATE["attn_hook_hits_total"] += 1
            return out

        # attention_function returns merged heads. Convert value to the same
        # pre-projection layout before blending the appended weak rows.
        value_path = value.reshape(value.shape[0], value.shape[1], -1)
        if value_path.shape != out.shape:
            if not _STATE["attn_shape_warned"]:
                _STATE["attn_shape_warned"] = True
                _log(
                    "self-attn value/output layout differ "
                    f"({tuple(value_path.shape)} vs {tuple(out.shape)}) "
                    "— skipping attention perturbation."
                )
            return out

        heads = sorted(
            _parse_attention_heads(
                str(_STATE.get("head_spec", "")),
                int(value.shape[-2]),
            )
        )
        if not heads:
            return out
        strength = float(_STATE["strength"])
        out_heads = out.reshape_as(value)
        result_heads = out_heads.clone()
        if method == "pag":
            target = value[a0:a1]
        elif method == "seg":
            # Uniform attention is the sequence-wise mean of values.
            target = value[a0:a1].mean(dim=1, keepdim=True).expand_as(value[a0:a1])
        else:
            return out
        weak = result_heads[a0:a1]
        head_idx = _head_index(heads, out_heads.device)
        weak[..., head_idx, :] = torch.lerp(
            out_heads[a0:a1, :, head_idx, :],
            target[..., head_idx, :],
            strength,
        )
        result_heads[a0:a1] = weak
        _STATE["attn_hook_hits"] += 1
        _STATE["attn_hook_hits_total"] += 1
        return result_heads.reshape_as(out)
    except Exception as e:  # never let the patch break sampling
        # 여기서는 기록만 한다. except 안에서 원본 op 를 다시 돌리면 살아 있는
        # 예외의 __traceback__ 이 실패한 호출의 중간 활성값을 붙잡은 채 두 번째
        # attention 이 돌아 VRAM 이 이중으로 든다.
        _log(f"attention perturb skipped: {type(e).__name__}: {e}")

    # except 블록 밖: 예외·traceback 이 풀린 뒤 반환/재시도한다(결과는 같다).
    if out is not None:
        return out
    query = None  # noqa: F841 - 교란된(SEG) query 를 재시도 전에 해제
    return original(original_query, key, value, *args, **kwargs)


_patched_anima_attention_op._anima_pag_owner = _PATCH_OWNER


def _make_selfattn_wrapper(idx: int, orig_forward):
    """Wrap ``self_attn.forward`` to raise ``active`` only for the attention
    target blocks, and only during the appended enlarged forward (``any_b0``)."""

    def _wrapped(*args, **kwargs):
        if _STATE["any_b0"] is not None and idx in _STATE["attn_targets"]:
            _STATE["active"] += 1
            try:
                return orig_forward(*args, **kwargs)
            finally:
                _STATE["active"] -= 1
        return orig_forward(*args, **kwargs)

    _wrapped._anima_pag_owner = _PATCH_OWNER
    return _wrapped


def _modulate_block_call(idx: int, args, kwargs):
    """Add the precomputed CLIP modulation to this block's AdaLN tensor.

    Forge passes ``adaln_lora_B_T_3D`` as a keyword today, while compatible
    Comfy-derived forks may pass it as positional argument 4. Return copied
    call containers so the shared base tensor remains untouched for the final
    layer and non-target blocks.
    """
    if not _MOD["on"] or idx not in _MOD["targets"] or torch is None:
        return args, kwargs

    positional = len(args) > 3
    adaln = (
        args[3]
        if positional
        else kwargs.get("adaln_lora_B_T_3D")
    )
    block_modulations = _MOD.get("block_modulations")
    if (
        not torch.is_tensor(adaln)
        or not torch.is_tensor(block_modulations)
        or idx >= block_modulations.shape[0]
    ):
        if not _MOD["warned"]:
            _MOD["warned"] = True
            _log(
                "CLIP modulation could not find a compatible "
                "adaln_lora_B_T_3D tensor; feature is inert."
            )
        return args, kwargs

    try:
        cache_key = (idx, str(adaln.device), str(adaln.dtype))
        delta = _MOD["typed"].get(cache_key)
        if delta is None:
            delta = block_modulations[idx].to(
                device=adaln.device,
                dtype=adaln.dtype,
            )
            _MOD["typed"][cache_key] = delta
        if delta.shape[-1] != adaln.shape[-1]:
            raise RuntimeError(
                f"AdaLN width mismatch: delta={delta.shape[-1]}, "
                f"model={adaln.shape[-1]}"
            )
        while delta.ndim < adaln.ndim:
            delta = delta.unsqueeze(0)
        modulated = adaln + delta

        if positional:
            changed = list(args)
            changed[3] = modulated
            call_args, call_kwargs = tuple(changed), kwargs
        else:
            changed = dict(kwargs)
            changed["adaln_lora_B_T_3D"] = modulated
            call_args, call_kwargs = args, changed

        _MOD["hits"] += 1
        return call_args, call_kwargs
    except Exception as exc:
        if not _MOD["warned"]:
            _MOD["warned"] = True
            _log(
                "CLIP modulation block injection fallback: "
                f"{type(exc).__name__}: {exc}"
            )
        return args, kwargs


# ---------------------------------------------------------------------------
# 앞쪽 블록 중복 제거(효율 보고서 2026-09-23 [2]).
#
# 확장 forward 에서 weak 행은 cond 행의 사본(x·timestep·조건 모두 index_select)이고,
# 첫 target 블록(PAG/SEG 의 attn_targets, SLG 의 slg_targets 중 가장 앞) 이전에는
# perturbation 이 없으므로 행마다 독립인 블록에서 weak 행 출력은 cond 행 출력과 같다.
# 그 구간에서는 원래 배치 행([:batch])만 원본 forward 로 돌리고 weak 자리에는 cond 행
# 출력을 복사한다. Modulation Guidance(adaln 에 행마다 같은 delta)·DAVE(행별 평균)·
# NegPiP(행별 마스크)·3.8B 커넥터(행별 run 마커로 DiT forward 에서 미리 확장한 조건)는
# 모두 행 단위라 이 전제를 깨지 않는다.
#
# 행 수를 전제로 하는 소비자와 충돌하지 않도록 블록 인자는 아는 것만 자른다.
# 모르는 kwargs, 행 수가 확장 배치도 1(브로드캐스트)도 아닌 텐서, transformer_options
# 안의 모르는 확장-배치 텐서가 있으면 그 블록부터 이번 forward 는 예전 경로(전체 배치)다.
# ---------------------------------------------------------------------------

# Forge Anima ``Block.forward`` 의 인자 이름(backend/nn/anima.py). Comfy 계열도 같다.
_DEDUP_BLOCK_KWARGS = frozenset({
    "x_B_T_H_W_D",
    "emb_B_T_D",
    "crossattn_emb",
    "rope_emb_L_1_1_D",
    "adaln_lora_B_T_3D",
    "extra_per_block_pos_emb",
    "transformer_options",
})
_DEDUP_MAX_POSITIONAL = 7
# transformer_options 안에서 확장 배치 행 수를 가진 텐서 중 잘라도 되는 것.
# negpip_mask: NegPiP 의 DiT forward 훅(또는 3.8B 런타임)이 c_negpip_mask 를 그대로
# 옮겨 둔 것. NegPiP cross-attn 은 x 행 수 // 마스크 행 수 로 반복하므로 x 와 같이
# 잘라야 한다. sigmas·cond_mark 는 Forge 가 원래 배치 행 수로 만들어 자를 것이 없다.
_DEDUP_OPTION_ROW_KEYS = frozenset({"negpip_mask"})
_DEDUP_UNSUPPORTED = object()


def _dedup_slice_tensor(value, batch: int, extended: int):
    if value is None:
        return value
    if not torch.is_tensor(value):
        return _DEDUP_UNSUPPORTED
    if value.ndim == 0 or value.shape[0] == 1:
        return value  # 스칼라·브로드캐스트(rope 등)
    if value.shape[0] == extended:
        return value[:batch]
    return _DEDUP_UNSUPPORTED


def _dedup_slice_options(options, batch: int, extended: int):
    if options is None:
        return options
    if not isinstance(options, dict):
        return _DEDUP_UNSUPPORTED
    changed = None
    for key, value in options.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == extended:
            if key not in _DEDUP_OPTION_ROW_KEYS:
                return _DEDUP_UNSUPPORTED
            if changed is None:
                changed = dict(options)  # 공유 딕셔너리는 건드리지 않는다
            changed[key] = value[:batch]
    return options if changed is None else changed


def _dedup_block_call(args, kwargs):
    """원래 배치 행으로 자른 (args, kwargs). 쓸 수 없으면 사유 문자열."""
    batch = _STATE["any_b0"]
    src = _STATE["dedup_src"]
    if batch is None or not torch.is_tensor(src) or src.ndim != 1:
        return "no layout"
    extended = int(batch) + int(src.shape[0])
    x = args[0] if args else kwargs.get("x_B_T_H_W_D")
    if not torch.is_tensor(x) or x.ndim == 0 or x.shape[0] != extended:
        return "x rows"
    if len(args) > _DEDUP_MAX_POSITIONAL:
        return f"{len(args)} positional args"
    new_args = []
    for position, value in enumerate(args):
        if isinstance(value, dict):
            sliced = _dedup_slice_options(value, batch, extended)
        else:
            sliced = _dedup_slice_tensor(value, batch, extended)
        if sliced is _DEDUP_UNSUPPORTED:
            return f"positional arg {position}"
        new_args.append(sliced)
    new_kwargs = {}
    for key, value in kwargs.items():
        if key not in _DEDUP_BLOCK_KWARGS:
            return f"unknown kwarg {key!r}"
        if key == "transformer_options":
            sliced = _dedup_slice_options(value, batch, extended)
        else:
            sliced = _dedup_slice_tensor(value, batch, extended)
        if sliced is _DEDUP_UNSUPPORTED:
            return f"kwarg {key!r}"
        new_kwargs[key] = sliced
    return tuple(new_args), new_kwargs


def _dedup_fallback(idx: int, reason: str) -> None:
    """이번 forward 는 이 블록부터 예전 경로(전체 배치)."""
    _STATE["dedup_until"] = None
    _STATE["dedup_fallbacks"] += 1
    if not _STATE["dedup_warned"]:
        _STATE["dedup_warned"] = True
        _log(
            f"앞쪽 블록 중복 제거를 블록 {idx} 부터 건너뜁니다(예전 전체 배치 경로): "
            f"{reason}"
        )


def _dedup_plan(idx: int, args, kwargs):
    """이 블록 호출을 원래 배치 행만으로 돌릴 수 있으면 잘린 (args, kwargs)."""
    until = _STATE.get("dedup_until")
    if until is None or _STATE["any_b0"] is None:
        return None
    if idx >= until or idx != _STATE["dedup_next"]:
        # 첫 target 도달, 또는 블록이 0 부터 연속으로 오지 않았다(감싸지 않은 블록·
        # 재호출 등) — weak 행이 cond 행과 같다는 전제를 더는 보장할 수 없다.
        _STATE["dedup_until"] = None
        return None
    plan = _dedup_block_call(args, kwargs)
    if isinstance(plan, str):
        _dedup_fallback(idx, plan)
        return None
    return plan


def _run_dedup_block(idx: int, orig_forward, plan):
    """원래 배치 행만 forward 하고 weak 자리에 cond 행 출력을 붙인다. 실패하면 None."""
    sliced_args, sliced_kwargs = plan
    batch = int(_STATE["any_b0"])
    src = _STATE["dedup_src"]
    failure = None
    try:
        out_b = orig_forward(*sliced_args, **sliced_kwargs)
    except Exception as exc:
        if _is_out_of_memory(exc):
            raise  # 확장 배치 OOM 경로(_model_wrapper_inner)가 처리한다
        # 여기서는 기록만 — 폴백 forward 는 예외·traceback 이 풀린 뒤에 돈다.
        failure = f"{type(exc).__name__}: {exc}"
    if failure is not None:
        _STATE["dedup_disabled"] = True  # 이 생성의 남은 스텝은 예전 경로
        _dedup_fallback(idx, failure)
        return None
    if not torch.is_tensor(out_b) or out_b.ndim == 0 or out_b.shape[0] != batch:
        _dedup_fallback(idx, "block output rows")
        return None
    out = torch.cat([out_b, out_b.index_select(0, src.to(out_b.device))], dim=0)
    _STATE["dedup_next"] = idx + 1
    _STATE["dedup_blocks"] += 1
    return out


def _make_block_wrapper(idx: int, orig_forward):
    """Compose CLIP modulation, original block, DAVE, then SLG restoration."""

    def _wrapped(*args, **kwargs):
        x_in = args[0] if args else kwargs.get("x_B_T_H_W_D", kwargs.get("x"))
        call_args, call_kwargs = _modulate_block_call(idx, args, kwargs)
        previous_spatial = _STATE.get("attn_spatial_shape")
        captures_spatial = (
            _STATE["any_b0"] is not None
            and idx in _STATE["attn_targets"]
            and torch.is_tensor(x_in)
            and x_in.ndim == 5
        )
        if captures_spatial:
            _STATE["attn_spatial_shape"] = tuple(int(v) for v in x_in.shape[1:4])
        dedup = _dedup_plan(idx, call_args, call_kwargs)
        try:
            out = None
            if dedup is not None:
                out = _run_dedup_block(idx, orig_forward, dedup)
                dedup = None  # 잘린 인자를 폴백 forward 동안 붙잡지 않는다
            if out is None:
                out = orig_forward(*call_args, **call_kwargs)
        finally:
            if captures_spatial:
                _STATE["attn_spatial_shape"] = previous_spatial
        try:
            dave_active = (
                _DAVE["on"]
                and idx in _DAVE["targets"]
                and (
                    float(_DAVE["tau"]) <= 0.0
                    or _pct_now() < float(_DAVE["tau"])
                )
            )
            if dave_active and torch.is_tensor(out):
                out = apply_dave(out, float(_DAVE["strength"]))
                _DAVE["steps"] += 1
        except Exception as e:
            _log(f"DAVE block {idx} fallback: {type(e).__name__}: {e}")
        try:
            s0, s1 = _STATE["slg_b0"], _STATE["slg_b1"]
            if s0 is not None and idx in _STATE["slg_targets"] and torch.is_tensor(out):
                if (torch.is_tensor(x_in) and x_in.shape == out.shape
                        and s1 <= out.shape[0]):
                    out = out.clone()
                    out[s0:s1] = x_in[s0:s1]
        except Exception as e:
            _log(f"slg skip skipped: {type(e).__name__}: {e}")
        return out

    _wrapped._anima_pag_owner = _PATCH_OWNER
    return _wrapped


def _ensure_patched(diffusion_model) -> int:
    """Install the active attention patch plus the per-block PAG/SLG wrappers.

    The actual self-attention instance determines the owner class. Looking up
    the descriptor through its MRO keeps this compatible with subclasses while
    preserving the core method's ``staticmethod`` semantics.
    """
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        _log("diffusion_model has no .blocks — unexpected Anima structure.")
        return 0

    first_self_attn = getattr(blocks[0], "self_attn", None)
    if first_self_attn is None:
        _log("Anima block has no self_attn — cannot perturb attention.")
        return 0

    owner_cls = descriptor = None
    for candidate in type(first_self_attn).__mro__:
        candidate_descriptor = candidate.__dict__.get("torch_attention_op")
        if candidate_descriptor is not None:
            owner_cls = candidate
            descriptor = candidate_descriptor
            break
    if owner_cls is None or descriptor is None:
        _log("self-attention class has no torch_attention_op — cannot perturb attention.")
        return 0

    current = (
        descriptor.__func__ if isinstance(descriptor, staticmethod)
        else descriptor
    )
    if not callable(current):
        _log("SelfCrossAttention.torch_attention_op is unavailable.")
        return 0

    global _ORIG_ANIMA_ATTN_OP, _PATCHED_ATTN_OWNER
    original = (
        getattr(owner_cls, "_pag_orig_torch_attention_op", None)
        or getattr(owner_cls, "_pag_orig_attention_op", None)
    )
    if (
        original is None
        and (
            getattr(current, "_anima_pag_owner", None) is not None
            or getattr(current, "__name__", "") == "_patched_anima_attention_op"
        )
    ):
        # Recover the core function from a wrapper left behind by WebUI's
        # "reload scripts" action, whose module globals remain reachable.
        original = getattr(current, "__globals__", {}).get("_ORIG_ANIMA_ATTN_OP")
    if original is None:
        original = current
    if not callable(original):
        _log("could not recover the original Anima attention op.")
        return 0

    _ORIG_ANIMA_ATTN_OP = original
    owner_cls._pag_orig_torch_attention_op = original
    owner_cls._pag_orig_attention_op = original  # migrate earlier sam-extra builds
    if getattr(current, "_anima_pag_owner", None) is not _PATCH_OWNER:
        owner_cls.torch_attention_op = staticmethod(_patched_anima_attention_op)
        owner_cls._pag_attention_op_patched = True
        _log(
            f"patched {owner_cls.__name__}.torch_attention_op "
            "(staticmethod) ✅"
        )
    _PATCHED_ATTN_OWNER = owner_cls

    wrapped_sa = wrapped_bl = 0
    for idx, block in enumerate(blocks):
        # SLG: wrap the whole block forward.
        if getattr(block.forward, "_anima_pag_owner", None) is not _PATCH_OWNER:
            try:
                if not hasattr(block, "_pag_orig_block_forward"):
                    block._pag_orig_block_forward = block.forward
                block.forward = _make_block_wrapper(idx, block._pag_orig_block_forward)
                block._pag_block_wrapped = True
                _PATCHED_BLOCKS.add(block)
                wrapped_bl += 1
            except Exception as e:
                _log(f"failed to wrap block {idx} forward: {type(e).__name__}: {e}")
        # PAG/SEG: wrap the self_attn forward.
        sa = getattr(block, "self_attn", None)
        if (sa is not None
                and getattr(sa.forward, "_anima_pag_owner", None) is not _PATCH_OWNER):
            try:
                if not hasattr(sa, "_pag_orig_forward"):
                    sa._pag_orig_forward = sa.forward
                sa.forward = _make_selfattn_wrapper(idx, sa._pag_orig_forward)
                sa._pag_wrapped = True
                _PATCHED_SELFATTN.add(sa)
                wrapped_sa += 1
            except Exception as e:
                _log(f"failed to wrap block {idx} self_attn: {type(e).__name__}: {e}")
    if wrapped_sa or wrapped_bl:
        _log(f"wrapped {wrapped_sa} self_attn + {wrapped_bl} block forward(s)")
    return len(blocks)


# ---------------------------------------------------------------------------
# CNS-inspired sampler-noise patch (global install, generation-gated fast path)
# ---------------------------------------------------------------------------


def _maybe_color_cns_noise(noise):
    if not _CNS["on"] or torch is None:
        return noise
    x_t = _RUNTIME.cns_x_t
    if not torch.is_tensor(x_t):
        return noise
    try:
        result = color_noise_wavelet(
            noise,
            x_t.to(device=noise.device, dtype=noise.dtype),
            strength=float(_CNS["strength"]),
            gamma_power=float(_CNS["gamma_power"]),
            gamma_scale=float(_CNS["gamma_scale"]),
        )
        _RUNTIME.cns_noise_calls += 1
        return result
    except Exception as e:
        if not _CNS["warned"]:
            _CNS["warned"] = True
            _log(f"CNS noise coloring fallback: {type(e).__name__}: {e}")
        return noise


def _patched_default_noise_sampler(x):
    factory = _ORIG_CNS_DEFAULT_FACTORY
    if not callable(factory):
        return lambda _sigma, _sigma_next: torch.randn_like(x)
    original_sampler = factory(x)
    if not _CNS["on"]:
        return original_sampler

    def _sample(sigma, sigma_next):
        return _maybe_color_cns_noise(original_sampler(sigma, sigma_next))

    return _sample


_patched_default_noise_sampler._anima_pag_owner = _PATCH_OWNER


def _patched_brownian_call(self, sigma, sigma_next):
    original = _ORIG_CNS_BROWNIAN_CALL
    if not callable(original):
        raise RuntimeError("original BrownianTreeNoiseSampler.__call__ unavailable")
    return _maybe_color_cns_noise(original(self, sigma, sigma_next))


_patched_brownian_call._anima_pag_owner = _PATCH_OWNER


def _ensure_cns_noise_patched() -> bool:
    """Patch both default and Brownian noise sources without editing Forge."""
    global _ORIG_CNS_DEFAULT_FACTORY, _ORIG_CNS_BROWNIAN_CALL
    modules_seen = set()
    installed = False
    found = False
    for module_name in (
        "k_diffusion.sampling",
        "modules_forge.packages.k_diffusion.sampling",
    ):
        try:
            sampling = importlib.import_module(module_name)
        except Exception:
            continue
        if id(sampling) in modules_seen:
            continue
        modules_seen.add(id(sampling))

        current_factory = getattr(sampling, "default_noise_sampler", None)
        if callable(current_factory):
            found = True
            original_factory = getattr(
                sampling, "_sam_extra_cns_orig_default_noise_sampler", None
            )
            if original_factory is None:
                if (
                    getattr(current_factory, "_anima_pag_owner", None) is not None
                    or getattr(current_factory, "__name__", "")
                    == "_patched_default_noise_sampler"
                ):
                    original_factory = getattr(
                        current_factory, "__globals__", {}
                    ).get("_ORIG_CNS_DEFAULT_FACTORY")
                else:
                    original_factory = current_factory
            if callable(original_factory):
                sampling._sam_extra_cns_orig_default_noise_sampler = original_factory
                _ORIG_CNS_DEFAULT_FACTORY = original_factory
                if (
                    getattr(current_factory, "_anima_pag_owner", None)
                    is not _PATCH_OWNER
                ):
                    sampling.default_noise_sampler = _patched_default_noise_sampler
                    installed = True

        brownian_cls = getattr(sampling, "BrownianTreeNoiseSampler", None)
        # NOTE: do not `continue` from here on a missing/non-callable Brownian
        # class. If this module already yielded default_noise_sampler
        # (found=True) we must still hit the `break` below — otherwise the loop
        # falls through to the second k-diffusion copy and patches *its*
        # default_noise_sampler too, leaving one module's slot pointing at our
        # wrapper while _ORIG_CNS_DEFAULT_FACTORY holds the *other* module's
        # original (cross-wired). Guard the Brownian work inline instead.
        if brownian_cls is not None:
            descriptor = brownian_cls.__dict__.get("__call__")
            current_call = descriptor
            if callable(current_call):
                found = True
                original_call = getattr(
                    brownian_cls, "_sam_extra_cns_orig_call", None
                )
                if original_call is None:
                    if (
                        getattr(current_call, "_anima_pag_owner", None) is not None
                        or getattr(current_call, "__name__", "")
                        == "_patched_brownian_call"
                    ):
                        original_call = getattr(
                            current_call, "__globals__", {}
                        ).get("_ORIG_CNS_BROWNIAN_CALL")
                    else:
                        original_call = current_call
                if callable(original_call):
                    brownian_cls._sam_extra_cns_orig_call = original_call
                    _ORIG_CNS_BROWNIAN_CALL = original_call
                    if (
                        getattr(current_call, "_anima_pag_owner", None)
                        is not _PATCH_OWNER
                    ):
                        brownian_cls.__call__ = _patched_brownian_call
                        installed = True
        if found:
            # Record the module we touched so _teardown_global_patches() can
            # restore it. Dedup by module identity.
            if not any(existing is sampling for existing, _ in _PATCHED_CNS_TARGETS):
                _PATCHED_CNS_TARGETS.append((sampling, brownian_cls))
            # The second name is only a fallback import path. Patching two
            # independently-loaded copies would make one global original
            # ambiguous across WebUI script reloads.
            break

    if installed:
        _log("patched k-diffusion default + Brownian noise sources for CNS ✅")
    elif not found:
        _log("CNS could not find k-diffusion noise sources; feature stays inert.")
    return found


def _teardown_global_patches() -> None:
    """Restore every global monkey-patch this script installed to its original.

    Covers the class-level Anima ``torch_attention_op``, the wrapped
    block/self_attn forwards, and the k-diffusion ``default_noise_sampler`` /
    ``BrownianTreeNoiseSampler.__call__``. Idempotent and safe to call when
    nothing was patched. Registered on script unload so a Forge "reload scripts"
    (or extension update) doesn't leave stale patches — and the sam-extra
    markers are cleared so a later reload re-patches from a clean base.
    """
    global _PATCHED_ATTN_OWNER

    owner = _PATCHED_ATTN_OWNER
    if owner is not None:
        try:
            original = getattr(owner, "_pag_orig_torch_attention_op", None)
            if callable(original):
                owner.torch_attention_op = staticmethod(original)
            for attr in (
                "_pag_attention_op_patched",
                "_pag_orig_torch_attention_op",
                "_pag_orig_attention_op",
            ):
                if hasattr(owner, attr):
                    try:
                        delattr(owner, attr)
                    except Exception:
                        pass
        except Exception:
            pass
        _PATCHED_ATTN_OWNER = None

    for block in list(_PATCHED_BLOCKS):
        try:
            original = getattr(block, "_pag_orig_block_forward", None)
            if original is not None:
                block.forward = original
            for attr in ("_pag_orig_block_forward", "_pag_block_wrapped"):
                if hasattr(block, attr):
                    delattr(block, attr)
        except Exception:
            pass
    _PATCHED_BLOCKS.clear()

    for sa in list(_PATCHED_SELFATTN):
        try:
            original = getattr(sa, "_pag_orig_forward", None)
            if original is not None:
                sa.forward = original
            for attr in ("_pag_orig_forward", "_pag_wrapped"):
                if hasattr(sa, attr):
                    delattr(sa, attr)
        except Exception:
            pass
    _PATCHED_SELFATTN.clear()

    for sampling, brownian_cls in _PATCHED_CNS_TARGETS:
        try:
            orig_factory = getattr(
                sampling, "_sam_extra_cns_orig_default_noise_sampler", None
            )
            if callable(orig_factory):
                sampling.default_noise_sampler = orig_factory
                delattr(sampling, "_sam_extra_cns_orig_default_noise_sampler")
        except Exception:
            pass
        if brownian_cls is not None:
            try:
                orig_call = getattr(brownian_cls, "_sam_extra_cns_orig_call", None)
                if callable(orig_call):
                    brownian_cls.__call__ = orig_call
                    delattr(brownian_cls, "_sam_extra_cns_orig_call")
            except Exception:
                pass
    _PATCHED_CNS_TARGETS.clear()
    try:
        clear_modulation_caches()
    except Exception:
        pass


try:
    script_callbacks.on_script_unloaded(_teardown_global_patches)
except Exception:
    # Older WebUI without this callback: patches simply persist as before.
    pass


# ---------------------------------------------------------------------------
# Conditioning-batch helpers
# ---------------------------------------------------------------------------


def _extend_c(c: dict, idx_tensor, batch: int) -> dict:
    """Append the pag (cond) rows to every batched tensor in the conditioning
    dict. Non-tensor entries (e.g. ``transformer_options``) pass through."""
    out = {}
    for k, v in c.items():
        if k == "transformer_options":
            out[k] = v
            continue
        if torch.is_tensor(v) and v.shape[0] == batch:
            out[k] = torch.cat([v, v.index_select(0, idx_tensor)], dim=0)
        else:
            out[k] = v
    return out


def _select_c(c: dict, idx_tensor, batch: int) -> dict:
    """Keep only the ``idx_tensor`` rows of every batched tensor (for the
    Adaptive-Guidance cond-only forward). Non-tensor entries pass through."""
    out = {}
    for k, v in c.items():
        if k == "transformer_options":
            out[k] = v
            continue
        if torch.is_tensor(v) and v.shape[0] == batch:
            out[k] = v.index_select(0, idx_tensor)
        else:
            out[k] = v
    return out


# cond/uncond 행 인덱스 텐서 캐시. ``torch.tensor(list, device=cuda)`` 는 스텝마다
# pageable 동기 H2D 였다. 값은 (행 목록, device)만으로 정해지므로 그 둘을 키로
# 두면 결과가 같다. 배치 배치(layout)는 몇 가지뿐이라 작게 두고 postprocess 에서 비운다.
_INDEX_TENSOR_CACHE: dict = {}
_INDEX_TENSOR_CACHE_MAX = 16


def _index_tensor(indices, device):
    key = (tuple(int(i) for i in indices), str(device))
    cached = _INDEX_TENSOR_CACHE.get(key)
    if cached is None:
        if len(_INDEX_TENSOR_CACHE) >= _INDEX_TENSOR_CACHE_MAX:
            _INDEX_TENSOR_CACHE.clear()
        cached = torch.tensor(key[0], device=device, dtype=torch.long)
        _INDEX_TENSOR_CACHE[key] = cached
    return cached


def _is_out_of_memory(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None) if torch is not None else None
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _owner_ref(p):
    """생성(p)을 붙잡지 않는 약한 참조. 지원하지 않는 객체면 None(이번 패스만)."""
    if p is None:
        return None
    try:
        return weakref.ref(p)
    except TypeError:
        return None


def _perturbation_oom_blocked(p) -> bool:
    """이 p 의 앞선 패스/배치에서 확장 배치가 OOM 으로 실패했는가."""
    ref = _STATE.get("pert_oom_owner")
    return ref is not None and p is not None and ref() is p


def _disable_perturbation_after_oom() -> None:
    """확장 배치 OOM 뒤: 이번 생성의 PAG/SEG/SLG 를 끄고 VRAM 을 돌려받는다.

    호출 시점에는 예외 객체와 traceback 이 이미 풀려 있어야 한다(except 블록 밖).
    그래야 실패한 forward 의 중간 활성값이 해제되어 폴백 forward 가 쓸 수 있다.
    """
    _STATE["on"] = False
    _STATE["pert_oom_owner"] = _STATE.get("pass_owner")
    try:
        gc.collect()  # traceback 순환 참조에 남은 프레임까지 정리
    except Exception:
        pass
    try:
        from backend import memory_management

        memory_management.soft_empty_cache()
    except Exception:
        pass


def _sampling_position() -> tuple[int, int]:
    """Return Forge's authoritative 0-based sampler position.

    A model wrapper may run multiple times for one denoise step (regional
    conditioning, low-VRAM splits, or a second-order sampler). Counting wrapper
    calls therefore corrupts all percent gates. ``shared.state`` is maintained
    by Forge's sampler callback and remains stable across those extra calls.
    The dict values are retained only as a standalone-test fallback.
    """
    try:
        state = getattr(shared, "state", None)
        step = int(getattr(state, "sampling_step"))
        total = int(getattr(state, "sampling_steps"))
        if total > 0:
            return max(0, step), total
    except Exception:
        pass
    return max(0, int(_STATE["step"])), max(1, int(_STATE["total"]))


def _pct_now() -> float:
    step, total = _sampling_position()
    return min(1.0, max(0.0, step / max(total - 1, 1)))


def _percent_in_range() -> bool:
    pct = _pct_now()
    return _STATE["start"] <= pct <= _STATE["end"]


def _adg_should_skip() -> bool:
    """Adaptive Guidance: skip the uncond forward this step?"""
    if not _ADG["on"]:
        return False
    if _pct_now() < float(_ADG["start"]):
        return False
    iv = int(_ADG["interval"])
    step, _ = _sampling_position()
    if iv > 0 and (step % iv == 0):
        return False  # periodically keep an uncond forward for safety
    return True


def reset_cfg_state() -> None:
    """Clear all stateful CFG transforms at a pass boundary or ADG skip."""
    _RUNTIME.reset_cfg_state()


# ---------------------------------------------------------------------------
# The two Forge-honoured hooks
# ---------------------------------------------------------------------------


_AGGREGATION_KEY = "_sam_extra_condition_aggregation"


class _ConditionAggregation:
    """Opaque conditioning metadata consumed before calling the model.

    Implements Forge's Condition concatenation protocol so each queued
    condition retains its original area/mult even when VRAM changes grouping.
    It never reaches diffusion_model.forward or changes its real conditioning.
    """
    def __init__(self, aggregate, prepared, positive):
        self.aggregate = aggregate
        self.prepared = prepared
        self.positive = positive

    def process_cond(self, **_kwargs):
        return self

    def can_concat(self, _other):
        return True

    def concat(self, others):
        return [self, *others]


def _prepare_condition_aggregation(model, cond, uncond, x, timestep, model_options):
    """Keep the exact Forge weights/regions for compositional weak predictions."""
    _STATE.pop("condition_aggregation", None)
    if not _STATE["on"] or not _percent_in_range() or not cond:
        return model, cond, uncond, x, timestep, model_options
    if len(cond) == 1 and not any(key in cond[0] for key in ("area", "mask")):
        return model, cond, uncond, x, timestep, model_options
    from backend.sampling.sampling_function import get_area_and_mult

    aggregate = {
        "cond": torch.zeros_like(x), "attn": torch.zeros_like(x),
        "slg": torch.zeros_like(x), "count": torch.ones_like(x) * 1e-37,
        "attn_seen": False, "slg_seen": False,
    }

    def attach(items, positive):
        if items is None:
            return None
        output = []
        for item in items:
            # Delegate mask strength, crop feathering and timestep filtering
            # to Forge itself; do not approximate them with a prompt average.
            prepared = get_area_and_mult(
                {**item, "model_conds": {}}, x, timestep,
            ) if positive else None
            metadata = None if prepared is None else (prepared.area, prepared.mult)
            changed = dict(item)
            changed["model_conds"] = {
                **item["model_conds"],
                _AGGREGATION_KEY: _ConditionAggregation(aggregate, metadata, positive),
            }
            output.append(changed)
        return output

    _STATE["condition_aggregation"] = aggregate
    return model, attach(cond, True), attach(uncond, False), x, timestep, model_options


def _accumulate_condition_predictions(metadata, output, attn, slg):
    chunks = len(metadata)
    batch = output.shape[0] // chunks
    cond_offset = 0
    for index, entry in enumerate(metadata):
        if not entry.positive:
            continue
        current = output[index * batch:(index + 1) * batch]
        weak_slice = slice(cond_offset, cond_offset + batch)
        cond_offset += batch
        if entry.prepared is None:
            continue
        area, mult = entry.prepared
        target = (..., slice(area[2], area[2] + area[0]), slice(area[3], area[3] + area[1]))
        aggregate = entry.aggregate
        aggregate["cond"][target] += current * mult
        aggregate["count"][target] += mult
        # A condition where ControlNet/another guard blocks perturbation
        # contributes an identity weak result, so only its own delta is zero.
        for key, weak in (("attn", attn), ("slg", slg)):
            prediction = current if weak is None else weak[weak_slice]
            aggregate[key][target] += prediction * mult
            aggregate[key + "_seen"] |= weak is not None


def _clear_markers():
    _STATE["any_b0"] = None
    _STATE["attn_b0"] = _STATE["attn_b1"] = None
    _STATE["slg_b0"] = _STATE["slg_b1"] = None
    _STATE["attn_spatial_shape"] = None
    _STATE["dedup_until"] = None
    _STATE["dedup_src"] = None
    _STATE["dedup_next"] = 0


def _model_wrapper(apply_model, w):
    metadata = (w.get("c") or {}).get(_AGGREGATION_KEY)
    if metadata is None:
        return _model_wrapper_inner(apply_model, w)
    changed = dict(w)
    changed["c"] = dict(w.get("c") or {})
    changed["c"].pop(_AGGREGATION_KEY, None)
    # A microbatch may have no active weak rows. Never reuse the preceding
    # microbatch's cache when folding its output into the full image.
    _STATE["attn_raw"] = _STATE["slg_raw"] = None
    output = _model_wrapper_inner(apply_model, changed)
    _accumulate_condition_predictions(
        metadata, output, _STATE["attn_raw"], _STATE["slg_raw"],
    )
    return output


def _model_wrapper_inner(apply_model, w):
    """model_function_wrapper: run an enlarged cond ``apply_model`` that also
    produces the weak predictions (attention-perturbed for PAG/SEG and/or
    layer-skipped for SLG) by appending copies of the cond rows. This supports
    both Forge's combined cond+uncond call and its low-VRAM split calls. No
    separate forward — every weak prediction rides the cond kernel launch.
    """
    x = w.get("input")
    ts = w.get("timestep")
    c = w.get("c") or {}
    cou = w.get("cond_or_uncond")

    if not (_STATE["on"] or _ADG["on"]) or torch is None:
        return apply_model(x, ts, **c)

    extended_forward = False  # OOM 이 weak 행을 붙인 확장 forward 에서 났는가
    try:
        _STATE["wrapper_calls"] += 1
        if cou is None or len(cou) == 0:
            return apply_model(x, ts, **c)

        batch = x.shape[0]
        if batch % len(cou) != 0:
            return apply_model(x, ts, **c)

        # Forge may call the wrapper once with [cond, uncond], or separately
        # with [uncond] then [cond] when the combined Anima batch does not fit
        # in VRAM. Treat all wrapper calls before post_cfg as one denoise step,
        # but never derive the step number from wrapper-call counts.
        if not _STATE["step_open"]:
            _STATE["step_open"] = True
            _STATE["step"] = _sampling_position()[0]
            _STATE["attn_raw"] = None
            _STATE["slg_raw"] = None
            _STATE["cond_raw"] = None
            _STATE["adg_skipped"] = False

        chunk = batch // len(cou)
        cond_idx, uncond_idx = [], []
        for i, marker in enumerate(cou):
            rng = list(range(i * chunk, (i + 1) * chunk))
            (cond_idx if int(marker) == 0 else uncond_idx).extend(rng)

        if cond_idx and uncond_idx:
            _STATE["combined_calls"] += 1
        elif cond_idx:
            _STATE["split_cond_calls"] += 1
        elif uncond_idx:
            _STATE["split_uncond_calls"] += 1

        idx = _index_tensor(cond_idx, x.device) if cond_idx else None
        uidx = _index_tensor(uncond_idx, x.device) if uncond_idx else None

        # A split uncond-only call cannot carry a perturbation copy. Forge's
        # post-CFG args already supply the aggregated denoised predictions, so
        # there is nothing else to cache from this half.
        if idx is None:
            return apply_model(x, ts, **c)

        n = len(cond_idx)

        # --- Adaptive Guidance: in the late steps, run cond-only and set the
        # uncond output = cond → CFG collapses to cond (guidance neutralized),
        # saving the uncond half of the batch. Takes precedence over the
        # (adds-work) perturbation path in those steps. ---
        if (_adg_should_skip() and uidx is not None
                and len(cond_idx) == len(uncond_idx)):
            x_c = x.index_select(0, idx)
            ts_c = ts.index_select(0, idx)
            c_c = _select_c(c, idx, batch)
            out_c = apply_model(x_c, ts_c, **c_c)
            _STATE["adg_skipped_steps"] += 1
            _STATE["adg_skipped"] = True
            # APG momentum from a preceding guided step must not leak through
            # a cond-only interval or into the next periodically-kept step.
            reset_cfg_state()
            out_full = torch.empty(
                (batch,) + tuple(out_c.shape[1:]),
                device=out_c.device, dtype=out_c.dtype,
            )
            out_full.index_copy_(0, idx, out_c)
            out_full.index_copy_(0, uidx, out_c)  # uncond := cond
            return out_full

        if (_STATE["on"] and _percent_in_range()
                and c.get("control") is not None):
            _STATE["control_blocked_calls"] += 1
        if not _STATE["on"] or not _percent_in_range() or c.get("control") is not None:
            return apply_model(x, ts, **c)

        attn_on = bool(_STATE["attn_method"]) and float(_STATE["attn_scale"]) > 0 \
            and bool(_STATE["attn_targets"])
        slg_on = bool(_STATE["slg_on"]) and float(_STATE["slg_scale"]) > 0 \
            and bool(_STATE["slg_targets"])
        if not attn_on and not slg_on:
            return apply_model(x, ts, **c)

        # Lay out the appended weak-prediction rows.
        cursor = batch
        appended = []
        a0 = a1 = s0 = s1 = None
        if attn_on:
            a0, a1 = cursor, cursor + n
            appended.append(idx); cursor += n
        if slg_on:
            s0, s1 = cursor, cursor + n
            appended.append(idx); cursor += n

        app_idx = torch.cat(appended) if len(appended) > 1 else appended[0]
        x_ext = torch.cat([x, x.index_select(0, app_idx)], dim=0)
        ts_ext = torch.cat([ts, ts.index_select(0, app_idx)], dim=0)
        c_ext = _extend_c(c, app_idx, batch)

        _STATE["attn_b0"], _STATE["attn_b1"] = a0, a1
        _STATE["slg_b0"], _STATE["slg_b1"] = s0, s1
        _STATE["any_b0"] = batch
        # 앞쪽 블록 중복 제거: 첫 target 블록 이전은 원래 배치 행만 돌린다.
        first_target = min(
            (_STATE["attn_targets"] if a0 is not None else set())
            | (_STATE["slg_targets"] if s0 is not None else set()),
            default=0,
        )
        _STATE["dedup_src"] = app_idx
        _STATE["dedup_next"] = 0
        _STATE["dedup_until"] = (
            int(first_target)
            if (
                _STATE.get("prefix_dedup")
                and not _STATE.get("dedup_disabled")
                and int(first_target) > 0
            )
            else None
        )
        if a0 is not None:
            _STATE["attn_hook_hits"] = 0
        extended_forward = True
        try:
            out_ext = apply_model(x_ext, ts_ext, **c_ext)
        finally:
            _clear_markers()
        extended_forward = False

        out = out_ext[:batch]
        # The cond rows the weak predictions were derived from. post_cfg must
        # difference the weak prediction against THIS tensor, not against
        # args["cond_denoised"]: another post-CFG hook (our own Skimmed CFG
        # included) may legitimately rewrite Forge's prediction tensors in
        # place before we run, and a rewritten cond leaks a constant offset
        # into scale·(cond − weak), diluting the PAG/SEG scale and strength.
        cond_rows = out.index_select(0, idx).detach().float()
        _STATE["cond_raw"] = cond_rows
        if a0 is not None:
            _STATE["attn_raw"] = out_ext[a0:a1].detach().float()
        if s0 is not None:
            _STATE["slg_raw"] = out_ext[s0:s1].detach().float()
        _STATE["weak_steps"] += 1

        # rel_delta 는 진단값이다(첫 스텝 로그 + 진단 켰을 때 [VERIFY] 요약).
        # float() 가 forward 전체를 기다리는 동기화라 그 둘일 때만 잰다.
        if a0 is not None and (
            not _STATE["attn_diag_logged"] or guidance_diagnostics_enabled()
        ):
            weak = _STATE["attn_raw"]
            cond = cond_rows
            rel_delta = None
            if weak is not None and weak.shape == cond.shape:
                denom = torch.linalg.vector_norm(cond).clamp_min(1e-8)
                rel_delta = float(torch.linalg.vector_norm(weak - cond) / denom)
            _STATE["attn_last_rel_delta"] = rel_delta

            if not _STATE["attn_diag_logged"]:
                hits = int(_STATE["attn_hook_hits"])
                if hits <= 0:
                    _log(
                        "PAG/SEG enabled but the self-attention hook was not "
                        "reached — weak prediction is unperturbed."
                    )
                elif rel_delta is None or rel_delta <= 1e-8:
                    _log(
                        f"attention hook reached ({hits} hit(s)) but weak/cond "
                        f"raw delta is neutral (relative={rel_delta})."
                    )
                else:
                    _log(
                        f"attention perturb active ✅ hits={hits} "
                        f"relative_raw_delta={rel_delta:.3e}"
                    )
                _STATE["attn_diag_logged"] = True
        return out
    except Exception as e:
        # 여기서는 기록만 한다. except 안에서 폴백 forward 를 돌리면 살아 있는
        # 예외의 __traceback__ 이 실패한 forward 프레임(중간 활성값)을 붙잡은
        # 채로 두 번째 forward 가 돌아 VRAM 이 이중으로 든다.
        failure = (
            type(e).__name__,
            str(e),
            extended_forward and _is_out_of_memory(e),
        )

    # except 블록 밖: 예외·traceback 이 풀린 뒤 폴백한다.
    name, message, oom = failure
    _STATE["wrapper_fallbacks"] += 1
    _clear_markers()
    _STATE["attn_raw"] = None
    _STATE["slg_raw"] = None
    _STATE["cond_raw"] = None
    x_ext = ts_ext = c_ext = out_ext = out = None  # noqa: F841 - 확장 배치 해제
    if oom and _STATE["on"]:
        _disable_perturbation_after_oom()
        _log(
            f"wrapper fallback → normal apply_model: {name}: {message} "
            "— 확장 배치(PAG/SEG/SLG weak 행)가 VRAM 부족으로 실패해 "
            "캐시를 비우고 이 생성의 perturbation 을 끕니다."
        )
    else:
        _log(f"wrapper fallback → normal apply_model: {name}: {message}")
    return apply_model(x, ts, **c)


# Owner tag so a co-loaded script (e.g. anima_ref_poc) can recognise our unet
# function wrapper in ``model_options`` and step aside instead of silently
# clobbering it — the single wrapper slot can't hold two batch-manipulating
# wrappers at once.
_model_wrapper._anima_pag_owner = _PATCH_OWNER


def _existing_unet_wrapper(unet):
    """Return the unet ``model_function_wrapper`` already installed on ``unet``
    (its clone carries the model_options of whatever ran earlier), or None."""
    try:
        return (getattr(unet, "model_options", None) or {}).get(
            "model_function_wrapper"
        )
    except Exception:
        return None


def _warn_foreign_unet_wrapper(unet) -> None:
    existing = _existing_unet_wrapper(unet)
    if (
        existing is not None
        and getattr(existing, "_anima_pag_owner", None) is not _PATCH_OWNER
    ):
        _log(
            "another extension already installed a unet model_function_wrapper "
            f"({getattr(existing, '__qualname__', existing)!r}); the Anima "
            "Guidance suite needs exclusive control of the cond/uncond batch, "
            "so it takes precedence and overrides that wrapper this generation."
        )


def _project(v0, v1):
    """Decompose ``v0`` into (parallel, orthogonal) components relative to the
    direction of ``v1``. Per-sample (reduce over every dim except batch)."""
    dims = list(range(1, v1.ndim))
    v1n = torch.nn.functional.normalize(v1, dim=dims)
    parallel = (v0 * v1n).sum(dim=dims, keepdim=True) * v1n
    orthogonal = v0 - parallel
    return parallel, orthogonal


def _apply_apg(args, effective_scale, guidance_override=None):
    """Adaptive Projected Guidance — returns a replacement CFG-combine result
    in denoised space. With eta=1, norm_threshold=0, momentum=0 this is exactly
    standard CFG. Falls back to None (→ caller keeps the original denoised) on
    any problem."""
    try:
        cond = args["cond_denoised"].float()
        uncond = args["uncond_denoised"].float()
        guidance = (
            guidance_override.float()
            if guidance_override is not None
            else cond - uncond
        )

        # Momentum: running average across steps; reset on a new pass (sigma up).
        mom = float(_APG["momentum"])
        if mom != 0.0:
            sigma = args.get("sigma")
            cur = float(sigma.flatten()[0]) if sigma is not None else None
            last = _APG["last_sigma"]
            avg = _APG["avg"]
            if (avg is None or avg.shape != guidance.shape
                    or (cur is not None and last is not None and cur > last + 1e-6)):
                avg = torch.zeros_like(guidance)
            avg = mom * avg + guidance
            _APG["avg"] = avg
            _APG["last_sigma"] = cur
            guidance = avg

        # Norm clamp: keep the guidance vector under a fixed L2 magnitude.
        nt = float(_APG["norm_threshold"])
        if nt > 0.0:
            dims = list(range(1, guidance.ndim))
            gnorm = guidance.norm(p=2, dim=dims, keepdim=True)
            scale = torch.clamp(nt / gnorm.clamp_min(1e-8), max=1.0)
            guidance = guidance * scale

        # Project onto cond, downweight the parallel (saturating) component.
        parallel, orthogonal = _project(guidance, cond)
        modified = orthogonal + float(_APG["eta"]) * parallel

        # uncond + w*modified  →  standard CFG when modified == (cond-uncond).
        return uncond + float(effective_scale) * modified
    except Exception as e:
        _log(f"APG skipped: {type(e).__name__}: {e}")
        return None


def _recover_effective_cfg(args, incoming, with_fit_error=True):
    """Fit the incoming CFG result to ``uncond + w_eff*(cond-uncond)``.

    Forge does not expose per-conditioning ``edit_strength`` in post-CFG args.
    Least-squares recovery retains it for linear CFG and quantifies how poorly
    a nonlinear/custom CFG result fits before any explicit base override.
    ``with_fit_error=False`` skips that diagnostic (two norms and a second
    GPU->CPU sync) and returns ``fit_error=None``; ``effective`` is unchanged.
    """
    cond = args["cond_denoised"].float()
    uncond = args["uncond_denoised"].float()
    guidance = cond - uncond
    residual = incoming.float() - uncond
    denom = (guidance * guidance).sum().clamp_min(1e-12)
    effective = float(((residual * guidance).sum() / denom).item())
    if not with_fit_error:
        return effective, None
    fitted = uncond + effective * guidance
    fit_error = float(
        (
            torch.linalg.vector_norm(incoming.float() - fitted)
            / torch.linalg.vector_norm(residual).clamp_min(1e-8)
        ).item()
    )
    return effective, fit_error


def _cfg_base_flags() -> tuple[bool, bool, bool]:
    """Resolve ``(smc, apg, cwm)`` from the independent toggles.

    The pre-split mutually-exclusive ``mode`` radio and the experimental stack
    checkbox are folded in with OR, so an old preset selecting ``SMC + CWM``
    behaves exactly like ticking both new toggles."""
    mode = str(_CFG["mode"])
    stacked = bool(_CFG["experimental_stack"])
    smc_on = bool(_CFG["smc_on"]) or stacked or mode in {"smc", "smc+cwm"}
    apg_on = bool(_CFG["apg_on"]) or stacked or mode == "apg"
    cwm_on = bool(_CFG["cwm_on"]) or stacked or mode in {"cwm", "smc+cwm"}
    return smc_on, apg_on, cwm_on


def _cfg_base_skip_reason(args) -> str | None:
    """Why the CFG base override must not run on this post-CFG call.

    At ``cond_scale`` ≈ 1 Forge never evaluates the uncond batch
    (``sampling_function_inner``: ``uncond_ = None``) and hands the hook an
    all-zero ``uncond_denoised``. SMC/APG/CWM all rewrite ``cond - uncond``,
    which then equals ``cond`` itself: APG with eta=0 projects most of the
    prediction away (near-black output) and CWM/SMC reweight the image
    instead of a guidance error. Skimmed CFG guards the same case. Either
    signal alone is enough — ``cond_scale`` covers
    ``disable_cfg1_optimization``, the zero check covers callers that omit
    the key. Returns None when the override may proceed.

    Forge always passes ``cond_scale`` (``sampling_function_inner``) and only
    drops the uncond batch when it is ≈1, so with a usable ``cond_scale`` the
    zero scan — a full-tensor ``torch.any`` read back to the host, i.e. a
    GPU->CPU sync every step — is redundant and runs only as a fallback."""
    cond_scale = args.get("cond_scale")
    if cond_scale is not None:
        try:
            scale = float(cond_scale)
        except (TypeError, ValueError):
            scale = None
        if scale is not None:
            if math.isclose(scale, 1.0):
                return f"cond_scale={scale:g}"
            return None
    uncond = args.get("uncond_denoised")
    if (
        torch.is_tensor(uncond)
        and uncond.numel() > 0
        and not bool(torch.any(uncond != 0))
    ):
        return "uncond_denoised is all zero (no uncond pass; cond_scale≈1)"
    return None


def _cfg1_skip_message(args, skip_reason: str) -> str:
    """CFG 1 가드 경고 한 줄(생성당 1회).

    ``disable_cfg1_optimization`` 이 켜져 있으면 Forge 는 CFG 1 에서도 uncond
    패스를 돌린다. 그때 건너뛰는 이유는 uncond 부재가 아니라 배율 1 이다
    (CFG 결과가 cond 예측 그 자체라 다듬을 가이던스가 없다)."""
    model_options = args.get("model_options") or {}
    forced_uncond = bool(
        isinstance(model_options, dict)
        and model_options.get("disable_cfg1_optimization", False)
        and skip_reason.startswith("cond_scale")
    )
    if forced_uncond:
        why = (
            "disable_cfg1_optimization is set, so Forge still ran the uncond "
            "pass, but at CFG 1 the incoming result is the cond prediction "
            "itself and there is no CFG guidance to smooth, reproject or "
            "reweight"
        )
    else:
        why = (
            "Forge runs no uncond at CFG 1, so there is no CFG error to "
            "smooth, reproject or reweight"
        )
    message = (
        f"CFG base override (SMC/APG/CWM) skipped: {skip_reason}. {why}; "
        "the incoming result is kept unchanged (set CFG > 1 to use these "
        "bases)."
    )
    if (
        _APG["on"]
        and _STATE.get("apg_autooff_rescale", True)
        and _STATE.get("on")
        and float(_STATE.get("rescale", 0.0) or 0.0) > 0
    ):
        message += (
            " APG did not run, so the PAG rescale auto-off is not applied "
            "either (rescale stays active)."
        )
    return message


def _apply_cfg_base(args, incoming):
    """Return the selected CFG base before PAG/SEG/SLG and DCW.

    SMC, APG and CWM are independent; whichever are on run in the fixed order
    SMC (smooth the CFG error across steps) -> APG (reproject it) -> CWM
    (reweight it per Haar band). With none on, the incoming CFG result from
    Forge/MaHiRo/other extensions is preserved untouched."""
    _CFG["base_skipped"] = False
    smc_on, apg_on, cwm_on = _cfg_base_flags()
    if not (smc_on or apg_on or cwm_on):
        return incoming.float()

    skip_reason = _cfg_base_skip_reason(args)
    if skip_reason is not None:
        _CFG["base_skipped"] = True
        if not _CFG["cfg1_warned"]:
            _CFG["cfg1_warned"] = True
            _log(_cfg1_skip_message(args, skip_reason))
        return incoming.float()

    # fit_error 는 진단값(경고 한 줄 + [VERIFY] 요약)이라 결과에 쓰이지 않는다.
    # 진단이 꺼져 있으면 패스의 첫 평가에서만 재서 비선형 CFG 경고를 살리고,
    # 이후 스텝의 두 번째 .item() 동기화를 건너뛴다. effective_scale 은 늘 잰다.
    measure_fit = guidance_diagnostics_enabled() or not _CFG["fit_checked"]
    effective_scale, fit_error = _recover_effective_cfg(
        args, incoming, with_fit_error=measure_fit
    )
    _CFG["effective_scale"] = effective_scale
    if measure_fit:
        _CFG["fit_checked"] = True
        _CFG["fit_error"] = fit_error
    model_options = args.get("model_options") or {}
    external_cfg = "sampler_cfg_function" in model_options
    _CFG["external_cfg_detected"] = bool(external_cfg)
    nonlinear = fit_error is not None and fit_error > 0.05
    if not _CFG["warned"] and (external_cfg or nonlinear):
        _CFG["warned"] = True
        fit_text = "?" if fit_error is None else f"{fit_error:.3e}"
        _log(
            "CFG base override requested while incoming CFG is custom/nonlinear "
            f"(sampler_cfg_function={external_cfg}, fit_error={fit_text}); "
            "the selected base intentionally replaces it."
        )

    cond = args["cond_denoised"].float()
    uncond = args["uncond_denoised"].float()
    sigma = args.get("sigma")

    if apg_on:
        # SMC -> APG -> CWM. APG consumes the (optionally SMC-smoothed) error
        # and already applies the CFG scale, so CWM runs on its output at 1.0.
        raw_error = cond - uncond
        if smc_on:
            raw_error, _RUNTIME.smc_prev = apply_smc_error(
                raw_error,
                _RUNTIME.smc_prev,
                float(_CFG["smc_lambda"]),
                float(_CFG["smc_k"]),
            )
        apg_result = _apply_apg(
            args, effective_scale, raw_error if smc_on else None
        )
        if apg_result is None:
            if not (smc_on or cwm_on):
                return incoming.float()  # APG alone failed: keep incoming CFG
            apg_result = uncond + effective_scale * raw_error
        else:
            _STATE["apg_steps"] += 1
        _CFG["steps"] += 1
        if not cwm_on:
            return apg_result
        return uncond + apply_cwm_error(
            apg_result - uncond,
            sigma,
            1.0,
            float(_CFG["alpha_low"]),
            float(_CFG["alpha_high"]),
        )

    # No APG: compose_cfg covers SMC only, CWM only, and SMC + CWM.
    result, next_previous = compose_cfg(
        cond=cond,
        uncond=uncond,
        sigma=sigma,
        effective_scale=effective_scale,
        mode="smc+cwm" if smc_on and cwm_on else ("smc" if smc_on else "cwm"),
        alpha_low=float(_CFG["alpha_low"]),
        alpha_high=float(_CFG["alpha_high"]),
        smc_lambda=float(_CFG["smc_lambda"]),
        smc_k=float(_CFG["smc_k"]),
        smc_previous=_RUNTIME.smc_prev,
    )
    if smc_on:
        _RUNTIME.smc_prev = next_previous
    _CFG["steps"] += 1
    return result


def _apply_perturbation(args, base):
    """Add the perturbation-guidance term(s) onto ``base`` (denoised space):
    ``scale·(cond_denoised − weak_denoised)`` for each active weak prediction
    (attention-perturbed for PAG/SEG, and/or layer-skipped for SLG). Forge's
    ``model.apply_model`` has already converted every prediction to denoised
    x0, so no eps/v/flow conversion is needed here. Each active term applies at
    its full configured scale (the former auto-decay safety brake, which halved
    each scale when >1 term was active, has been removed).
    Returns ``base`` unchanged on any problem."""
    cd = args["cond_denoised"].float()
    # Prefer the cond captured next to the weak predictions. Any post-CFG hook
    # that ran before us may have rewritten args["cond_denoised"] in place, and
    # differencing a rewritten cond against an un-rewritten weak injects a
    # constant offset that dilutes both the scale and the strength control.
    cond_raw = _STATE.get("cond_raw")
    if torch.is_tensor(cond_raw) and cond_raw.shape == cd.shape:
        cd = cond_raw.float()
    attn_raw = _STATE["attn_raw"]
    slg_raw = _STATE["slg_raw"]

    terms = []
    if attn_raw is not None and attn_raw.shape == cd.shape:
        terms.append((float(_STATE["attn_scale"]), attn_raw))
    if slg_raw is not None and slg_raw.shape == cd.shape:
        terms.append((float(_STATE["slg_scale"]), slg_raw))
    if not terms:
        return base

    guidance = torch.zeros_like(base, dtype=torch.float32)
    for scale, weak in terms:
        guidance = guidance + scale * (cd - weak)

    if not _STATE["delta_logged"]:
        _STATE["delta_logged"] = True
        delta = float((cd - terms[0][1]).detach().abs().mean())
        _log(
            "first active-step mean |cond - weak|="
            f"{delta:.8f}; terms={len(terms)}"
        )

    # Match the original Safe-PAG rescale. ``full`` derives the factor from the
    # incoming CFG result plus PAG/SEG guidance; ``partial`` uses the conditional
    # prediction plus guidance. In both modes only the new guidance term is
    # scaled. Scaling the entire CFG base every denoise step drains image energy.
    r = float(_STATE["rescale"])
    # CFG 1 가드로 APG 를 건너뛴 스텝에서는 자동 끄기도 적용하지 않는다.
    apg_governs = (
        _APG["on"]
        and _STATE.get("apg_autooff_rescale", True)
        and not _CFG.get("base_skipped", False)
    )
    if r > 0 and not apg_governs:
        guided = (
            cd + guidance
            if _STATE.get("rescale_mode", "full") == "partial"
            else base.float() + guidance
        )
        dims = list(range(1, guided.ndim))
        std_c = cd.std(dim=dims, keepdim=True).clamp_min(1e-6)
        std_r = guided.std(dim=dims, keepdim=True).clamp_min(1e-6)
        factor = r * (std_c / std_r) + (1.0 - r)
        guidance = guidance * factor

    result = base.float() + guidance
    _STATE["applied_steps"] += 1
    return result


def _post_cfg(args):
    """Single post-CFG orchestrator.

    Order is fixed and visible: capture live x_t for CNS; handle ADG state;
    select one CFG base (or the explicit experimental stack); add perturbation
    deltas; run DCW last inside sam-extra.
    """
    denoised = args["denoised"]
    aggregate = _STATE.pop("condition_aggregation", None)
    if aggregate is not None:
        count = aggregate["count"]
        _STATE["cond_raw"] = aggregate["cond"] / count
        for key in ("attn", "slg"):
            _STATE[key + "_raw"] = (
                aggregate[key] / count if aggregate[key + "_seen"] else None
            )
    if torch is None:
        return denoised
    live_input = args.get("input")
    if torch.is_tensor(live_input):
        _RUNTIME.cns_x_t = live_input.detach()

    if _STATE["adg_skipped"]:
        reset_cfg_state()
        _RUNTIME.close_step()
        return denoised

    has_base_override = any(_cfg_base_flags())
    has_pert = _STATE["on"] and (
        _STATE["attn_raw"] is not None or _STATE["slg_raw"] is not None
    )
    if not has_base_override and not has_pert and not _DCW["on"]:
        _RUNTIME.close_step()
        return denoised

    try:
        result = _apply_cfg_base(args, denoised)

        if has_pert:
            result = _apply_perturbation(args, result)

        if _DCW["on"]:
            try:
                result = apply_dcw(
                    result,
                    live_input,
                    args.get("sigma"),
                    (
                        float(_DCW["lambda_low"])
                        if _DCW["dcw_on"] else 0.0
                    ),
                    (
                        float(_DCW["lambda_high"])
                        if _DCW["dcw_on"] else 0.0
                    ),
                    rdc_tau=(
                        float(_DCW["rdc_tau"])
                        if _DCW["rdc_on"] else 0.0
                    ),
                    rdc_alpha_ll=float(_DCW["rdc_alpha_ll"]),
                    rdc_alpha_hh=float(_DCW["rdc_alpha_hh"]),
                    rdc_state=_RUNTIME.rdc_state,
                )
                _DCW["steps"] += 1
                if _DCW["dcw_on"]:
                    _DCW["dcw_steps"] += 1
                if _DCW["rdc_on"]:
                    _DCW["rdc_steps"] += 1
            except Exception as e:
                # DCW/RDC is the final optional transform. A bad/missing live
                # latent must not discard an already-valid CFG/PAG result.
                _log(
                    "DCW/RDC fallback (earlier guidance kept): "
                    f"{type(e).__name__}: {e}"
                )

        return result.to(denoised.dtype)
    except Exception as e:
        _log(f"post_cfg fallback → unmodified cfg: {type(e).__name__}: {e}")
        return denoised
    finally:
        _RUNTIME.close_step()


# ---------------------------------------------------------------------------
# Block-index parsing
# ---------------------------------------------------------------------------


def _parse_blocks(spec: str, n: int) -> set:
    """Parse "18", "14-27", "14,16,18" → valid block indices.

    The upstream Anima Safe PAG default is one balanced late block (18 on the
    common 28-block model). The old port treated an empty field as *every*
    block in the later half (14-27), which multiplied a soft perturbation into
    a destructive one. Keep the safe single-block default and clamp it for
    smaller compatible models.
    """
    spec = (spec or "").strip()
    if not spec:
        return {min(18, n - 1)} if n > 0 else set()
    out: set = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                out.update(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return {i for i in out if 0 <= i < n}


def _get_diffusion_model(unet):
    model = getattr(unet, "model", None)
    for attr in ("diffusion_model",):
        dm = getattr(model, attr, None)
        if dm is not None:
            return dm
    return None


_EXTRA_NETWORK_TAG = re.compile(
    r"<(?:lora|lyco|hypernet|embedding):[^>]+>",
    flags=re.IGNORECASE,
)


def _forge_and_models_roots() -> tuple[Path, Path]:
    from modules import paths

    return Path(paths.script_path), Path(paths.models_path)


def _clip_model_choices() -> list[str]:
    try:
        _, models_root = _forge_and_models_roots()
        return list_clip_l_models(models_root)
    except Exception:
        return []


def _first_processing_prompt(p, *, negative: bool = False) -> str:
    """Get the first fully-expanded WebUI prompt and remove extra-network tags."""
    plural = "all_negative_prompts" if negative else "all_prompts"
    singular = "negative_prompt" if negative else "prompt"
    value = getattr(p, plural, None)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    elif not isinstance(value, str):
        value = getattr(p, singular, "")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
    return _EXTRA_NETWORK_TAG.sub("", str(value or "")).strip()


# ---------------------------------------------------------------------------
# XYZ plot integration — lets users grid an ON/OFF (and param) comparison.
# The AxisOption apply functions mutate ``p`` per grid cell; the script's
# ``process_before_every_sampling`` then reads ``p._anima_safe_pag_xyz`` and
# overrides its own UI args from it (same pattern as the SAM3 script).
# ---------------------------------------------------------------------------


def _pag_xyz_set(p, x, xs, *, field: str):
    if not hasattr(p, "_anima_safe_pag_xyz"):
        p._anima_safe_pag_xyz = {}
    p._anima_safe_pag_xyz[field] = x


def _make_pag_xyz_axis() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break
    if xyz_grid is None:
        return

    bool_choices = lambda: ["True", "False"]  # noqa: E731

    axis = [
        # The headline request: ON/OFF comparison grid.
        xyz_grid.AxisOption(
            "[Anima Pert] Enable", str,
            partial(_pag_xyz_set, field="enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Attn Method", str,
            partial(_pag_xyz_set, field="method"),
            choices=lambda: ["PAG", "SEG", "None"],
        ),
        xyz_grid.AxisOption("[Anima Pert] Attn Scale", float, partial(_pag_xyz_set, field="scale")),
        xyz_grid.AxisOption(
            "[Anima Pert] Perturbation Strength", float,
            partial(_pag_xyz_set, field="strength"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Legacy Soft/Approx", str,
            partial(_pag_xyz_set, field="legacy_attn"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Legacy Perturbation Strength", float,
            partial(_pag_xyz_set, field="legacy_strength"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] SEG Blur Sigma", float,
            partial(_pag_xyz_set, field="seg_sigma"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Attn Block Indices", str,
            partial(_pag_xyz_set, field="blocks"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] Attn Head Indices", str,
            partial(_pag_xyz_set, field="heads"),
        ),
        xyz_grid.AxisOption(
            "[Anima Pert] SLG Enable", str,
            partial(_pag_xyz_set, field="slg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima Pert] SLG Scale", float, partial(_pag_xyz_set, field="slg_scale")),
        xyz_grid.AxisOption(
            "[Anima Pert] SLG Block Indices", str,
            partial(_pag_xyz_set, field="slg_blocks"),
        ),
        xyz_grid.AxisOption("[Anima Pert] Start Percent", float, partial(_pag_xyz_set, field="start")),
        xyz_grid.AxisOption("[Anima Pert] End Percent", float, partial(_pag_xyz_set, field="end")),
        xyz_grid.AxisOption("[Anima Pert] Rescale", float, partial(_pag_xyz_set, field="rescale")),
        xyz_grid.AxisOption(
            "[Anima Pert] Rescale Mode", str,
            partial(_pag_xyz_set, field="rescale_mode"),
            choices=lambda: ["full", "partial"],
        ),
        # APG axes (ON/OFF + the three knobs) for the same style of comparison.
        xyz_grid.AxisOption(
            "[Anima APG] Enable", str,
            partial(_pag_xyz_set, field="apg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima APG] Eta", float, partial(_pag_xyz_set, field="apg_eta")),
        xyz_grid.AxisOption("[Anima APG] Norm Threshold", float, partial(_pag_xyz_set, field="apg_norm")),
        xyz_grid.AxisOption("[Anima APG] Momentum", float, partial(_pag_xyz_set, field="apg_momentum")),
        # Adaptive Guidance (speed) axes.
        xyz_grid.AxisOption(
            "[Anima AdaptiveG] Enable", str,
            partial(_pag_xyz_set, field="adg_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption("[Anima AdaptiveG] Skip After", float, partial(_pag_xyz_set, field="adg_start")),
        xyz_grid.AxisOption("[Anima AdaptiveG] Keep Every", float, partial(_pag_xyz_set, field="adg_interval")),
        # Unified CFG-base and wavelet/control suite axes.
        xyz_grid.AxisOption(
            "[Anima CFG] Base Mode", str,
            partial(_pag_xyz_set, field="cfg_mode"),
            choices=lambda: [
                "Preserve incoming", "APG", "CWM", "SMC", "SMC + CWM"
            ],
        ),
        xyz_grid.AxisOption(
            "[Anima CFG] Experimental Stack", str,
            partial(_pag_xyz_set, field="cfg_stack"), choices=bool_choices,
        ),
        # xyz_grid persists a chosen axis by its INTEGER index in this list, so
        # the order here is a compatibility surface exactly like the script-arg
        # list: reordering it silently repoints every saved grid at a different
        # control. Keep the historical CWM -> SMC -> DCW order and append new
        # axes at the end (see "[Anima SMC] Preset" after the Mod block below).
        # The UI panel order is presentation-only and may differ.
        xyz_grid.AxisOption(
            "[Anima CWM] Enable", str,
            partial(_pag_xyz_set, field="cwm_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima CWM] Alpha Low", float,
            partial(_pag_xyz_set, field="cwm_alpha_low"),
        ),
        xyz_grid.AxisOption(
            "[Anima CWM] Alpha High", float,
            partial(_pag_xyz_set, field="cwm_alpha_high"),
        ),
        xyz_grid.AxisOption(
            "[Anima SMC] Enable", str,
            partial(_pag_xyz_set, field="smc_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima SMC] Lambda", float,
            partial(_pag_xyz_set, field="smc_lambda"),
        ),
        xyz_grid.AxisOption(
            "[Anima SMC] K", float,
            partial(_pag_xyz_set, field="smc_k"),
        ),
        xyz_grid.AxisOption(
            "[Anima DCW] Enable", str,
            partial(_pag_xyz_set, field="dcw_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima DCW] Lambda Low", float,
            partial(_pag_xyz_set, field="dcw_lambda_low"),
        ),
        xyz_grid.AxisOption(
            "[Anima DCW] Lambda High", float,
            partial(_pag_xyz_set, field="dcw_lambda_high"),
        ),
        xyz_grid.AxisOption(
            "[Anima DAVE] Enable", str,
            partial(_pag_xyz_set, field="dave_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima DAVE] Strength", float,
            partial(_pag_xyz_set, field="dave_strength"),
        ),
        xyz_grid.AxisOption(
            "[Anima DAVE] Tau", float,
            partial(_pag_xyz_set, field="dave_tau"),
        ),
        xyz_grid.AxisOption(
            "[Anima DAVE] Block Indices", str,
            partial(_pag_xyz_set, field="dave_blocks"),
        ),
        xyz_grid.AxisOption(
            "[Anima CNS] Enable", str,
            partial(_pag_xyz_set, field="cns_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima CNS] Strength", float,
            partial(_pag_xyz_set, field="cns_strength"),
        ),
        xyz_grid.AxisOption(
            "[Anima CNS] Gamma Power", float,
            partial(_pag_xyz_set, field="cns_gamma_power"),
        ),
        xyz_grid.AxisOption(
            "[Anima CNS] Gamma Scale", float,
            partial(_pag_xyz_set, field="cns_gamma_scale"),
        ),
        xyz_grid.AxisOption(
            "[Anima Mod] Enable", str,
            partial(_pag_xyz_set, field="mod_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima Mod] Direction Weight", float,
            partial(_pag_xyz_set, field="mod_weight"),
        ),
        xyz_grid.AxisOption(
            "[Anima Mod] Start Block", float,
            partial(_pag_xyz_set, field="mod_start_layer"),
        ),
        xyz_grid.AxisOption(
            "[Anima Mod] End Block", float,
            partial(_pag_xyz_set, field="mod_end_layer"),
        ),
        # Appended last on purpose — see the ordering note above.
        xyz_grid.AxisOption(
            "[Anima SMC] Preset", str,
            partial(_pag_xyz_set, field="smc_preset"),
            choices=lambda: list(SMC_PRESET_NAMES),
        ),
        # RDC was introduced upstream after the existing 47-axis compatibility
        # prefix. Append only: xyz_grid persists the integer axis index.
        xyz_grid.AxisOption(
            "[Anima RDC] Enable", str,
            partial(_pag_xyz_set, field="rdc_enabled"), choices=bool_choices,
        ),
        xyz_grid.AxisOption(
            "[Anima RDC] Tau", float,
            partial(_pag_xyz_set, field="rdc_tau"),
        ),
        xyz_grid.AxisOption(
            "[Anima RDC] Alpha LL", float,
            partial(_pag_xyz_set, field="rdc_alpha_ll"),
        ),
        xyz_grid.AxisOption(
            "[Anima RDC] Alpha HH", float,
            partial(_pag_xyz_set, field="rdc_alpha_hh"),
        ),
    ]

    # Register per label so a WebUI "reload scripts" after an extension update
    # can add newly introduced axes without duplicating the older ones.
    existing_labels = {getattr(option, "label", None) for option in xyz_grid.axis_options}
    xyz_grid.axis_options.extend(
        option for option in axis if option.label not in existing_labels
    )


def _pag_on_before_ui() -> None:
    try:
        _make_pag_xyz_axis()
    except Exception:
        _log("xyz_grid axis registration failed:\n" + traceback.format_exc())


script_callbacks.on_before_ui(_pag_on_before_ui)


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _finite_clamp(value, low: float, high: float, default: float) -> float:
    """Clamp UI/XYZ numeric input and reject NaN/Inf safely."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return min(high, max(low, number))


def _clear_extra_generation_params(p) -> None:
    """Remove per-cell guidance metadata before applying XYZ overrides.

    ``p`` is reused across XYZ cells. Without this cleanup, a False cell keeps
    the preceding True cell's "Anima Perturbation Guidance" infotext even
    though the hook is disabled.
    """
    params = getattr(p, "extra_generation_params", None)
    if not isinstance(params, dict):
        return
    for key in _EXTRA_GENERATION_PARAM_KEYS:
        params.pop(key, None)


# ---------------------------------------------------------------------------
# The extension script
# ---------------------------------------------------------------------------


class AnimaSafePAG(scripts.Script):
    @property
    def section(self):
        # Forge 는 사용자 섹션을 설정값 칼럼(#txt2img_settings) 안에 만든다 → 1열 "ANIMA 튜닝" 자리.
        # 설정(sam3_layout_sections)을 끄거나 img2img 면 None 이라 예전과 똑같이 스크립트 컨테이너로 간다.
        return layout_lanes.anima_section(bool(getattr(self, "is_img2img", False)))

    # sorting_priority still governs the accordion position. Current Forge Neo
    # accidentally defines process_before_every_sampling twice; the later raw
    # alwayson_scripts loop wins and ignores this priority for execution.
    # Skimmed CFG therefore enforces its post-CFG precedence by prepending its
    # callback explicitly. We still clone from the CURRENT forge_objects.unet,
    # so other unet-patching scripts compose regardless of load order.
    sorting_priority = -27

    def title(self):
        return "Anima Perturbation Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        clip_choices = _clip_model_choices()
        default_clip = clip_choices[0] if clip_choices else None
        with gr.Accordion("Anima Perturbation Guidance (PAG / SEG / SLG)", open=False):
            gr.Markdown(
                "Anima/Cosmos/Predict2 계열 **DiT 전용** perturbation guidance. 후반 블록에 "
                "*약한 예측*을 만들어 CFG를 그 반대로 밀어 구조·디테일을 강화합니다. **Anima "
                "엔진에서만** 동작하고 그 외/오류 시 폴백합니다. 약한 예측은 별도 forward 없이 "
                "**같은 배치에 접어** 계산합니다.\n\n"
                "- **PAG**(value-only) ↔ **SEG**(uniform/blur)는 성격이 겹쳐 **택1**\n"
                "- **SLG**(레이어 스킵)는 PAG/SEG와 **병용 가능**"
            )
            enabled = gr.Checkbox(
                label="Enable Perturbation Guidance",
                value=False,
                elem_id="anima_safe_pag_enable",
                elem_classes=["sam3-on", "sam3-on--pag"],
            )
            gr.Markdown(
                "#### PAG / SEG — Attention perturbation\n"
                "기본은 **공식 경로**입니다: PAG는 value-only, SEG는 실제 H·W query에 "
                "Gaussian blur를 적용합니다. Strength=1이면 원 기법의 전체 perturbation, "
                "기본 0.75는 Anima Safe PAG의 부드러운 권장값입니다. `None`은 SLG만 "
                "사용할 때 선택하세요."
            )
            attn_method = gr.Radio(
                label="Attention perturbation method",
                choices=["PAG", "SEG", "None"],
                value="PAG",
                info="PAG와 SEG는 둘 중 하나만 선택합니다. SLG만 쓸 때는 None을 선택하세요.",
                elem_id="anima_safe_pag_method",
            )
            scale = gr.Slider(
                label="Attn Scale — PAG / SEG guidance scale (cond−weak 배율)",
                minimum=0.0, maximum=15.0, step=0.1, value=4.0,
                info="이미지가 찢어지거나 배경·구도가 과하게 변하면 이 값을 먼저 낮추세요.",
                elem_id="anima_safe_pag_scale",
            )
            official_strength = gr.Slider(
                label="Perturbation strength (PAG→value · SEG→blurred query · 1=전체)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.75,
                info="윤곽·질감이 깨지거나 노이즈가 늘면 낮추세요. 1.0이 가장 강합니다.",
                elem_id="anima_safe_pag_official_strength",
            )
            seg_sigma = gr.Slider(
                label="SEG Gaussian query blur sigma (공식 · >9999=uniform query)",
                minimum=0.0, maximum=10000.0, step=1.0, value=100.0,
                info="SEG 전용입니다. 형태가 뭉개지거나 영향이 과하면 낮추세요.",
                elem_id="anima_safe_pag_seg_sigma",
            )
            with gr.Accordion("Legacy Soft/Approx 호환", open=False):
                legacy_attn = gr.Checkbox(
                    label="기존 Soft PAG / SEG-Approx 사용 (공식 모드 끄기)",
                    value=False,
                    elem_id="anima_safe_pag_legacy_attn",
                )
                legacy_strength = gr.Slider(
                    label="Legacy perturbation strength (PAG→value · SEG→uniform value)",
                    minimum=0.0, maximum=1.0, step=0.01, value=0.75,
                    info="Legacy 모드 전용입니다. 이미지가 깨지거나 검게 무너지면 낮추세요.",
                    elem_id="anima_safe_pag_strength",
                )
            block_indices = gr.Textbox(
                label="Attention block indices (권장·빈칸 기본=18, 예: 18 / 18-20)",
                value="18",
                info="여러 블록은 효과와 위험을 함께 키웁니다. 이상하면 18 하나로 되돌리세요.",
                elem_id="anima_safe_pag_blocks",
            )
            head_indices = gr.Textbox(
                label="Attention head indices (빈칸=전체, 예: 0,2,4-7)",
                value="",
                info="실험용 세부 선택입니다. 띠·분할 무늬가 생기면 빈칸으로 복귀한 뒤 Scale/Strength부터 낮추세요.",
                elem_id="anima_safe_pag_heads",
            )

            gr.Markdown(
                "#### 공통 적용 범위·보정 — PAG / SEG / SLG\n"
                "아래 Start·End·Rescale은 **활성화한 모든 perturbation에 공통 적용**됩니다. "
                "SLG를 끈 상태에서도 PAG/SEG에 그대로 적용됩니다."
            )
            with gr.Row():
                start_percent = gr.Slider(
                    label="Start percent (공통 · 0=샘플링 시작)",
                    minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                    info="초반 구도·인물 배치가 무너지면 값을 올려 더 늦게 시작하세요.",
                    elem_id="anima_safe_pag_start",
                )
                end_percent = gr.Slider(
                    label="End percent (공통 · 1=샘플링 끝)",
                    minimum=0.0, maximum=1.0, step=0.01, value=0.7,
                    info="후반 윤곽·세부가 깨지면 값을 낮춰 더 일찍 끝내세요.",
                    elem_id="anima_safe_pag_end",
                )
            rescale = gr.Slider(
                label="Rescale (공통 · 대비/채도 과다 억제)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.20,
                info="과채도·과대비면 올리고, 색이 탁하거나 대비가 눌리면 낮추거나 0으로 비교하세요.",
                elem_id="anima_safe_pag_rescale",
            )
            rescale_mode = gr.Radio(
                label="Rescale mode (full=CFG 결과 기준 · partial=cond 기준)",
                choices=["full", "partial"],
                value="full",
                info="대부분 full을 권장합니다. partial에서 결과가 불안정하면 full로 되돌리세요.",
                elem_id="anima_safe_pag_rescale_mode",
            )

            gr.Markdown(
                "#### SLG — Skip Layer Guidance\n"
                "선택한 block을 건너뛴 별도 weak prediction을 만듭니다. PAG 또는 SEG와 "
                "병용할 수 있지만 아직 전용 E2E 검증은 없습니다."
            )
            slg_on = gr.Checkbox(
                label="Enable SLG (skip layers)",
                value=False,
                elem_id="anima_safe_pag_slg_enable",
                elem_classes=["sam3-on", "sam3-on--slg"],
            )
            slg_scale = gr.Slider(
                label="SLG guidance scale",
                minimum=0.0, maximum=15.0, step=0.1, value=3.0,
                info="인물·배경이 갈라지거나 형태가 깨지면 낮추세요.",
                elem_id="anima_safe_pag_slg_scale",
            )
            slg_blocks = gr.Textbox(
                label="SLG skip block indices (빈칸 기본=18)",
                value="18",
                info="여러 블록을 건너뛸수록 불안정해질 수 있습니다. 이상하면 18 하나만 쓰세요.",
                elem_id="anima_safe_pag_slg_blocks",
            )

            # Removed: the "auto-decay" safety brake that divided each PAG/SEG/
            # SLG scale by the active-term count. Perturbations now always apply
            # at their full configured scale. A hidden, inert placeholder keeps
            # this script argument at its historical index (11) so saved API
            # payloads / XYZ presets that pass the full positional array still
            # line up — matching the extension's append-only arg contract.
            auto_decay = gr.Checkbox(value=False, visible=False)

            gr.Markdown("---\n### APG (Adaptive Projected Guidance)")
            gr.Markdown(
                "⚠️ **실험 구현 · CFG > 1 전용.** 높은 CFG의 과채도·번짐 성분을 줄이기 "
                "위해 Forge의 post-CFG denoised(x0) 공간에서 projection·norm·momentum을 "
                "적용합니다. 추가 forward는 없지만 reference APG와 계산 위치가 달라 동등한 "
                "결과를 보장하지 않습니다. PAG와 함께 쓸 때는 고정 시드 A/B로 확인하세요."
            )
            apg_enabled = gr.Checkbox(
                label="Enable APG (실험 · CFG > 1)",
                value=False,
                elem_id="anima_safe_pag_apg_enable",
                elem_classes=["sam3-on", "sam3-on--apg"],
            )
            apg_autooff = gr.Checkbox(
                label="APG 켜지면 PAG rescale 자동 끄기 (이중 크기보정 방지)",
                value=True,
                elem_id="anima_safe_pag_apg_autooff",
            )
            with gr.Accordion("APG Advanced (세부값)", open=False):
                gr.Markdown(
                    "기본값이면 무난합니다. 파고들 때: **eta**=평행(과채도) 성분 비중 "
                    "(0=최대 억제, 1=표준 CFG로 환원), **norm threshold**=guidance 크기 "
                    "상한(0=off), **momentum**=스텝 간 running-average(음수 권장, 0=off). "
                    "guidance 세기는 메인 **CFG Scale** 슬라이더를 그대로 씁니다."
                )
                apg_eta = gr.Slider(
                    label="APG eta (평행 성분 비중 · 낮을수록 과채도↓)",
                    minimum=-10.0, maximum=10.0, step=0.05, value=0.0,
                    info="과채도·번짐은 낮추고, 스타일이나 구조가 너무 약해지면 1 쪽으로 올리세요.",
                    elem_id="anima_safe_pag_apg_eta",
                )
                apg_norm = gr.Slider(
                    label="APG norm threshold (guidance 크기 상한 · 0=off)",
                    minimum=0.0, maximum=50.0, step=0.5, value=15.0,
                    info="강한 CFG 아티팩트에는 낮추고, 결과가 지나치게 눌리면 올리거나 0으로 끄세요.",
                    elem_id="anima_safe_pag_apg_norm",
                )
                apg_momentum = gr.Slider(
                    label="APG momentum (스텝 간 running-average · 음수 권장 · 0=off)",
                    minimum=-1.0, maximum=1.0, step=0.05, value=0.0,
                    info="잔상·고스팅이나 스텝 간 불안정이 보이면 0 쪽으로 되돌리세요.",
                    elem_id="anima_safe_pag_apg_momentum",
                )

            gr.Markdown("---\n### Adaptive Guidance (속도)")
            gr.Markdown(
                "⚠️ 논문의 cosine-similarity 판정이 아닌 **고정 Skip-after 방식**입니다. "
                "Forge가 cond/uncond를 같은 batch로 호출할 때만 후반 uncond row를 생략합니다. "
                "low-VRAM 분리 호출에서는 스킵되지 않아 속도 차이가 없으며, 특정 향상률을 "
                "보장하지 않습니다. 생략된 스텝에서는 perturbation도 함께 쉽니다."
            )
            adg_enabled = gr.Checkbox(
                label="Enable Adaptive Guidance (combined-batch에서만 후반 uncond 생략)",
                value=False,
                elem_id="anima_safe_pag_adg_enable",
                elem_classes=["sam3-on", "sam3-on--adg"],
            )
            adg_start = gr.Slider(
                label="Skip after (이 지점 이후 uncond 생략)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                info="품질 손실이 보이면 올려 더 늦게 생략하고, 속도 우선이면 낮추세요.",
                elem_id="anima_safe_pag_adg_start",
            )
            with gr.Accordion("Adaptive Guidance Advanced (세부값)", open=False):
                gr.Markdown(
                    "**keep every N** — 생략 구간에서도 N스텝마다 uncond를 한 번씩 유지해 "
                    "품질을 보수적으로 지킵니다(0=항상 생략, 값 클수록 더 자주 생략)."
                )
                adg_interval = gr.Slider(
                    label="Keep every N steps (0 = 항상 생략)",
                    minimum=0, maximum=10, step=1, value=0,
                    info="품질 손실 시 1~2로 두고, 속도 우선이면 값을 키우거나 0(항상 생략)을 쓰세요.",
                    elem_id="anima_safe_pag_adg_interval",
                )
            gr.Markdown(
                "Adaptive Guidance와 다른 변형을 병용할 때는 "
                "`Skip after ≥ 0.65`부터 시작하는 편이 안전합니다. "
                "TeaCache는 이 Suite에 포함하지 않습니다."
            )

            gr.Markdown("---\n### Guidance Orchestrator (CFG / Wavelet / Control)")
            gr.Markdown(
                "**DCW · RDC · CWM · SMC · APG는 서로 독립 토글**입니다. 원하는 만큼 함께 켤 수 "
                "있고, 여러 개가 켜지면 항상 **SMC → APG → CWM** 순서로 적용됩니다. "
                "APG 토글은 위 APG 섹션에 있습니다. CFG 세 기능(SMC·APG·CWM)을 "
                "모두 끄면 Forge·MaHiRo·다른 CFG "
                "확장의 결과를 그대로 보존합니다. 아래 패널은 찾기 쉽도록 "
                "**DCW → RDC → CWM → SMC** 순서로 배치했습니다."
            )

            gr.Markdown("#### DCW — post-CFG wavelet correction")
            dcw_enabled = gr.Checkbox(
                label="Enable DCW",
                value=False,
                elem_id="anima_guidance_dcw_enable",
                elem_classes=["sam3-on", "sam3-on--dcw"],
            )
            with gr.Row():
                dcw_lambda_low = gr.Slider(
                    label="DCW lambda low",
                    minimum=-0.5, maximum=0.5, step=0.005, value=0.10,
                    info="구도·밝기·큰 색면이 어색해지면 0 쪽으로 줄이세요.",
                    elem_id="anima_guidance_dcw_lambda_low",
                )
                dcw_lambda_high = gr.Slider(
                    label="DCW lambda high",
                    minimum=-0.5, maximum=0.5, step=0.005, value=0.02,
                    info="윤곽 링·미세 노이즈가 생기면 0 쪽으로 줄이세요.",
                    elem_id="anima_guidance_dcw_lambda_high",
                )

            gr.Markdown("#### RDC — band-wise reverse drift compensation")
            rdc_enabled = gr.Checkbox(
                label="Enable RDC",
                value=False,
                info=(
                    "여러 스텝에 걸쳐 구도·포즈가 서서히 표류하는 현상을 대역별 "
                    "EMA로 억제합니다. DCW와 독립적으로 켤 수 있습니다."
                ),
                elem_id="anima_guidance_rdc_enable",
                elem_classes=["sam3-on", "sam3-on--rdc"],
            )
            rdc_tau = gr.Slider(
                label="RDC tau (EMA 기억 구간)",
                minimum=0.0, maximum=0.5, step=0.01, value=0.15,
                info=(
                    "0.05~0.10은 빠른 반응, 0.15~0.20은 권장 절충값입니다. "
                    "구도가 초반 상태에 고착되면 낮추세요."
                ),
                elem_id="anima_guidance_rdc_tau",
            )
            with gr.Row():
                rdc_alpha_ll = gr.Slider(
                    label="RDC alpha LL (구조 drift)",
                    minimum=0.0, maximum=0.3, step=0.005, value=0.03,
                    info=(
                        "권장 시작값 0.02~0.05. 포즈·구도 변화가 지나치게 "
                        "고정되면 0 쪽으로 낮추세요."
                    ),
                    elem_id="anima_guidance_rdc_alpha_ll",
                )
                rdc_alpha_hh = gr.Slider(
                    label="RDC alpha HH (텍스처 drift)",
                    minimum=0.0, maximum=0.1, step=0.001, value=0.0,
                    info=(
                        "기본 0 권장. 텍스처가 흐려지면 0으로 되돌리세요. "
                        "필요한 경우에도 0.01 이하부터 시작하세요."
                    ),
                    elem_id="anima_guidance_rdc_alpha_hh",
                )

            gr.Markdown("#### CWM — CFG wavelet mixing (주파수 대역별 CFG 재가중)")
            cwm_enabled = gr.Checkbox(
                label="Enable CWM",
                value=False,
                info="alpha low·high가 모두 0이면 켜도 표준 CFG와 같습니다.",
                elem_id="anima_guidance_cwm_enable",
                elem_classes=["sam3-on", "sam3-on--cwm"],
            )
            with gr.Row():
                cwm_alpha_low = gr.Slider(
                    label="CWM alpha low (초반 저주파 CFG)",
                    minimum=-1.0, maximum=1.0, step=0.01, value=0.30,
                    info="전체 구도·큰 색면이 과하게 변하면 0 쪽으로 줄이세요.",
                    elem_id="anima_guidance_cwm_alpha_low",
                )
                cwm_alpha_high = gr.Slider(
                    label="CWM alpha high (후반 고주파 CFG)",
                    minimum=-1.0, maximum=1.0, step=0.01, value=0.15,
                    info="인물 복제·윤곽 링·세부 노이즈가 생기면 0 쪽으로 줄이세요.",
                    elem_id="anima_guidance_cwm_alpha_high",
                )
            alpha_high_warning = gr.Markdown(
                "⚠️ **Anima 주의:** `alpha high > +0.15`는 16채널 HH 대역에서 "
                "한 인물이 여러 인물로 갈라지는 현상을 만들 수 있습니다.",
                visible=False,
            )

            def _show_alpha_high_warning(value):
                return gr.update(visible=float(value) > 0.15)

            cwm_alpha_high.change(
                fn=_show_alpha_high_warning,
                inputs=[cwm_alpha_high],
                outputs=[alpha_high_warning],
                show_progress=False,
            )

            gr.Markdown("#### SMC — sliding-mode control (스텝 간 CFG error 안정화)")
            smc_master_enabled = gr.Checkbox(
                label="Enable SMC",
                value=False,
                info=(
                    "프리셋 값은 유지한 채 SMC만 즉시 ON/OFF합니다. "
                    "CWM과 함께 켜면 SMC → CWM 순서로 실행됩니다."
                ),
                elem_id="anima_guidance_smc_master_enable",
                elem_classes=["sam3-on", "sam3-on--smc"],
            )
            smc_preset = gr.Dropdown(
                label="SMC preset",
                choices=list(SMC_PRESET_NAMES[1:]),
                value="Auto",
                info=(
                    "Auto는 현재 모델을 감지해 원본 ComfyUI-DCW 값을 적용합니다. "
                    "Anima는 Cosmos / Wan (lambda 6.0, k 0.20)으로 판별됩니다."
                ),
                elem_id="anima_guidance_smc_preset",
            )
            gr.Markdown(
                "`Custom`을 선택했을 때만 아래 두 값이 사용됩니다. "
                "다른 프리셋에서는 표시값과 무관하게 프리셋 값이 적용됩니다."
            )
            with gr.Row():
                smc_lambda = gr.Slider(
                    label="Custom SMC lambda",
                    minimum=0.5, maximum=30.0, step=0.1, value=6.0,
                    info="스텝 간 흔들림이나 과보정이 보이면 낮추세요.",
                    elem_id="anima_guidance_smc_lambda",
                )
                smc_k = gr.Slider(
                    label="Custom SMC k",
                    minimum=0.0, maximum=5.0, step=0.01, value=0.10,
                    info="보정이 튀거나 디테일이 깨지면 낮추세요. 0이면 Custom SMC가 중립입니다.",
                    elem_id="anima_guidance_smc_k",
                )

            with gr.Accordion("Legacy CFG base mode (구버전 호환)", open=False):
                gr.Markdown(
                    "예전의 상호배타 라디오입니다. 저장된 infotext·API 호출·XYZ 그리드가 "
                    "그대로 동작하도록 남겨 두었으며, 위 토글들과 **OR**로 합쳐집니다. "
                    "새로 설정할 때는 건드릴 필요가 없습니다."
                )
                cfg_mode = gr.Radio(
                    label="CFG base mode (legacy · 상호배타)",
                    choices=[
                        "Preserve incoming",
                        "APG",
                        "CWM",
                        "SMC",
                        "SMC + CWM",
                    ],
                    value="Preserve incoming",
                    info="호환 문제나 이미지 붕괴가 생기면 Preserve incoming으로 되돌리세요.",
                    elem_id="anima_guidance_cfg_mode",
                    elem_classes=["sam3-on-radio"],
                )
                experimental_stack = gr.Checkbox(
                    label="Experimental stack: SMC → APG → CWM (legacy 단축)",
                    value=False,
                    info="세 토글을 모두 켜는 것과 같습니다. 새 토글을 쓰면 필요 없습니다.",
                    elem_id="anima_guidance_experimental_stack",
                    elem_classes=["sam3-on", "sam3-on--apg", "sam3-on--smc", "sam3-on--cwm"],
                )
                smc_enabled = gr.Checkbox(
                    label="Enable SMC (legacy)",
                    value=False,
                    info=(
                        "이전 버전의 저장값·API 인덱스용입니다. 켜면 위 preset이 "
                        "Off여도 Custom lambda/k로 SMC가 활성화됩니다."
                    ),
                    elem_id="anima_guidance_smc_enable",
                    elem_classes=["sam3-on", "sam3-on--smc"],
                )

            gr.Markdown("#### DAVE — Anima diversity · block DC attenuation")
            dave_enabled = gr.Checkbox(
                label="Enable DAVE",
                value=False,
                elem_id="anima_guidance_dave_enable",
                elem_classes=["sam3-on", "sam3-on--dave"],
            )
            dave_strength = gr.Slider(
                label="DAVE strength",
                minimum=0.0, maximum=1.0, step=0.01, value=0.30,
                info="구조가 사라지거나 결과가 지나치게 달라지면 낮추세요.",
                elem_id="anima_guidance_dave_strength",
            )
            dave_tau = gr.Slider(
                label="DAVE early-step tau (0=전 구간, 권장 0.10)",
                minimum=0.0, maximum=1.0, step=0.01, value=0.10,
                info="영향 시간이 길면 0을 피하고 0.05~0.10처럼 낮은 양수로 줄이세요.",
                elem_id="anima_guidance_dave_tau",
            )
            dave_blocks = gr.Textbox(
                label="DAVE block indices (기본 8-18)",
                value="8-18",
                info="형태가 불안정하면 대상 블록 수를 줄여 한두 블록부터 비교하세요.",
                elem_id="anima_guidance_dave_blocks",
            )

            gr.Markdown("#### CNS-inspired Wavelet Noise — ancestral/SDE")
            gr.Markdown(
                "기존 seeded/Brownian noise를 새로 만들지 않고 **재색칠**합니다. "
                "noise_sampler를 쓰지 않는 결정론적 sampler에서는 자동으로 무효입니다."
            )
            cns_enabled = gr.Checkbox(
                label="Enable CNS-inspired Wavelet Noise",
                value=False,
                elem_id="anima_guidance_cns_enable",
                elem_classes=["sam3-on", "sam3-on--cns"],
            )
            cns_strength = gr.Slider(
                label="CNS strength",
                minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                info="색 노이즈·거친 입자·구조 변형이 과하면 먼저 낮추세요.",
                elem_id="anima_guidance_cns_strength",
            )
            cns_gamma_power = gr.Slider(
                label="CNS gamma power",
                minimum=0.05, maximum=2.0, step=0.05, value=0.5,
                info="주파수별 색 노이즈가 어색하면 기본값 0.5로 되돌린 뒤 Strength를 낮추세요.",
                elem_id="anima_guidance_cns_gamma_power",
            )
            cns_gamma_scale = gr.Slider(
                label="CNS gamma scale (Anima 시작값 3.0)",
                minimum=0.25, maximum=25.0, step=0.25, value=3.0,
                info="노이즈 분포가 과장되면 기본값 3.0으로 되돌린 뒤 Strength를 낮추세요.",
                elem_id="anima_guidance_cns_gamma_scale",
            )

            gr.Markdown(
                "---\n### Anima Modulation Guidance (보조 CLIP-L)\n"
                "⚠️ **실험 기능 · Anima 전용 · 기본 OFF.** 메인 Qwen conditioning을 "
                "교체하지 않습니다. 별도 CLIP-L의 pooled embedding을 공개 Cosmos/Anima "
                "어댑터로 투영해 선택한 DiT block의 AdaLN에 더합니다. 첫 사용 시 공식 "
                "어댑터 약 171 MB를 내려받으며, CLIP-L과 어댑터는 이후 생성을 위해 CPU "
                "RAM에 캐시됩니다. 스타일 LoRA가 약해지거나 단순히 결과가 달라질 수 "
                "있으므로 여러 seed로 비교하세요."
            )
            mod_enabled = gr.Checkbox(
                label="Enable Anima Modulation Guidance (CLIP-L)",
                value=False,
                elem_id="anima_mod_guidance_enable",
                elem_classes=["sam3-on", "sam3-on--mod"],
            )
            mod_clip_model = gr.Dropdown(
                label="CLIP-L model (models/text_encoder)",
                choices=clip_choices,
                value=default_clip,
                allow_custom_value=True,
                info=(
                    "권장: Anzhc NoobAI11 CLIP L Anime. CLIP-G는 제공된 "
                    "768차원 어댑터와 호환되지 않습니다."
                ),
                elem_id="anima_mod_guidance_clip_model",
            )
            mod_weight = gr.Slider(
                label="Direction weight w",
                minimum=-20.0,
                maximum=20.0,
                step=0.05,
                value=3.0,
                info=(
                    "positive−negative 방향의 배율입니다. 이미지가 과하게 바뀌거나 "
                    "스타일이 약해지면 낮추세요. w=0도 base CLIP modulation은 남습니다."
                ),
                elem_id="anima_mod_guidance_weight",
            )
            with gr.Row():
                mod_start_layer = gr.Slider(
                    label="Start block (포함)",
                    minimum=0,
                    maximum=63,
                    step=1,
                    value=0,
                    info="구도·프롬프트 준수가 약해지면 더 뒤 block부터 시작하세요.",
                    elem_id="anima_mod_guidance_start_layer",
                )
                mod_end_layer = gr.Slider(
                    label="End block (포함 · -1=마지막)",
                    minimum=-1,
                    maximum=63,
                    step=1,
                    value=-1,
                    info="영향이 과하면 범위를 줄이세요. -1은 마지막 block까지입니다.",
                    elem_id="anima_mod_guidance_end_layer",
                )
            mod_base_source = gr.Radio(
                label="Base CLIP prompt source",
                choices=["Main positive", "Custom"],
                value="Main positive",
                info="Main positive는 현재 생성 프롬프트에서 LoRA 태그만 제거해 사용합니다.",
                elem_id="anima_mod_guidance_base_source",
            )
            mod_base_prompt = gr.Textbox(
                label="Custom base CLIP prompt",
                value="",
                lines=2,
                info="Base source가 Custom일 때만 사용합니다.",
                elem_id="anima_mod_guidance_base_prompt",
            )
            mod_positive_prompt = gr.Textbox(
                label="Positive direction prompt",
                value="masterpiece, best quality, highres",
                lines=2,
                info="이 방향이 과하게 스타일을 덮으면 태그를 줄이거나 w를 낮추세요.",
                elem_id="anima_mod_guidance_positive_prompt",
            )
            mod_negative_source = gr.Radio(
                label="Negative direction source",
                choices=["Main negative", "Custom"],
                value="Main negative",
                info="Main negative는 현재 네거티브 프롬프트를 사용합니다.",
                elem_id="anima_mod_guidance_negative_source",
            )
            mod_negative_prompt = gr.Textbox(
                label="Custom negative direction prompt",
                value="worst quality, low quality",
                lines=2,
                info="Negative source가 Custom일 때만 사용합니다.",
                elem_id="anima_mod_guidance_negative_prompt",
            )
            with gr.Accordion("Modulation adapter (고급)", open=False):
                mod_adapter_mode = gr.Radio(
                    label="Adapter source",
                    choices=["Auto-download official", "Local file"],
                    value="Auto-download official",
                    elem_id="anima_mod_guidance_adapter_mode",
                )
                mod_adapter_path = gr.Textbox(
                    label="Local adapter path",
                    value="",
                    info="Local file 모드에서만 사용합니다.",
                    elem_id="anima_mod_guidance_adapter_path",
                )
        return [
            enabled, attn_method, scale, legacy_strength, block_indices,
            slg_on, slg_scale, slg_blocks,
            start_percent, end_percent, rescale, auto_decay,
            apg_enabled, apg_eta, apg_norm, apg_momentum, apg_autooff,
            adg_enabled, adg_start, adg_interval,
            legacy_attn, seg_sigma,
            cfg_mode, experimental_stack,
            cwm_alpha_low, cwm_alpha_high, smc_lambda, smc_k,
            dcw_enabled, dcw_lambda_low, dcw_lambda_high,
            dave_enabled, dave_strength, dave_tau, dave_blocks,
            cns_enabled, cns_strength, cns_gamma_power, cns_gamma_scale,
            # Appended to preserve every pre-v0.13 script-argument index.
            official_strength, head_indices, rescale_mode,
            # Appended for the same reason when SMC/CWM became independent
            # toggles instead of entries in the cfg_mode radio.
            smc_enabled, cwm_enabled,
            # Appended in v0.20 so every older script-argument index remains
            # stable across infotext/API restoration.
            mod_enabled, mod_clip_model, mod_weight,
            mod_start_layer, mod_end_layer,
            mod_base_source, mod_base_prompt, mod_positive_prompt,
            mod_negative_source, mod_negative_prompt,
            mod_adapter_mode, mod_adapter_path,
            # Appended after v0.20 to preserve all 56 older argument indexes.
            smc_preset,
            # Appended after v0.21.2. Keep the complete historical prefix:
            # current SMC gets an explicit master toggle and RDC gets its own
            # independent toggle plus all three upstream parameters.
            smc_master_enabled,
            rdc_enabled, rdc_tau, rdc_alpha_ll, rdc_alpha_hh,
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        if torch is None:
            return

        # XYZ-plot overrides (set per grid cell by the AxisOption apply fns).
        xyz = getattr(p, "_anima_safe_pag_xyz", {}) or {}
        _clear_extra_generation_params(p)
        _STATE.pop("condition_aggregation", None)
        _RUNTIME.reset_pass()
        _APG.update(on=False, avg=None, last_sigma=None)
        _ADG["on"] = False
        _CFG.update(
            smc_on=False, apg_on=False, cwm_on=False,
            mode="preserve", experimental_stack=False, steps=0,
            smc_preset="Off", smc_resolved_preset="Off",
            fit_error=None, effective_scale=None,
            external_cfg_detected=False, warned=False, cfg1_warned=False,
            base_skipped=False, fit_checked=False,
        )
        _DCW.update(
            on=False,
            dcw_on=False,
            rdc_on=False,
            steps=0,
            dcw_steps=0,
            rdc_steps=0,
        )
        _DAVE.update(on=False, targets=set(), steps=0)
        _CNS.update(on=False, warned=False)
        _MOD.update(
            on=False,
            targets=set(),
            block_modulations=None,
            typed={},
            hits=0,
            warned=False,
            clip_model="",
            adapter_path="",
        )
        _STATE.update(
            on=False, attn_method=None, attn_targets=set(), head_spec="",
            strength=0.75, rescale_mode="full",
            slg_on=False, slg_targets=set(), active=0,
            wrapper_calls=0, weak_steps=0, applied_steps=0, apg_steps=0,
            adg_skipped_steps=0, combined_calls=0, split_cond_calls=0,
            split_uncond_calls=0, control_blocked_calls=0,
            wrapper_fallbacks=0, requested_pert=False, requested_apg=False,
            requested_adg=False, requested_cfg_mode="preserve",
            requested_cfg_stack=False, requested_smc=False,
            requested_cwm=False, requested_dcw=False,
            requested_rdc=False,
            requested_dave=False, requested_cns=False,
            requested_modulation=False,
            requested_method=None, engine="?",
            diag_started_at=time.perf_counter(), delta_logged=False,
            attn_hook_hits=0, attn_hook_hits_total=0,
            attn_last_rel_delta=None,
            attn_diag_logged=False, attn_shape_warned=False,
            attn_spatial_shape=None, attn_raw=None, slg_raw=None,
            cond_raw=None,
            adg_skipped=False, step_open=False,
            # 결과를 미세하게 바꾸는 Forge 설정 두 개 — 패스마다 읽어 infotext 와 맞춘다.
            prefix_dedup=_read_bool_option(OPT_PREFIX_DEDUP, True),
            seg_separable=_read_bool_option(OPT_SEG_SEPARABLE, True),
            dedup_blocks=0, dedup_fallbacks=0, dedup_warned=False,
            dedup_disabled=False,
        )
        _clear_markers()
        # 같은 p(hires 패스, n_iter 다음 배치)면 앞선 OOM 차단을 유지하고,
        # 다른 생성이면 지난 기록을 버린다.
        if not _perturbation_oom_blocked(p):
            _STATE["pert_oom_owner"] = None
        _STATE["pass_owner"] = _owner_ref(p)

        def _xyz_num(key, cur):
            if key in xyz:
                try:
                    value = float(xyz[key])
                    return value if math.isfinite(value) else cur
                except (TypeError, ValueError):
                    return cur
            return cur

        def _arg(i, default):
            return args[i] if len(args) > i else default

        # ---- Perturbation enable (checkbox, XYZ can override for ON/OFF grids) ----
        try:
            pert_enabled = bool(_arg(0, False))
        except Exception:
            pert_enabled = False
        if "enabled" in xyz:
            pert_enabled = _as_bool(xyz["enabled"], pert_enabled)

        # ---- APG enable ----
        try:
            apg_enabled = bool(_arg(12, False))
        except Exception:
            apg_enabled = False
        if "apg_enabled" in xyz:
            apg_enabled = _as_bool(xyz["apg_enabled"], apg_enabled)

        # ---- Adaptive Guidance enable ----
        try:
            adg_enabled = bool(_arg(17, False))
        except Exception:
            adg_enabled = False
        if "adg_enabled" in xyz:
            adg_enabled = _as_bool(xyz["adg_enabled"], adg_enabled)

        # ---- New orchestrator modes (appended UI args preserve old indexes) ----
        raw_cfg_mode = str(
            xyz["cfg_mode"] if "cfg_mode" in xyz
            else _arg(22, "Preserve incoming")
        ).strip().lower()
        cfg_mode_map = {
            "preserve incoming": "preserve",
            "preserve": "preserve",
            "apg": "apg",
            "cwm": "cwm",
            "smc": "smc",
            "smc + cwm": "smc+cwm",
            "smc+cwm": "smc+cwm",
        }
        cfg_mode = cfg_mode_map.get(raw_cfg_mode, "preserve")
        experimental_stack = (
            _as_bool(xyz["cfg_stack"], _as_bool(_arg(23, False), False))
            if "cfg_stack" in xyz
            else _as_bool(_arg(23, False), False)
        )
        if cfg_mode == "preserve" and apg_enabled:
            cfg_mode = "apg"  # backwards-compatible quick checkbox
        legacy_smc_enabled = (
            _as_bool(xyz["smc_enabled"], _as_bool(_arg(42, False), False))
            if "smc_enabled" in xyz
            else _as_bool(_arg(42, False), False)
        )
        smc_preset = normalize_smc_preset(
            xyz["smc_preset"]
            if "smc_preset" in xyz
            else _arg(56, "Off")
        )
        cwm_enabled = (
            _as_bool(xyz["cwm_enabled"], _as_bool(_arg(43, False), False))
            if "cwm_enabled" in xyz
            else _as_bool(_arg(43, False), False)
        )
        dcw_enabled = (
            _as_bool(xyz["dcw_enabled"], _as_bool(_arg(28, False), False))
            if "dcw_enabled" in xyz
            else _as_bool(_arg(28, False), False)
        )
        rdc_enabled = (
            _as_bool(xyz["rdc_enabled"], _as_bool(_arg(58, False), False))
            if "rdc_enabled" in xyz
            else _as_bool(_arg(58, False), False)
        )
        dave_enabled = (
            _as_bool(xyz["dave_enabled"], _as_bool(_arg(31, False), False))
            if "dave_enabled" in xyz
            else _as_bool(_arg(31, False), False)
        )
        cns_enabled = (
            _as_bool(xyz["cns_enabled"], _as_bool(_arg(35, False), False))
            if "cns_enabled" in xyz
            else _as_bool(_arg(35, False), False)
        )
        mod_enabled = (
            _as_bool(xyz["mod_enabled"], _as_bool(_arg(44, False), False))
            if "mod_enabled" in xyz
            else _as_bool(_arg(44, False), False)
        )
        resolved_apg = apg_enabled or cfg_mode == "apg" or experimental_stack
        legacy_smc_requested = (
            legacy_smc_enabled
            or experimental_stack
            or cfg_mode in {"smc", "smc+cwm"}
        )
        smc_master_is_explicit = "smc_enabled" in xyz or len(args) > 57
        smc_master_enabled = (
            _as_bool(xyz["smc_enabled"], False)
            if "smc_enabled" in xyz
            else _as_bool(_arg(57, False), False)
        )
        # Before the appended master checkbox existed, choosing a non-Off
        # preset itself enabled SMC. Preserve that contract for short API/
        # infotext argument lists; new UI calls use the explicit checkbox.
        resolved_smc = legacy_smc_requested or (
            smc_master_enabled
            if smc_master_is_explicit
            else smc_preset != "Off"
        )
        effective_smc_preset = (
            (smc_preset if smc_preset != "Off" else "Custom")
            if resolved_smc else "Off"
        )
        resolved_cwm = (
            cwm_enabled or experimental_stack or cfg_mode in {"cwm", "smc+cwm"}
        )

        _STATE.update(
            requested_pert=pert_enabled,
            requested_apg=resolved_apg,
            requested_adg=adg_enabled,
            requested_cfg_mode=cfg_mode,
            requested_cfg_stack=experimental_stack,
            requested_smc=resolved_smc,
            requested_cwm=resolved_cwm,
            requested_dcw=dcw_enabled,
            requested_rdc=rdc_enabled,
            requested_dave=dave_enabled,
            requested_cns=cns_enabled,
            requested_modulation=mod_enabled,
        )

        if not any((
            pert_enabled,
            adg_enabled,
            cfg_mode != "preserve",
            experimental_stack,
            resolved_smc,
            resolved_cwm,
            dcw_enabled,
            rdc_enabled,
            dave_enabled,
            cns_enabled,
            mod_enabled,
        )):
            return

        # ---- Read every knob (with XYZ overrides) ----
        try:
            method_str = str(xyz["method"] if "method" in xyz else _arg(1, "PAG")).strip().lower()
            attn_method = method_str if method_str in ("pag", "seg") else None
            _STATE["requested_method"] = attn_method
            scale = _xyz_num("scale", float(_arg(2, 4.0)))
            legacy_strength = _xyz_num(
                "legacy_strength", float(_arg(3, 0.75))
            )
            official_strength = _xyz_num(
                "strength", float(_arg(39, 0.75))
            )
            # Before the dedicated Legacy axis existed, the generic
            # Perturbation Strength axis was the only strength control. Keep it
            # effective in either mode for saved XYZ presets.
            if "strength" in xyz and "legacy_strength" not in xyz:
                legacy_strength = official_strength
            block_spec = str(xyz["blocks"]) if "blocks" in xyz else str(_arg(4, "18"))
            head_spec = (
                str(xyz["heads"])
                if "heads" in xyz else str(_arg(40, ""))
            )
            slg_on = _as_bool(xyz["slg_enabled"], _as_bool(_arg(5, False), False)) \
                if "slg_enabled" in xyz else _as_bool(_arg(5, False), False)
            slg_scale = _xyz_num("slg_scale", float(_arg(6, 3.0)))
            slg_block_spec = str(xyz["slg_blocks"]) if "slg_blocks" in xyz else str(_arg(7, "18"))
            start = _xyz_num("start", float(_arg(8, 0.0)))
            end = _xyz_num("end", float(_arg(9, 0.7)))
            rescale = _xyz_num("rescale", float(_arg(10, 0.20)))
            rescale_mode = str(
                xyz["rescale_mode"]
                if "rescale_mode" in xyz else _arg(41, "full")
            ).strip().lower()
            # arg index 11 (formerly auto_decay) is now an inert hidden
            # placeholder — the safety brake was removed, so it is not read.
            apg_eta = _xyz_num("apg_eta", float(_arg(13, 0.0)))
            apg_norm = _xyz_num("apg_norm", float(_arg(14, 15.0)))
            apg_momentum = _xyz_num("apg_momentum", float(_arg(15, 0.0)))
            apg_autooff = _as_bool(_arg(16, True), True)
            adg_start = _xyz_num("adg_start", float(_arg(18, 0.5)))
            adg_interval = _xyz_num("adg_interval", float(_arg(19, 0)))
            legacy_attn = _as_bool(xyz["legacy_attn"], _as_bool(_arg(20, False), False)) \
                if "legacy_attn" in xyz else _as_bool(_arg(20, False), False)
            seg_sigma = _xyz_num("seg_sigma", float(_arg(21, 100.0)))
            cwm_alpha_low = _xyz_num("cwm_alpha_low", float(_arg(24, 0.30)))
            cwm_alpha_high = _xyz_num("cwm_alpha_high", float(_arg(25, 0.15)))
            smc_lambda = _xyz_num("smc_lambda", float(_arg(26, 6.0)))
            # Index 27 predates the appended preset selector. Keep its omitted
            # API/infotext fallback at the historical 0.20; the new Custom UI
            # still supplies its explicit upstream default of 0.10.
            smc_k = _xyz_num("smc_k", float(_arg(27, 0.20)))
            dcw_lambda_low = _xyz_num(
                "dcw_lambda_low", float(_arg(29, 0.10))
            )
            dcw_lambda_high = _xyz_num(
                "dcw_lambda_high", float(_arg(30, 0.02))
            )
            rdc_tau = _xyz_num("rdc_tau", float(_arg(59, 0.15)))
            rdc_alpha_ll = _xyz_num(
                "rdc_alpha_ll", float(_arg(60, 0.03))
            )
            rdc_alpha_hh = _xyz_num(
                "rdc_alpha_hh", float(_arg(61, 0.0))
            )
            dave_strength = _xyz_num(
                "dave_strength", float(_arg(32, 0.30))
            )
            dave_tau = _xyz_num("dave_tau", float(_arg(33, 0.10)))
            dave_block_spec = (
                str(xyz["dave_blocks"])
                if "dave_blocks" in xyz else str(_arg(34, "8-18"))
            )
            cns_strength = _xyz_num(
                "cns_strength", float(_arg(36, 1.0))
            )
            cns_gamma_power = _xyz_num(
                "cns_gamma_power", float(_arg(37, 0.5))
            )
            cns_gamma_scale = _xyz_num(
                "cns_gamma_scale", float(_arg(38, 3.0))
            )
            mod_clip_model = str(_arg(45, "") or "").strip()
            mod_weight = _xyz_num(
                "mod_weight", float(_arg(46, 3.0))
            )
            mod_start_layer = _xyz_num(
                "mod_start_layer", float(_arg(47, 0))
            )
            mod_end_layer = _xyz_num(
                "mod_end_layer", float(_arg(48, -1))
            )
            mod_base_source = str(
                _arg(49, "Main positive")
            ).strip().lower()
            mod_base_custom = str(_arg(50, "") or "")
            mod_positive_prompt = str(
                _arg(51, "masterpiece, best quality, highres") or ""
            )
            mod_negative_source = str(
                _arg(52, "Main negative")
            ).strip().lower()
            mod_negative_custom = str(
                _arg(53, "worst quality, low quality") or ""
            )
            mod_adapter_mode = str(
                _arg(54, "Auto-download official") or
                "Auto-download official"
            )
            mod_adapter_path = str(_arg(55, "") or "")
        except Exception as e:
            _STATE["on"] = False
            _APG["on"] = False
            _ADG["on"] = False
            _log(f"bad args, disabling: {type(e).__name__}: {e}")
            return

        # Gradio sliders constrain interactive values, but XYZ axes accept
        # arbitrary strings (including NaN/Inf). Keep the same safe domains
        # when values arrive through XYZ or an API client.
        scale = _finite_clamp(scale, 0.0, 15.0, 4.0)
        legacy_strength = _finite_clamp(
            legacy_strength, 0.0, 1.0, 0.75
        )
        official_strength = _finite_clamp(
            official_strength, 0.0, 1.0, 0.75
        )
        strength = legacy_strength if legacy_attn else official_strength
        slg_scale = _finite_clamp(slg_scale, 0.0, 15.0, 3.0)
        start = _finite_clamp(start, 0.0, 1.0, 0.0)
        end = _finite_clamp(end, 0.0, 1.0, 0.7)
        rescale = _finite_clamp(rescale, 0.0, 1.0, 0.20)
        rescale_mode = (
            rescale_mode if rescale_mode in {"full", "partial"} else "full"
        )
        apg_eta = _finite_clamp(apg_eta, -10.0, 10.0, 0.0)
        apg_norm = _finite_clamp(apg_norm, 0.0, 50.0, 15.0)
        apg_momentum = _finite_clamp(apg_momentum, -1.0, 1.0, 0.0)
        adg_start = _finite_clamp(adg_start, 0.0, 1.0, 0.5)
        adg_interval = int(_finite_clamp(adg_interval, 0.0, 10.0, 0.0))
        seg_sigma = _finite_clamp(seg_sigma, 0.0, 10000.0, 100.0)
        cwm_alpha_low = _finite_clamp(cwm_alpha_low, -1.0, 1.0, 0.30)
        cwm_alpha_high = _finite_clamp(cwm_alpha_high, -1.0, 1.0, 0.15)
        # Keep 0 valid for legacy/API neutral values even though the upstream
        # Custom UI starts lambda at 0.5.
        smc_lambda = _finite_clamp(smc_lambda, 0.0, 30.0, 6.0)
        smc_k = _finite_clamp(smc_k, 0.0, 5.0, 0.20)
        dcw_lambda_low = _finite_clamp(dcw_lambda_low, -0.5, 0.5, 0.10)
        dcw_lambda_high = _finite_clamp(dcw_lambda_high, -0.5, 0.5, 0.02)
        rdc_tau = _finite_clamp(rdc_tau, 0.0, 0.5, 0.15)
        rdc_alpha_ll = _finite_clamp(rdc_alpha_ll, 0.0, 0.3, 0.03)
        rdc_alpha_hh = _finite_clamp(rdc_alpha_hh, 0.0, 0.1, 0.0)
        dave_strength = _finite_clamp(dave_strength, 0.0, 1.0, 0.30)
        dave_tau = _finite_clamp(dave_tau, 0.0, 1.0, 0.10)
        cns_strength = _finite_clamp(cns_strength, 0.0, 1.0, 1.0)
        cns_gamma_power = _finite_clamp(cns_gamma_power, 0.05, 2.0, 0.5)
        cns_gamma_scale = _finite_clamp(cns_gamma_scale, 0.25, 25.0, 3.0)
        mod_weight = _finite_clamp(mod_weight, -20.0, 20.0, 3.0)
        mod_start_layer = int(
            _finite_clamp(mod_start_layer, 0.0, 1024.0, 0.0)
        )
        mod_end_layer = int(
            _finite_clamp(mod_end_layer, -1.0, 1024.0, -1.0)
        )

        requested_start, requested_end = min(start, end), max(start, end)
        effective_end = requested_end
        range_mode = "continuous"
        if adg_enabled and adg_interval == 0:
            effective_end = min(requested_end, adg_start)
            if effective_end < requested_end:
                range_mode = "continuous-cut-by-adaptive"
        elif adg_enabled and adg_interval > 0 and requested_end >= adg_start:
            range_mode = f"pulsed-keep-every-{adg_interval}"

        _CFG.update(
            smc_on=resolved_smc,
            apg_on=resolved_apg,
            cwm_on=resolved_cwm,
            mode=cfg_mode,
            experimental_stack=experimental_stack,
            alpha_low=cwm_alpha_low,
            alpha_high=cwm_alpha_high,
            smc_preset=effective_smc_preset,
            smc_resolved_preset=effective_smc_preset,
            smc_lambda=smc_lambda,
            smc_k=smc_k,
        )
        _DCW.update(
            on=dcw_enabled or rdc_enabled,
            dcw_on=dcw_enabled,
            rdc_on=rdc_enabled,
            lambda_low=dcw_lambda_low,
            lambda_high=dcw_lambda_high,
            rdc_tau=rdc_tau,
            rdc_alpha_ll=rdc_alpha_ll,
            rdc_alpha_hh=rdc_alpha_hh,
        )
        _CNS.update(
            on=cns_enabled,
            strength=cns_strength,
            gamma_power=cns_gamma_power,
            gamma_scale=cns_gamma_scale,
        )
        if cwm_alpha_high > 0.15 and resolved_cwm:
            _log(
                "CWM alpha_high > +0.15 on Anima can split one character "
                "into multiple subjects; this is a warning, not a clamp."
            )
        if cns_enabled and adg_enabled and adg_start < 0.65:
            _log(
                "CNS + Adaptive Guidance: skip_after < 0.65 may remove "
                "conditioning too early; start at 0.65 or later for A/B."
            )

        sd_model = getattr(p, "sd_model", None)
        engine = type(sd_model).__name__ if sd_model is not None else "?"
        _STATE["engine"] = engine
        (
            _smc_requested_preset,
            smc_resolved_preset,
            smc_lambda,
            smc_k,
        ) = resolve_smc_preset(
            effective_smc_preset,
            sd_model,
            custom_lambda=smc_lambda,
            custom_k=smc_k,
        )
        smc_preset_label = (
            "Off"
            if not resolved_smc
            else (
                "Legacy Custom"
                if smc_preset == "Off" and legacy_smc_requested
                else (
                    f"Auto→{smc_resolved_preset}"
                    if smc_preset == "Auto"
                    else smc_resolved_preset
                )
            )
        )
        _CFG.update(
            smc_preset=effective_smc_preset,
            smc_resolved_preset=smc_resolved_preset,
            smc_lambda=smc_lambda,
            smc_k=smc_k,
        )
        if resolved_smc and smc_preset == "Auto":
            _log(
                f"SMC Auto detected {smc_resolved_preset}: "
                f"lambda={smc_lambda:g}, k={smc_k:g}"
            )
        forge_objects = getattr(sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        if unet is None:
            _STATE["on"] = False
            _APG["on"] = False
            _log("no forge_objects.unet — cannot attach guidance.")
            return

        # ---- APG: model-agnostic, needs no patching. Reset momentum buffer. ----
        if resolved_apg:
            _APG.update(
                on=True, eta=apg_eta, norm_threshold=apg_norm,
                momentum=apg_momentum, avg=None, last_sigma=None,
            )
        else:
            _APG["on"] = False
        _STATE["apg_autooff_rescale"] = apg_autooff

        # ---- Adaptive Guidance: model-agnostic, needs no patching. ----
        if adg_enabled:
            _ADG.update(on=True, start=adg_start, interval=max(0, adg_interval))
        else:
            _ADG["on"] = False

        # ---- Anima block features: perturbation, DAVE, CLIP modulation. ----
        pert_ok = False
        dave_ok = False
        mod_ok = False
        attn_targets: set = set()
        slg_targets: set = set()
        dave_targets: set = set()
        mod_targets: set = set()
        nblocks = 0
        if pert_enabled or dave_enabled or mod_enabled:
            if engine != "Anima":
                if pert_enabled:
                    _log(
                        f"engine={engine} (not 'Anima') — perturbation skipped."
                    )
                if dave_enabled:
                    _log(f"engine={engine} (not 'Anima') — DAVE skipped.")
                if mod_enabled:
                    _log(
                        f"engine={engine} (not 'Anima') — CLIP modulation skipped."
                    )
            else:
                dm = _get_diffusion_model(unet)
                if dm is None:
                    _log(
                        "could not resolve diffusion_model — Anima block "
                        "features skipped."
                    )
                else:
                    blocks = getattr(dm, "blocks", None)
                    nblocks = len(blocks) if blocks is not None else 0
                    if nblocks and mod_enabled:
                        normalized_end = (
                            nblocks - 1
                            if mod_end_layer < 0
                            else min(mod_end_layer, nblocks - 1)
                        )
                        normalized_start = min(
                            max(mod_start_layer, 0),
                            nblocks - 1,
                        )
                        if normalized_start > normalized_end:
                            _log(
                                "CLIP modulation has an empty block range "
                                f"({mod_start_layer}..{mod_end_layer}); skipped."
                            )
                        else:
                            if not mod_clip_model:
                                choices = _clip_model_choices()
                                mod_clip_model = choices[0] if choices else ""
                            main_positive = _first_processing_prompt(
                                p,
                                negative=False,
                            )
                            main_negative = _first_processing_prompt(
                                p,
                                negative=True,
                            )
                            base_prompt = (
                                mod_base_custom
                                if mod_base_source == "custom"
                                else main_positive
                            )
                            negative_prompt = (
                                mod_negative_custom
                                if mod_negative_source == "custom"
                                else main_negative
                            )
                            try:
                                forge_root, models_root = (
                                    _forge_and_models_roots()
                                )
                                prepared_at = time.perf_counter()
                                block_modulations, mod_meta = (
                                    prepare_block_modulations(
                                        forge_root=forge_root,
                                        models_root=models_root,
                                        clip_model=mod_clip_model,
                                        prompts=(
                                            base_prompt,
                                            mod_positive_prompt,
                                            negative_prompt,
                                        ),
                                        adapter_mode=mod_adapter_mode,
                                        adapter_path=mod_adapter_path,
                                        diffusion_model=dm,
                                        weight=mod_weight,
                                    )
                                )
                                mod_targets = set(
                                    range(
                                        normalized_start,
                                        normalized_end + 1,
                                    )
                                )
                                _MOD.update(
                                    on=True,
                                    weight=mod_weight,
                                    targets=mod_targets,
                                    block_modulations=block_modulations,
                                    typed={},
                                    hits=0,
                                    warned=False,
                                    clip_model=mod_meta["clip_path"],
                                    adapter_path=mod_meta["adapter_path"],
                                    start_layer=normalized_start,
                                    end_layer=normalized_end,
                                    base_prompt=base_prompt,
                                    positive_prompt=mod_positive_prompt,
                                    negative_prompt=negative_prompt,
                                )
                                mod_ok = True
                                _log(
                                    "CLIP modulation prepared ✅ "
                                    f"clip={Path(mod_meta['clip_path']).name} "
                                    f"w={mod_weight:g} blocks="
                                    f"{normalized_start}-{normalized_end} "
                                    f"({time.perf_counter() - prepared_at:.2f}s)"
                                )
                            except Exception as exc:
                                _MOD["on"] = False
                                _MOD["block_modulations"] = None
                                _log(
                                    "CLIP modulation disabled for this "
                                    "generation: "
                                    f"{type(exc).__name__}: {exc}"
                                )

                    if nblocks and (pert_enabled or dave_enabled or mod_ok):
                        nblocks = _ensure_patched(dm)
                        if not nblocks and mod_ok:
                            mod_ok = False
                            _MOD["on"] = False
                            _MOD["block_modulations"] = None
                    if nblocks:
                        if pert_enabled:
                            if attn_method and scale > 0:
                                attn_targets = _parse_blocks(block_spec, nblocks)
                            if slg_on and slg_scale > 0:
                                slg_targets = _parse_blocks(
                                    slg_block_spec, nblocks
                                )
                            pert_ok = bool(
                                (attn_method and attn_targets)
                                or (slg_on and slg_targets)
                            )
                            if not pert_ok:
                                _log(
                                    "no valid target blocks — perturbation skipped."
                                )
                        if dave_enabled and dave_strength > 0:
                            dave_targets = _parse_blocks(
                                dave_block_spec, nblocks
                            )
                            dave_ok = bool(dave_targets)
                            if not dave_ok:
                                _log("no valid DAVE blocks — DAVE skipped.")

        _DAVE.update(
            on=dave_ok,
            strength=dave_strength,
            tau=dave_tau,
            targets=dave_targets,
        )
        if not mod_ok:
            _MOD.update(
                on=False,
                targets=set(),
                block_modulations=None,
                typed={},
            )

        if pert_ok and _perturbation_oom_blocked(p):
            pert_ok = False
            _log(
                "PAG/SEG/SLG 확장 배치가 이 생성의 앞선 패스에서 VRAM 부족(OOM)"
                "으로 실패해 perturbation 을 끈 상태를 유지합니다."
            )

        if pert_ok:
            _STATE.update(
                on=True, attn_method=(attn_method if attn_targets else None),
                attn_scale=scale, strength=strength, legacy_attn=legacy_attn,
                seg_sigma=seg_sigma, head_spec=head_spec,
                attn_targets=attn_targets,
                slg_on=bool(slg_on and slg_targets), slg_scale=slg_scale,
                slg_targets=slg_targets, rescale=rescale,
                rescale_mode=rescale_mode,
                start=requested_start, end=effective_end,
                requested_start=requested_start, requested_end=requested_end,
                range_mode=range_mode,
                active=0, attn_raw=None, slg_raw=None, cond_raw=None,
                attn_hook_hits=0, attn_last_rel_delta=None,
                attn_diag_logged=False, attn_shape_warned=False,
                attn_spatial_shape=None, adg_skipped=False,
                step_open=False,
            )
            _clear_markers()
        else:
            _STATE["on"] = False

        if not any((
            _STATE["on"],
            _APG["on"],
            _ADG["on"],
            *_cfg_base_flags(),
            _DCW["on"],
            _DAVE["on"],
            _CNS["on"],
            _MOD["on"],
        )):
            return

        # Step counter / total drive both perturbation range and AG gating.
        _STATE["total"] = int(getattr(p, "steps", 20) or 20)
        _STATE["step"] = 0
        _STATE["step_open"] = False

        # ---- Attach hooks (clone so Forge core / other gens are untouched) ----
        try:
            if _CNS["on"] and not _ensure_cns_noise_patched():
                _CNS["on"] = False
            unet = unet.clone()
            # The wrapper is needed for perturbation AND for Adaptive Guidance
            # (both manipulate the cond/uncond batch before apply_model).
            if _STATE["on"] or _ADG["on"]:
                _warn_foreign_unet_wrapper(unet)
                unet.set_model_unet_function_wrapper(_model_wrapper)
            if _STATE["on"]:
                unet.set_model_sampler_pre_cfg_function(_prepare_condition_aggregation)
            unet.set_model_sampler_post_cfg_function(_post_cfg)
            p.sd_model.forge_objects.unet = unet

            if not hasattr(p, "extra_generation_params"):
                p.extra_generation_params = {}
            if _STATE["on"]:
                parts = []
                if _STATE["attn_method"]:
                    mode = "legacy-soft/approx" if legacy_attn else "official"
                    detail = (
                        f"strength={strength}"
                        + (f" sigma={seg_sigma}" if attn_method == "seg" else "")
                    )
                    heads = head_spec.strip() or "all"
                    parts.append(f"{_STATE['attn_method'].upper()} mode={mode} scale={scale} "
                                 f"{detail} blocks={sorted(attn_targets)} heads={heads}")
                if _STATE["slg_on"]:
                    parts.append(f"SLG scale={slg_scale} skip={sorted(slg_targets)}")
                p.extra_generation_params["Anima Perturbation Guidance"] = (
                    "; ".join(parts)
                    + f"; requested_range={_STATE['requested_start']:.2f}-"
                      f"{_STATE['requested_end']:.2f}"
                    + f"; effective_range={_STATE['start']:.2f}-{_STATE['end']:.2f}"
                    + f"; range_mode={_STATE['range_mode']}"
                    + f"; rescale={rescale}({rescale_mode})"
                )
                # 결과를 미세하게 바꾸는 설정(재현성). 붙여넣기 때 같은 Forge 설정으로
                # 돌아가도록 OptionInfo(infotext=...) 이름·"True"/"False" 로 적는다.
                p.extra_generation_params[INFOTEXT_PREFIX_DEDUP] = str(
                    bool(_STATE["prefix_dedup"])
                )
                if (
                    _STATE["attn_method"] == "seg"
                    and not legacy_attn
                    and 0.0 < float(seg_sigma) <= 9999.0
                ):
                    p.extra_generation_params[INFOTEXT_SEG_SEPARABLE] = str(
                        bool(_STATE["seg_separable"])
                    )
            if _APG["on"]:
                p.extra_generation_params["Anima APG"] = (
                    f"eta={apg_eta}, norm={apg_norm}, momentum={apg_momentum}"
                )
            smc_on, apg_on, cwm_on = _cfg_base_flags()
            if smc_on or apg_on or cwm_on:
                active = [
                    name for name, on in
                    (("SMC", smc_on), ("APG", apg_on), ("CWM", cwm_on)) if on
                ]
                p.extra_generation_params["Anima CFG Orchestrator"] = (
                    "→".join(active)
                    + (f", alpha=({cwm_alpha_low},{cwm_alpha_high})" if cwm_on else "")
                    + (
                        f", smc={smc_preset_label}"
                        f"({smc_lambda:g},{smc_k:g})"
                        if smc_on else ""
                    )
                )
            if _DCW["dcw_on"]:
                p.extra_generation_params["Anima DCW"] = (
                    f"lambda_low={dcw_lambda_low}, "
                    f"lambda_high={dcw_lambda_high}"
                )
            if _DCW["rdc_on"]:
                p.extra_generation_params["Anima RDC"] = (
                    f"tau={rdc_tau}, alpha_ll={rdc_alpha_ll}, "
                    f"alpha_hh={rdc_alpha_hh}"
                )
            if _DAVE["on"]:
                p.extra_generation_params["Anima DAVE"] = (
                    f"strength={dave_strength}, tau={dave_tau}, "
                    f"blocks={sorted(dave_targets)}"
                )
            if _CNS["on"]:
                p.extra_generation_params["Anima CNS Wavelet Noise"] = (
                    f"strength={cns_strength}, gamma_power={cns_gamma_power}, "
                    f"gamma_scale={cns_gamma_scale}"
                )
            if _MOD["on"]:
                p.extra_generation_params["Anima Modulation Guidance"] = (
                    f"w={_MOD['weight']:g}, "
                    f"blocks={_MOD['start_layer']}-{_MOD['end_layer']}, "
                    f"clip={Path(_MOD['clip_model']).name}, "
                    f"adapter={Path(_MOD['adapter_path']).name}"
                )
            if _ADG["on"]:
                p.extra_generation_params["Anima Adaptive Guidance"] = (
                    f"skip_after={_ADG['start']:.2f}, keep_every={_ADG['interval']}"
                )
            _log(
                f"attached ✅ engine={engine} "
                f"pert={'on' if _STATE['on'] else 'off'} "
                f"(attn={_STATE['attn_method']}:{sorted(attn_targets)} scale={scale} "
                f"mode={'legacy' if legacy_attn else 'official'} "
                f"strength={strength} heads={head_spec.strip() or 'all'} "
                f"seg_sigma={seg_sigma} "
                f"slg={_STATE['slg_on']}:{sorted(slg_targets)} scale={slg_scale} "
                f"range={_STATE['requested_start']:.2f}-{_STATE['requested_end']:.2f}"
                f"→{_STATE['start']:.2f}-{_STATE['end']:.2f}"
                f"({_STATE['range_mode']}) "
                f"rescale={'auto-off' if (_APG['on'] and apg_autooff) else rescale}"
                f"({rescale_mode}) "
                f"prefix_dedup={bool(_STATE['prefix_dedup'])} "
                f"seg_separable={bool(_STATE['seg_separable'])}) "
                f"APG={'on' if _APG['on'] else 'off'} "
                f"(eta={apg_eta} norm={apg_norm} mom={apg_momentum}) "
                f"AdaptiveG={'on' if _ADG['on'] else 'off'} "
                f"(skip_after={_ADG['start']} keep_every={_ADG['interval']}) "
                f"CFGBase={_CFG['mode']} stack={_CFG['experimental_stack']} "
                f"SMC={smc_preset_label}"
                f"({smc_lambda:g},{smc_k:g}) "
                f"DCW={_DCW['dcw_on']} RDC={_DCW['rdc_on']} "
                f"DAVE={_DAVE['on']} CNS={_CNS['on']} "
                f"Mod={'on' if _MOD['on'] else 'off'}"
                + (
                    f"(w={_MOD['weight']:g}, "
                    f"blocks={_MOD['start_layer']}-{_MOD['end_layer']})"
                    if _MOD["on"] else ""
                )
            )
        except Exception as e:
            _STATE["on"] = False
            _APG["on"] = False
            _ADG["on"] = False
            _CFG["mode"] = "preserve"
            _CFG["experimental_stack"] = False
            _DCW.update(on=False, dcw_on=False, rdc_on=False)
            _DAVE["on"] = False
            _CNS["on"] = False
            _MOD.update(
                on=False,
                targets=set(),
                block_modulations=None,
                typed={},
            )
            _RUNTIME.reset_pass()
            _log(f"failed to attach hooks: {type(e).__name__}: {e}")

    def postprocess(self, p, processed, *args):
        # Belt-and-suspenders: make sure guidance doesn't leak into a later
        # generation if forge_objects reuse ever changed.
        if _STATE["wrapper_calls"]:
            _log(
                "generation summary: "
                f"wrapper_calls={_STATE['wrapper_calls']} "
                f"weak_steps={_STATE['weak_steps']} "
                f"applied_steps={_STATE['applied_steps']}"
                + (
                    f" prefix_dedup_blocks={_STATE['dedup_blocks']}"
                    f" prefix_dedup_fallbacks={_STATE['dedup_fallbacks']}"
                    if _STATE["weak_steps"] else ""
                )
            )
        verify_requested = any((
            _STATE["requested_pert"],
            _STATE["requested_apg"],
            _STATE["requested_adg"],
            _STATE["requested_smc"],
            _STATE["requested_cwm"],
            _STATE["requested_cfg_mode"] != "preserve",
            _STATE["requested_cfg_stack"],
            _STATE["requested_dcw"],
            _STATE["requested_dave"],
            _STATE["requested_cns"],
            _STATE["requested_modulation"],
        ))
        if guidance_diagnostics_enabled() and verify_requested:
            elapsed = None
            if _STATE["diag_started_at"] is not None:
                elapsed = max(0.0, time.perf_counter() - float(_STATE["diag_started_at"]))

            if not _STATE["requested_pert"]:
                pert_verdict = "OFF"
            elif _STATE["applied_steps"] > 0:
                pert_verdict = f"APPLIED({_STATE['applied_steps']} steps)"
            elif _STATE["control_blocked_calls"] > 0:
                pert_verdict = "NO-OP(ControlNet guard)"
            else:
                pert_verdict = "NO-OP"

            apg_verdict = (
                "OFF" if not _STATE["requested_apg"]
                else (f"APPLIED({_STATE['apg_steps']} steps)"
                      if _STATE["apg_steps"] > 0 else "NO-OP")
            )
            if not _STATE["requested_adg"]:
                adg_verdict = "OFF"
            elif _STATE["adg_skipped_steps"] > 0:
                adg_verdict = f"SKIPPED-UNCOND({_STATE['adg_skipped_steps']} steps)"
            elif _STATE["split_cond_calls"] or _STATE["split_uncond_calls"]:
                adg_verdict = "NO-SKIP(split/low-VRAM calls)"
            else:
                adg_verdict = "NO-SKIP"
            elapsed_text = f"{elapsed:.2f}s" if elapsed is not None else "?"

            _log(
                "[VERIFY] verdict: "
                f"perturb={pert_verdict}, APG={apg_verdict}, Adaptive={adg_verdict}; "
                f"engine={_STATE['engine']}, method={_STATE['requested_method']}"
            )
            _log(
                "[VERIFY] counters: "
                f"wrapper={_STATE['wrapper_calls']}, weak={_STATE['weak_steps']}, "
                f"combined={_STATE['combined_calls']}, "
                f"split_cond={_STATE['split_cond_calls']}, "
                f"split_uncond={_STATE['split_uncond_calls']}, "
                f"control_guard={_STATE['control_blocked_calls']}, "
                f"fallbacks={_STATE['wrapper_fallbacks']}, "
                f"elapsed={elapsed_text}"
            )

            if (
                not _STATE["requested_pert"]
                or _STATE["requested_method"] not in {"pag", "seg"}
            ):
                attention_verdict = "OFF"
            elif _STATE["attn_hook_hits_total"] <= 0:
                attention_verdict = "NO-HOOK"
            elif (
                _STATE["attn_last_rel_delta"] is None
                or _STATE["attn_last_rel_delta"] <= 1e-8
            ):
                attention_verdict = (
                    f"NEUTRAL({_STATE['attn_hook_hits_total']} hits)"
                )
            else:
                attention_verdict = (
                    f"ACTIVE({_STATE['attn_hook_hits_total']} hits, "
                    f"rel={_STATE['attn_last_rel_delta']:.3e})"
                )

            requested_parts = [
                name for name, requested in (
                    ("SMC", _STATE["requested_smc"]),
                    ("APG", _STATE["requested_apg"]),
                    ("CWM", _STATE["requested_cwm"]),
                )
                if requested
            ]
            requested_mode = "→".join(requested_parts) or "PRESERVE"
            cfg_requested = bool(requested_parts)
            cfg_verdict = (
                "OFF"
                if not cfg_requested
                else (
                    f"APPLIED({_CFG['steps']} evals)"
                    if _CFG["steps"] > 0 else "NO-OP"
                )
            )
            fit_text = (
                "?"
                if _CFG["fit_error"] is None
                else f"{float(_CFG['fit_error']):.3e}"
            )
            scale_text = (
                "?"
                if _CFG["effective_scale"] is None
                else f"{float(_CFG['effective_scale']):.4g}"
            )
            dcw_verdict = (
                "OFF"
                if not _STATE["requested_dcw"]
                else (
                    f"APPLIED({_DCW['dcw_steps']} evals)"
                    if _DCW["dcw_steps"] > 0 else "NO-OP"
                )
            )
            rdc_verdict = (
                "OFF"
                if not _STATE["requested_rdc"]
                else (
                    f"APPLIED({_DCW['rdc_steps']} evals)"
                    if _DCW["rdc_steps"] > 0 else "NO-OP"
                )
            )
            dave_verdict = (
                "OFF"
                if not _STATE["requested_dave"]
                else (
                    f"APPLIED({_DAVE['steps']} block hits)"
                    if _DAVE["steps"] > 0 else "NO-OP"
                )
            )
            cns_verdict = (
                "OFF"
                if not _STATE["requested_cns"]
                else (
                    f"APPLIED({_RUNTIME.cns_noise_calls} noise calls)"
                    if _RUNTIME.cns_noise_calls > 0
                    else "INERT(no ancestral/SDE noise call)"
                )
            )
            modulation_verdict = (
                "OFF"
                if not _STATE["requested_modulation"]
                else (
                    f"APPLIED({_MOD['hits']} block hits)"
                    if _MOD["hits"] > 0 else "NO-OP"
                )
            )
            _log(
                "[VERIFY] suite: "
                f"attention={attention_verdict}, "
                f"CFG={requested_mode}:{cfg_verdict} "
                f"(w_eff={scale_text}, fit={fit_text}), "
                f"DCW={dcw_verdict}, RDC={rdc_verdict}, "
                f"DAVE={dave_verdict}, CNS={cns_verdict}, "
                f"Modulation={modulation_verdict}"
            )
        _STATE["on"] = False
        _APG["on"] = False
        _ADG["on"] = False
        _CFG.update(
            smc_on=False,
            apg_on=False,
            cwm_on=False,
            mode="preserve",
            experimental_stack=False,
            smc_preset="Off",
            smc_resolved_preset="Off",
            steps=0,
            fit_error=None,
            effective_scale=None,
            external_cfg_detected=False,
            warned=False,
            cfg1_warned=False,
            base_skipped=False,
            fit_checked=False,
        )
        _DCW.update(
            on=False,
            dcw_on=False,
            rdc_on=False,
            steps=0,
            dcw_steps=0,
            rdc_steps=0,
        )
        _DAVE.update(on=False, targets=set(), steps=0)
        _CNS.update(on=False, warned=False)
        _MOD.update(
            on=False,
            targets=set(),
            block_modulations=None,
            typed={},
            hits=0,
            warned=False,
        )
        # Drop the stashed latent-sized tensors so they don't sit in VRAM
        # between generations (they'd otherwise linger until the next gen).
        _STATE["attn_raw"] = None
        _STATE["slg_raw"] = None
        _STATE["cond_raw"] = None
        _STATE.pop("condition_aggregation", None)
        _STATE["adg_skipped"] = False
        _STATE["attn_spatial_shape"] = None
        _STATE["attn_hook_hits"] = 0
        _STATE["attn_hook_hits_total"] = 0
        _STATE["attn_last_rel_delta"] = None
        _STATE["attn_diag_logged"] = False
        _STATE["attn_shape_warned"] = False
        _STATE["step_open"] = False
        _STATE["wrapper_calls"] = 0
        _STATE["weak_steps"] = 0
        _STATE["applied_steps"] = 0
        _STATE["apg_steps"] = 0
        _STATE["adg_skipped_steps"] = 0
        _STATE["combined_calls"] = 0
        _STATE["split_cond_calls"] = 0
        _STATE["split_uncond_calls"] = 0
        _STATE["control_blocked_calls"] = 0
        _STATE["wrapper_fallbacks"] = 0
        _STATE["requested_pert"] = False
        _STATE["requested_apg"] = False
        _STATE["requested_adg"] = False
        _STATE["requested_cfg_mode"] = "preserve"
        _STATE["requested_cfg_stack"] = False
        _STATE["requested_smc"] = False
        _STATE["requested_cwm"] = False
        _STATE["requested_dcw"] = False
        _STATE["requested_dave"] = False
        _STATE["requested_cns"] = False
        _STATE["requested_modulation"] = False
        _STATE["requested_method"] = None
        _STATE["engine"] = "?"
        _STATE["diag_started_at"] = None
        _STATE["delta_logged"] = False
        _STATE["pass_owner"] = None
        _STATE["pert_oom_owner"] = None
        _STATE["dedup_blocks"] = 0
        _STATE["dedup_fallbacks"] = 0
        _STATE["dedup_warned"] = False
        _STATE["dedup_disabled"] = False
        _INDEX_TENSOR_CACHE.clear()
        _RUNTIME.reset_pass()
        _clear_markers()
