"""Anima VAE DeGrid — 장치·정밀도·Forge 메모리 관리를 묶어 PIL 이미지 한 장을 처리한다.

- 장치(설정 ``sam3_degrid_device``): ``auto`` 는 Forge 가 쓰는 장치(``backend.memory_management.get_torch_device``,
  Forge 밖이면 CUDA 가 있을 때 CUDA), ``cpu`` 는 늘 CPU. ``backend`` 는 여기서 새로 import 하지 않는다 — Forge 에서는 이미
  올라와 있고, 테스트에서 import 하면 CUDA 를 조회한다.
- 정밀도(설정 ``sam3_degrid_gpu_precision``, 기본 ``fp32``): ``fp32`` 는 ComfyUI 노드와 같은 계산(``a.float()``)이다.
  합성곱의 TF32 는 torch 설정 그대로 둔다 — ``cudnn.allow_tf32`` 는 torch 기본이 켬이고 Forge 도 바꾸지 않으며(노드도 같은
  기본으로 돈다), 프로세스 전역 설정이라 여기서 바꾸면 Forge 모델에도 번진다. RTX 5090 실측(1216×1856, 타일 512)에서 TF32 와
  엄격 fp32 는 잔차가 최대 0.11/255 다르고 속도는 같았다(둘 다 0.275 초). NAFNet 에는 행렬곱(Linear)이 없어
  ``matmul.allow_tf32`` 는 상관없다. CUDA 에서 ``fp16`` 을 고르면 가중치는 fp32 로 두고 ``torch.autocast(fp16)`` 로 돈다 —
  모델이 fp16 AMP 로 학습됐고(NAFNet-c QDG 설정 ``use_amp: true, amp_dtype: fp16``), autocast 는 LayerNorm 의 제곱·평균을
  fp32 로 두고 마지막 ``x + inp`` 도 fp32 로 더한다. 값이 넘친 타일(inf/NaN)은 그 타일만 fp32 로 다시 한다. 같은 GPU 에서
  fp16 autocast 가 더 느리고(0.362 초 대 0.275 초) VRAM 도 거의 같아(타일 512 최대 활성 276 대 292 MiB) 기본이 아니다 —
  옛 GPU 에서는 더 빠를 수 있어 선택지로 남긴다(측정 안 함). fp32 와의 잔차 차이는 GPU 최대 0.32~0.43/255(8비트로 픽셀
  2~18% 가 1 단계), CPU 최대 0.27/255. CPU 는 늘 fp32. bf16 은 쓰지 않는다(CPU 실측: 잔차 오차가 평균 0.36/255 로
  잔차 크기와 비슷, fp16 은 0.04/255).
- 메모리: GPU 로 돌 때는 ComfyUI 노드(``load_models_gpu([patcher], memory_required, force_full_load=True)``)·이 확장의
  VAE 2x 처럼 Forge ``ModelPatcher`` 로 감싸 ``load_models_gpu`` 로 올린다 — 모자라면 Forge 가 다른 모델을 내리고, 이 모델도
  Forge 목록에 들어간다. 끝나면 목록에서 빼고 ``model_unload``(→ CPU 로) + ``soft_empty_cache`` 로 VRAM 을 돌려준다.
  설정 ``sam3_degrid_keep_loaded`` 를 켜면 그대로 두어 Forge 가 자리가 필요할 때 내린다. Forge 가 부분 로드만 한 경우
  (``--novram`` 등 — 일반 ``nn.Conv2d`` 는 수동 캐스트가 없어 CPU 에 남음)에는 직접 옮긴다(약 117 MB).
- 모델 생성과 직접 하는 장치 이동은 ``torch.inference_mode(False)`` 에서, 계산은 ``inference_mode`` 안에서 한다(Forge 는
  이미지 후처리를 inference_mode 안에서 부른다 — ``model_for`` 참고). ``load_models_gpu`` 는 감싸지 않고 부른 문맥 그대로
  부른다 — 그 안에서 Forge 가 inference_mode 안에서 만든 자기 모델을 내릴 수 있다(``_acquire`` 참고).
- 타일은 모델의 ``padder_size``(NAFNet-small 16) 배수로 반사 패딩해 넣고 자른다 — spandrel NAFNet 은 안에서 0 으로 채워
  16 배수가 아닌 크기의 오른쪽·아래 가장자리에 큰 잔차를 낸다. 16 배수면 그대로라 노드 팩과 같다.
- 모델 출력이 잔차가 아니라 이미지처럼 보이면(입력을 따라감 — 일반 복원 NAFNet 을 고른 경우) ``NotResidualModelError`` 로
  거절하고, 잔차가 폭주하면(|평균| 100/255 초과 — 화면을 채운 1px 줄무늬 같은 무늬) ``ResidualBlowUpError`` 로 그 이미지를
  건너뛴다(``vae_degrid.check_residual``). 둘 다 ``DegridSkipError`` — 원본을 그대로 두고 infotext 에는 문구만 남는다.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field

import torch
from PIL import Image

from . import vae_degrid as vd
from .vae_degrid_models import ModelEntry, load_nafnet

OPT_DEVICE = "sam3_degrid_device"
# 예전 키 ``sam3_degrid_precision``(기본 fp16)은 읽지 않는다 — Forge 는 ``add_option`` 때 기본값을 ``opts.data`` 에 넣어
# 설정을 저장할 때 config.json 에 함께 적으므로(``modules/options.py``), 같은 키로 기본만 fp32 로 바꾸면 이미 쓰던 설치에는
# 저장된 fp16 이 그대로 남는다. fp16 autocast 를 쓰려면 Settings 에서 다시 고른다.
OPT_PRECISION = "sam3_degrid_gpu_precision"
LEGACY_OPT_PRECISION = "sam3_degrid_precision"
OPT_KEEP_LOADED = "sam3_degrid_keep_loaded"

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"
DEFAULT_DEVICE = DEVICE_AUTO
DEFAULT_PRECISION = PRECISION_FP32
DEFAULT_KEEP_LOADED = False

# load_models_gpu 의 memory_required — 타일 픽셀당 바이트. 최대 활성 실측: CPU fp32 1148 B/px(타일 256·512 같음), GPU
# (RTX 5090, 타일 512) fp32 292 MiB = 1168 B/px · fp16 autocast 276 MiB. 여기에 cudnn 작업 공간 여유를 더했다. 타일 512 →
# 384 MiB.
BYTES_PER_PIXEL = 1536

LOG_PREFIX = "[AnimaDeGrid]"


def log(message: str) -> None:
    """생성 경로에서 부르므로 cp949 콘솔의 UnicodeEncodeError 로 생성을 죽이지 않는다."""
    text = f"{LOG_PREFIX} {message}"
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "backslashreplace").decode(encoding, "replace"))
        except Exception:
            pass
    except Exception:
        pass


def read_option(name: str, default):
    try:
        from modules import shared

        value = getattr(shared.opts, name, default)
    except Exception:
        return default
    return default if value is None else value


def _forge_memory():
    """Forge 메모리 관리 — ``(backend.memory_management, ModelPatcher)``. 이미 import 된 경우만(Forge 밖이면 None)."""
    memory_management = sys.modules.get("backend.memory_management")
    if memory_management is None:
        return None
    try:
        from backend.patcher.base import ModelPatcher
    except Exception:
        return None
    return memory_management, ModelPatcher


def choose_device(requested) -> torch.device:
    if str(requested or "").strip().lower() == DEVICE_CPU:
        return torch.device("cpu")
    memory_management = sys.modules.get("backend.memory_management")
    getter = getattr(memory_management, "get_torch_device", None)
    if callable(getter):
        try:
            return torch.device(getter())
        except Exception:
            pass
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def use_fp16_autocast(device, requested) -> bool:
    """CUDA 에서 ``fp16`` 을 고른 경우만 fp16 autocast. 그 밖(기본 fp32·CPU·모르는 값)은 fp32."""
    return torch.device(device).type == "cuda" and str(requested or "").strip().lower() == PRECISION_FP16


def is_oom(exc: BaseException) -> bool:
    memory_management = sys.modules.get("backend.memory_management")
    checker = getattr(memory_management, "is_oom", None)
    if callable(checker):
        try:
            if checker(exc):
                return True
        except Exception:
            pass
    return vd.is_oom_error(exc)


def memory_estimate(height: int, width: int, tile: int) -> float:
    if tile > 0:
        height, width = min(height, tile), min(width, tile)
    return float(BYTES_PER_PIXEL * height * width)


class DegridSkipError(ValueError):
    """이 이미지에는 DeGrid 를 쓰지 않는다(원본 그대로) — 예상한 건너뜀이라 infotext·로그에는 예외 이름 없이 문구만
    (``failure_reason``, '모델 없음' 과 같은 모양)."""


class NotResidualModelError(DegridSkipError):
    """모델 출력이 잔차가 아니라 이미지다 — DeGrid 가 아닌 일반 NAFNet(노이즈 제거·디블러 등)을 고른 경우."""


class ResidualBlowUpError(DegridSkipError):
    """잔차가 폭주했다(|평균| > ``vd.RESIDUAL_BLOWUP_ABS_MEAN``) — 화면을 채운 1px 줄무늬 같은, 학습에 없던 무늬."""


def failure_reason(exc: BaseException) -> str:
    """infotext ``Anima DeGrid error``·로그에 남길 이유 — 예상한 건너뜀(``DegridSkipError``)은 문구만, 그 밖은 예외 이름을 붙인다."""
    if isinstance(exc, DegridSkipError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def not_residual_message(model_name: str, check: vd.ResidualCheck) -> str:
    """``NotResidualModelError`` 문구(infotext ``Anima DeGrid error`` 로도 남는다) — 무엇을 보고 거절했는지 숫자로."""
    if check.correlation is not None:
        correlation = f"{check.correlation:+.2f}"
    else:
        correlation = "n/a (single-color input)" if check.input_flat else "n/a"
    brightness = ""
    if check.follows_input_mean and check.dc_ratio is not None:
        brightness = (
            f"; mean output {check.signed_mean * 255:+.1f}/255 follows the input brightness "
            f"(x{check.dc_ratio:.2f} of the input mean)"
        )
    return (
        f"not a DeGrid residual model: {model_name} output does not look like a residual - it follows the input like "
        f"an image (mean |output| {check.mean_abs * 255:.1f}/255; correlation with the input {correlation}{brightness}) - "
        "use a VAE DeGrid NAFNet"
    )


def blow_up_message(model_name: str, check: vd.ResidualCheck) -> str:
    """``ResidualBlowUpError`` 문구 — infotext 는 ``Anima DeGrid error: output blew up (…)``."""
    return (
        f"output blew up (mean |residual| {check.mean_abs * 255:.1f}/255 > {vd.RESIDUAL_BLOWUP_ABS_MEAN * 255:.0f}/255 - "
        f"{model_name} does not handle this image, e.g. full-frame 1px stripes) - image kept without DeGrid"
    )


def model_pad_multiple(model) -> int:
    """모델이 안에서 0 으로 채워 맞추는 배수 — spandrel NAFNet ``padder_size``(= 2 ** 인코더 단 수, NAFNet-small 16)."""
    try:
        multiple = int(getattr(model, "padder_size", vd.PAD_MULTIPLE))
    except (TypeError, ValueError):
        multiple = vd.PAD_MULTIPLE
    return max(1, multiple)


def make_residual_fn(model, device: torch.device, use_autocast: bool):
    """타일 → 잔차(float32) 함수와 'fp16 이 넘쳐 fp32 로 다시 한 타일 수' 칸. 타일은 계산 장치로 옮겨 fp32 로 넣는다
    (노드 팩 ``a.float()``). 모델의 패딩 배수로 반사 패딩해 부르고 자른다(``vd.call_padded`` — 배수면 그대로).
    ``use_autocast`` 면 fp16 autocast 로 돌고, inf/NaN 이 나오면 그 타일만 autocast 없이 다시."""
    retiles = [0]
    multiple = model_pad_multiple(model)

    def residual(piece: torch.Tensor) -> torch.Tensor:
        piece = piece.to(device=device, dtype=torch.float32)
        if use_autocast:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = vd.call_padded(model, piece, multiple)
            if bool(torch.isfinite(out).all()):
                return out.float()
            retiles[0] += 1
        return vd.call_padded(model, piece, multiple).float()

    return residual, retiles


def release_after_run(keep_loaded: bool, device) -> bool:
    """이미지 하나를 끝낸 뒤 VRAM 에서 내릴지 — 남기기를 켠 GPU 실행만 남긴다(CPU 는 늘 CPU 에 있음)."""
    return (not keep_loaded) or torch.device(device).type == "cpu"


@dataclass
class DegridOutcome:
    image: Image.Image
    model_name: str
    mode: str
    strength: float
    tile: int
    tile_used: int
    device: str
    precision: str
    seconds: float
    stats: dict = field(default_factory=dict)
    fp32_retiles: int = 0
    skipped: bool = False

    def summary(self) -> str:
        if self.skipped:
            return f"강도 0 — 건너뜀 ({self.model_name})"
        tile = "없음" if self.tile_used <= 0 else str(self.tile_used)
        if self.tile_used != self.tile:
            tile += f" (OOM 으로 {self.tile} → {self.tile_used})"
        parts = [
            f"{self.model_name}", vd.MODE_LABELS.get(self.mode, self.mode), f"강도 {self.strength:g}",
            f"타일 {tile}", f"{self.device}/{self.precision}", f"{self.seconds:.2f}s",
        ]
        stats = self.stats or {}
        if "abs_mean_255" in stats:
            parts.append(f"잔차 |평균| {stats['abs_mean_255']:.2f}/255 · 최대 {stats['abs_max_255']:.1f}/255")
        if "changed_pixels" in stats:
            parts.append(f"바뀐 픽셀 {stats['changed_pixels'] * 100:.1f}%")
        if self.fp32_retiles:
            parts.append(f"fp16 이 넘쳐 fp32 로 다시 한 타일 {self.fp32_retiles}")
        return " · ".join(parts)


class DegridRuntime:
    def __init__(self, *, loader=None, forge_memory=None, logger=None):
        self._loader = loader or load_nafnet
        self._forge_memory = forge_memory or _forge_memory
        self._log = logger or log
        self._lock = threading.RLock()
        self._model = None
        self._model_key = None
        self._patcher = None
        self._stranded_logged = False

    # ── 모델 캐시 ──
    def model_for(self, path):
        """경로의 NAFNet(CPU fp32). 같은 파일(크기·수정 시각)이면 캐시를 쓰고, 다르면 앞 모델을 내리고 버린다.

        부르는 문맥과 상관없이 ``inference_mode(False)`` + ``no_grad`` 에서 만든다 — Forge 는 생성 탭 후처리
        (``postprocess_image_after_composite``)를 ``torch.inference_mode()`` 안에서 부르므로, 그대로 만들면 파라미터가
        inference 텐서가 된다. 그것을 GPU 로 옮기면(``load_models_gpu``·``_move`` → ``Module._apply`` 의 ``param.data =``)
        버전 카운터 없는 파라미터가 되어 다음 계산이 'Inference tensors do not track version counter.' 로 죽고, 캐시가
        파일 기준이라 Forge 를 다시 켤 때까지 모든 이미지가 실패한다(``anima38/runtime.py`` ``_outside_inference_mode`` 와
        같은 이유).
        """
        stat = os.stat(path)
        key = (os.path.abspath(str(path)), int(stat.st_size), int(stat.st_mtime_ns))
        with self._lock:
            if self._model is not None and self._model_key == key:
                return self._model
            self._release_locked()
            self._model = self._model_key = self._patcher = None
            with torch.inference_mode(False), torch.no_grad():
                model = self._loader(path)
            self._model, self._model_key = model, key
            return model

    @property
    def loaded_path(self) -> str | None:
        return self._model_key[0] if self._model_key else None

    # ── 장치 이동(테스트에서 바꿔 끼운다) ──
    @staticmethod
    def _move(model, device) -> None:
        with torch.inference_mode(False):
            model.to(device)

    @staticmethod
    def _off_device(model, device) -> bool:
        target = torch.device(device)
        for param in model.parameters():
            if param.device.type != target.type:
                return True
            if target.index is not None and param.device.index != target.index:
                return True
        return False

    def _empty_cache(self, memory) -> None:
        try:
            if memory is not None and callable(getattr(memory[0], "soft_empty_cache", None)):
                memory[0].soft_empty_cache()
            elif torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _acquire(self, model, device: torch.device, memory_required: float) -> None:
        """계산 장치에 올린다. CPU 면 CPU 로(앞서 GPU 에 남겨 둔 것이 있으면 먼저 내림)."""
        if device.type == "cpu":
            if self._off_device(model, device):
                self._release_locked()
            return
        memory = self._forge_memory()
        if memory is None:
            self._move(model, device)
            return
        memory_management, model_patcher = memory
        patcher = self._patcher
        if patcher is None or getattr(patcher, "model", None) is not model \
                or torch.device(getattr(patcher, "load_device", "cpu")) != device:
            if patcher is not None:
                self._release_locked()
            patcher = model_patcher(model, load_device=device, offload_device=torch.device("cpu"))
            self._patcher = patcher
        # 부른 문맥 그대로(scripts/anima_vae_2x.py ``_load_decoder`` 와 같음) — VRAM 이 모자라면 Forge 가 여기서 자기
        # UNet·TE·VAE 를 (일부) 내리는데, 이것들은 Forge 가 inference_mode 안에서 만들어 inference_mode(False) 로 감싸 옮기면
        # 다시 올릴 때까지 'Inference tensors do not track version counter.' 상태가 된다. 이 모델은 ``model_for`` 가
        # inference_mode 밖에서 만들어 어느 문맥에서 옮겨도 계산된다.
        memory_management.load_models_gpu([patcher], memory_required=memory_required, force_full_load=True)
        if self._off_device(model, device):
            if not self._stranded_logged:
                self._stranded_logged = True
                state = getattr(getattr(memory_management, "vram_state", None), "name", None)
                self._log(
                    f"Forge 가 모델을 {device} 에 다 올리지 않아(VRAM State: {state or '?'}) 직접 옮깁니다(약 117 MB). "
                    "이 안내는 한 번만 표시합니다."
                )
            self._move(model, device)

    def release(self) -> bool:
        """VRAM 에 남은 모델을 내린다(Forge 목록에서도 뺌). 내렸으면 True."""
        with self._lock:
            return self._release_locked()

    def _release_locked(self) -> bool:
        released = False
        patcher = self._patcher
        memory = self._forge_memory() if patcher is not None else None
        if memory is not None:
            loaded = getattr(memory[0], "current_loaded_models", None)
            for index, entry in enumerate(list(loaded or ())):
                if getattr(entry, "model", None) is patcher:
                    with torch.inference_mode(False):
                        loaded.pop(index)
                        entry.model_unload()   # → ModelPatcher.detach → 모델을 offload_device(CPU)로
                    released = True
                    break
        model = self._model
        if model is not None and self._off_device(model, "cpu"):
            self._move(model, torch.device("cpu"))
            released = True
        if released:
            self._empty_cache(memory if memory is not None else self._forge_memory())
        return released

    # ── 실행 ──
    def run(
        self,
        image: Image.Image,
        entry: ModelEntry,
        *,
        mode=vd.DEFAULT_MODE,
        strength=vd.DEFAULT_STRENGTH,
        tile=vd.DEFAULT_TILE,
        device=None,
        precision=None,
        keep_loaded=None,
    ) -> DegridOutcome:
        mode_key = vd.normalize_mode(mode) or vd.DEFAULT_MODE
        strength = vd.coerce_strength(strength)
        tile = vd.coerce_tile(tile)
        start = time.perf_counter()
        if strength == 0.0:
            return DegridOutcome(image.copy(), entry.name, mode_key, strength, tile, tile, "-", "-", 0.0, skipped=True)
        with self._lock:
            model = self.model_for(entry.path)
            dev = choose_device(device if device is not None else read_option(OPT_DEVICE, DEFAULT_DEVICE))
            wanted = precision if precision is not None else read_option(OPT_PRECISION, DEFAULT_PRECISION)
            keep = bool(read_option(OPT_KEEP_LOADED, DEFAULT_KEEP_LOADED) if keep_loaded is None else keep_loaded)
            use_autocast = use_fp16_autocast(dev, wanted)
            precision_label = "fp16-autocast" if use_autocast else "fp32"
            x, alpha = vd.pil_to_tensor(image)
            height, width = int(x.shape[-2]), int(x.shape[-1])
            residual, retiles = make_residual_fn(model, dev, use_autocast)

            def on_retry(new_tile: int, exc: BaseException) -> None:
                self._log(f"메모리 부족 — 타일을 {new_tile} 으로 줄여 다시 합니다 ({type(exc).__name__})")
                self._empty_cache(self._forge_memory())

            try:
                self._acquire(model, dev, memory_estimate(height, width, tile))
                with torch.inference_mode():
                    delta, used = vd.tiled_residual_with_oom_retry(
                        x, residual, tile=tile, overlap=vd.TILE_OVERLAP, out_device="cpu", is_oom=is_oom,
                        on_retry=on_retry,
                    )
                    check = vd.check_residual(x, delta)
                    if check.looks_like_image:
                        raise NotResidualModelError(not_residual_message(entry.name, check))
                    if check.blew_up:
                        raise ResidualBlowUpError(blow_up_message(entry.name, check))
                    result = vd.finalize(vd.apply_residual(x, delta, mode_key, strength))
            finally:
                if release_after_run(keep, dev):
                    self._release_locked()
            out = vd.tensor_to_pil(result, alpha)
            stats = vd.residual_stats(delta, image, out)
            return DegridOutcome(
                out, entry.name, mode_key, strength, tile, used, str(dev), precision_label,
                time.perf_counter() - start, stats, retiles[0],
            )


_SHARED: DegridRuntime | None = None
_SHARED_LOCK = threading.Lock()


def shared_runtime() -> DegridRuntime:
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = DegridRuntime()
        return _SHARED
