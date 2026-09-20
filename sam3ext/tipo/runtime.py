"""TIPO 런타임 — 모델 파일 확인·받기, 로드, CPU/GPU, 생성.

- 가중치는 사용자가 "모델 받기"를 누를 때만 받는다(고정 revision). 저장소에는 넣지 않는다(Kohaku License 1.0).
- 모델 코드는 번들한 ``kohaku`` 패키지를 쓴다 — trust_remote_code 없음.
- 쉴 때는 CPU RAM 에 fp16(약 2 GB)으로 둔다. GPU 는 누를 때만 fp16 으로 올렸다가 끝나면 CPU 로 내린다. 이 모델은 Forge
  메모리 관리 밖이라 VRAM 에 상주하면 Forge 가 Anima 모델을 부분 로드로 밀어낼 수 있다. 여유 VRAM 이 3 GB 보다 적으면 그
  회차는 CPU 로 돈다. CPU 는 fp32 로 돌리고 끝나면 fp16 으로 되돌린다.
- dtype 은 파라미터만 바꾼다 — RoPE 의 inv_freq 같은 fp32 버퍼는 fp32 로 둔다(``model.to(dtype)`` 는 버퍼까지 바꾼다).
- 장치 이동은 inference_mode 밖에서, 생성만 안에서 한다(inference_mode 안에서 옮긴 파라미터는 나중에 망가진다 — anima38 참고).
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "KBlueLeaf/TIPO-v2.1-1B-A200M"
REVISION = "f5a318524a4ab30cdbbf51816cf406170f454e65"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
MODEL_BYTES_LABEL = "1.98 GB"
GPU_MIN_FREE_BYTES = 3 * 1024 ** 3
BOS_ID, EOS_ID, PAD_ID = 64000, 64001, 64002


def forge_models_dir() -> Path:
    try:
        from modules import paths

        return Path(paths.models_path)
    except Exception:
        # <forge>/extensions/<ext>/sam3ext/tipo/runtime.py
        return Path(__file__).resolve().parents[4] / "models"


def model_dir(models_dir: Path | None = None) -> Path:
    return Path(models_dir or forge_models_dir()) / "TIPO" / "TIPO-v2.1-1B-A200M"


def missing_files(directory) -> list[str]:
    directory = Path(directory)
    return [name for name in FILES if not (directory / name).is_file()]


def download(directory, downloader=None) -> None:
    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in missing_files(directory):
        downloader(repo_id=REPO_ID, filename=name, revision=REVISION, local_dir=str(directory))


def choose_device(requested: str, cuda_available: bool, free_bytes: int) -> tuple[str, str | None]:
    if requested != "GPU":
        return "cpu", None
    if not cuda_available:
        return "cpu", "CUDA 를 쓸 수 없어 CPU 로 돌렸습니다"
    if free_bytes < GPU_MIN_FREE_BYTES:
        return "cpu", f"GPU 여유가 {free_bytes / 1024 ** 3:.1f} GB 라 CPU 로 돌렸습니다"
    return "cuda", None


def _cuda_info() -> tuple[bool, int]:
    import torch

    if not torch.cuda.is_available():
        return False, 0
    free, _total = torch.cuda.mem_get_info()
    return True, int(free)


def _load_pretrained(directory):
    import torch
    from transformers import PreTrainedTokenizerFast

    from .kohaku import KohakuConfig, KohakuForCausalLM

    config = KohakuConfig.from_pretrained(directory)
    model = KohakuForCausalLM.from_pretrained(directory, config=config, dtype=torch.float16)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
    return model.eval().requires_grad_(False), tokenizer


@dataclass
class GenerationResult:
    text: str
    seed: int
    device: str
    seconds: float
    note: str | None
    finished: bool = True   # EOS 로 끝났는지 — False 면 토큰 한도에서 잘렸다


class TipoRuntime:
    def __init__(self, loader=None, directory=None, cuda_info=None):
        self._loader = loader or _load_pretrained
        self._directory = Path(directory) if directory is not None else None
        self._cuda_info = cuda_info or _cuda_info
        self._model = None
        self._tokenizer = None
        self._download_lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._directory or model_dir()

    def missing_files(self) -> list[str]:
        return missing_files(self.directory)

    def download(self, downloader=None) -> bool:
        """받았거나 이미 다 있으면 True. 다른 곳(다른 탭·새로 고친 창)에서 받는 중이면 기다리지 않고 False.

        없는 파일은 잠금 안에서 다시 센다 — 두 번째 호출이 막 받은 파일을 지우고 다시 받지 않게.
        """
        if not self._download_lock.acquire(blocking=False):
            return False
        try:
            download(self.directory, downloader=downloader)
        finally:
            self._download_lock.release()
        return True

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer = self._loader(self.directory)
        return self._model, self._tokenizer

    # ── CUDA 를 실제로 건드리는 곳(테스트에서 막는다) ──
    @staticmethod
    def _place(model, device, dtype):
        """장치를 옮기고 파라미터만 dtype 을 바꾼다(버퍼는 불러온 dtype 그대로). inference_mode 밖에서 부른다."""
        model.to(device=device)
        for param in model.parameters():
            if param.dtype != dtype:
                param.data = param.data.to(dtype)

    def _move_inputs(self, ids, device):
        return ids.to(device)

    def _rng_devices(self, device):
        import torch

        return [torch.cuda.current_device()] if device == "cuda" else []

    def _empty_cache(self):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode(self, tokenizer, prompt_text):
        import torch

        ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else BOS_ID
        if ids.shape[1] == 0 or int(ids[0, 0]) != bos:
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        return ids

    def generate(self, prompt_text: str, *, requested_device: str, max_new_tokens: int, seed: int) -> GenerationResult:
        import torch

        model, tokenizer = self._ensure_loaded()
        cuda_available, free_bytes = self._cuda_info()
        device, note = choose_device(requested_device, cuda_available, free_bytes)
        seed = int(seed)
        if seed < 0:
            seed = random.randrange(2 ** 31)
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else EOS_ID
        start = time.perf_counter()
        try:
            # 장치 이동은 inference_mode 밖에서 — 생성만 안에서 한다. 옮기다 실패해도(OOM) finally 가 CPU 로 되돌린다.
            self._place(model, device, torch.float16 if device == "cuda" else torch.float32)
            ids = self._move_inputs(self._encode(tokenizer, prompt_text), device)
            with torch.random.fork_rng(devices=self._rng_devices(device)):   # Forge 의 전역 난수 상태는 그대로
                torch.manual_seed(seed)
                with torch.inference_mode():
                    output = model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=int(max_new_tokens),
                        do_sample=True,
                        temperature=1.0,
                        min_p=0.1,
                        top_k=0,
                        top_p=1.0,
                        eos_token_id=eos_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else PAD_ID,
                    )
            new_ids = output[0, ids.shape[1]:].cpu()
            finished = bool((new_ids == eos_id).any())
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
        finally:
            self._place(model, "cpu", torch.float16)
            if device == "cuda":
                self._empty_cache()
        return GenerationResult(text, seed, device, time.perf_counter() - start, note, finished)


_SHARED: TipoRuntime | None = None


def shared_runtime() -> TipoRuntime:
    global _SHARED
    if _SHARED is None:
        _SHARED = TipoRuntime()
    return _SHARED
