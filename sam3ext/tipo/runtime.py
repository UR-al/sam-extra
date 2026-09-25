"""TIPO 런타임 — 모델 파일 확인·받기, 로드, CPU/GPU, 생성.

- 가중치는 사용자가 "모델 받기"를 누를 때만 받는다(고정 revision). 저장소에는 넣지 않는다(Kohaku License 1.0).
- 모델 코드는 번들한 ``kohaku`` 패키지를 쓴다 — trust_remote_code 없음.
- 쉴 때는 CPU RAM 에 둔다 — 불러온 직후와 GPU 로 돌린 뒤에는 fp16(약 2 GB), CPU 로 돌린 뒤에는 fp32(약 4 GB).
  여유 VRAM 이 3 GB 보다 적으면 그 회차는 CPU 로 돈다(이미 GPU 에 남아 있는 TIPO 가중치는 여유로 친다).
- GPU 회차는 두 가지다(``generate(keep_on_gpu=)``, UI 의 "GPU 에 남겨 두기" 토글).
  · 맡기기(켬): 모델을 Forge ``ModelPatcher``(load_device=GPU, offload_device=CPU)로 감싸 ``load_models_gpu`` 로 올리고
    끝나도 그대로 둔다. 이후는 Forge 메모리 관리가 정한다 — 여유가 있으면 GPU 에 남아 다음 클릭은 올림(H2D)조차 없고,
    이미지 생성 등에서 자리가 필요하면 Forge ``free_memory`` 가 다른 모델처럼 퇴출한다(부분 퇴출 없이 통째로). 올릴 때도
    Forge 가 필요한 만큼 다른 모델을 내린다. Forge 밖(``backend`` 를 못 부름)이면 끄기와 같다.
  · 끄기: 누를 때만 fp16 으로 올렸다가 끝나면 내린다(Forge 가 모르는 VRAM 을 남기지 않는다).
- 내릴 때 GPU→CPU 복사는 하지 않는다: 올리기 전의 CPU 파라미터 텐서를 쥐고 있다가 그대로 되돌린다(추론은 가중치를 바꾸지
  않으므로 같은 값). 맡기기로 GPU 에 남아 있는 동안에도 그 RAM 사본(fp16 약 2 GB)을 쥐고 있다 — Forge 가 퇴출할 때도 복사
  없이 되돌리고, 다음 클릭은 H2D 만. CPU 는 fp32 로 돌리고, 같은 장치로 계속 누르는 동안에는 dtype 을 바꾸지 않는다 — 클릭마다
  1B 파라미터를 fp16↔fp32 로 두 번 캐스트하지 않게(장치를 바꿀 때만 한 번). fp16→fp32→fp16 은 정확한 변환이라 결과는 비트
  단위로 같다. 어느 경로든 생성에 쓰는 가중치(GPU fp16 / CPU fp32)는 같아 같은 시드면 같은 텍스트다.
- dtype 은 파라미터만 바꾼다 — RoPE 의 inv_freq 같은 fp32 버퍼는 fp32 로 둔다(``model.to(dtype)`` 는 버퍼까지 바꾼다).
- 장치 이동은 inference_mode 밖에서, 생성만 안에서 한다(inference_mode 안에서 옮긴 파라미터는 나중에 망가진다 — anima38 참고).
  Forge 가 샘플링 문맥 안에서 퇴출해도 ``inference_mode(False)`` 로 감싸 옮긴다.
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
EXPAND_JOB = "sam3_tipo_expand"   # 🪄 의 대기열 job 이름
KEEP_UNAVAILABLE_NOTE = "Forge 메모리 관리를 쓸 수 없어 GPU 에 남기지 않았습니다"


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


def _forge_memory():
    """Forge 메모리 관리 — ``(backend.memory_management, ModelPatcher)``. Forge 밖(테스트 등)이면 None."""
    try:
        from backend import memory_management
        from backend.patcher.base import ModelPatcher
    except Exception:
        return None
    return memory_management, ModelPatcher


_PATCHER_CLASSES: dict = {}


def _patcher_class(base):
    """TIPO 전용 ModelPatcher — 올리고 내리는 일은 런타임에 맡긴다(쥐어 둔 CPU 텐서 되돌리기, 파라미터만 fp16).

    Forge 는 ``LoadedModel.model_load`` 에서 ``partially_load`` 로 올리고, ``free_memory`` 에서 ``partially_unload`` →
    (모자라면) ``detach`` 로 내린다. 모듈이 Forge ops(수동 캐스트)가 아니라 반쯤 올린 모델은 쓸 수 없으므로 부분 적재·부분
    퇴출 없이 통째로만 — ``partially_unload`` 가 0 을 돌려주면 Forge 가 ``detach`` 로 통째로 내린다.
    """
    cls = _PATCHER_CLASSES.get(base)
    if cls is not None:
        return cls

    class TipoPatcher(base):
        def __init__(self, model, load_device, offload_device, size=0, *, current_device=None,
                     weight_inplace_update=False, runtime=None):
            super().__init__(model, load_device, offload_device, size, current_device=current_device,
                             weight_inplace_update=weight_inplace_update)
            self.tipo_runtime = runtime

        def model_size(self) -> int:
            return self.size   # GPU 에 올라가는 fp16 크기(쉬는 dtype 이 fp32 여도)

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            return self.tipo_runtime._forge_load(self, device_to)

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            return 0

        def detach(self, unpatch_all=True):
            # load_models_gpu 는 이미 올라간 자신을 목록에서 뺐다 다시 넣으며 detach(unpatch_all=False) 를 부른다 — 그대로 둔다.
            if unpatch_all and self.tipo_runtime is not None:
                self.tipo_runtime._forge_unload(self)
            return self.model

    _PATCHER_CLASSES[base] = TipoPatcher
    return TipoPatcher


@dataclass
class GenerationResult:
    text: str
    seed: int
    device: str
    seconds: float
    note: str | None
    finished: bool = True   # EOS 로 끝났는지 — False 면 토큰 한도에서 잘렸다
    resident: bool = False  # 끝난 뒤 GPU 에 남겨 Forge 메모리 관리에 맡겼는지


class TipoRuntime:
    def __init__(self, loader=None, directory=None, cuda_info=None, forge_memory=None):
        self._loader = loader or _load_pretrained
        self._directory = Path(directory) if directory is not None else None
        self._cuda_info = cuda_info or _cuda_info
        self._forge_memory = forge_memory or _forge_memory
        self._model = None
        self._tokenizer = None
        self._download_lock = threading.Lock()
        # 생성과 내리기가 겹치지 않게(Forge 는 퇴출을 다른 스레드에서 부를 수 있다).
        self._lock = threading.RLock()
        self._patcher = None               # 맡기기 회차에 처음 만든다
        self._resident_cpu_params = None   # GPU 에 남아 있는 동안만 — 올리기 전 CPU 파라미터 텐서
        self._resident_bytes = 0

    @property
    def directory(self) -> Path:
        return self._directory or model_dir()

    @property
    def resident(self) -> bool:
        """GPU 에 남아 있는지(맡기기로 올린 뒤 Forge 가 아직 퇴출하지 않았는지)."""
        return self._resident_cpu_params is not None

    def release(self) -> bool:
        """GPU 에 남은 모델을 내리고 Forge 목록에서도 뺀다 — 쥐어 둔 CPU 텐서를 되돌릴 뿐 GPU→CPU 복사는 없다. 내렸으면 True.

        돌고 있는 생성이 있으면 끝날 때까지 기다린다.
        """
        with self._lock:
            if self._resident_cpu_params is None:
                return False
            patcher = self._patcher
            memory = self._forge_memory() if patcher is not None else None
            if memory is not None:
                loaded = getattr(memory[0], "current_loaded_models", None)
                for index, entry in enumerate(list(loaded or ())):
                    if getattr(entry, "model", None) is patcher:
                        loaded.pop(index)
                        entry.model_unload()   # → patcher.detach() → _forge_unload
                        break
            self._forge_unload(patcher)
            self._empty_cache()
            return True

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
        """장치를 옮기고 파라미터만 dtype 을 바꾼다(버퍼는 불러온 dtype 그대로). inference_mode 밖에서 부른다.

        파라미터는 하나씩 장치·dtype 을 함께 바꾼다 — fp32 로 쉬던 모델을 통째로 GPU 에 올린 뒤 캐스트하면 VRAM 이 잠깐
        4 GB 넘게 든다. 이미 그 장치·dtype 이면 ``.to`` 가 같은 텐서를 돌려주므로 아무것도 복사하지 않는다.
        """
        for param in model.parameters():
            data = param.data
            moved = data.to(device=device, dtype=dtype)
            if moved is not data:
                param.data = moved
        model.to(device=device)   # 버퍼(파라미터는 이미 옮겼다)

    @staticmethod
    def _cpu_params(model):
        """GPU 로 올리기 전 CPU 파라미터 텐서를 쥔다 — 내릴 때 GPU→CPU 복사 없이 그대로 되돌린다."""
        return [(param, param.data) for param in model.parameters()]

    @staticmethod
    def _restore_cpu(model, cpu_params):
        """``_cpu_params`` 로 쥔 CPU 텐서를 다시 붙인다(GPU 사본은 버려진다). 일부만 옮겨졌다 실패해도 전부 CPU 로."""
        for param, data in cpu_params:
            param.data = data
        model.to(device="cpu")   # 버퍼만 — 파라미터는 이미 CPU

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

    @staticmethod
    def _nbytes(cpu_params):
        """쥐어 둔 CPU 텐서의 바이트 수 — GPU 사본도 같은 fp16 이라 GPU 에 올라간 가중치 크기와 같다."""
        return sum(int(data.numel()) * int(data.element_size()) for _param, data in cpu_params)

    @staticmethod
    def _gpu_bytes(model) -> int:
        """GPU 에 올렸을 때의 크기 — 파라미터 fp16 + 버퍼(그대로). Forge 가 적재·퇴출을 판단하는 model_size."""
        total = sum(int(param.data.numel()) * 2 for param in model.parameters())
        buffers = getattr(model, "buffers", None)
        if callable(buffers):
            total += sum(int(buf.numel()) * int(buf.element_size()) for buf in buffers())
        return total

    # ── 맡기기(Forge 메모리 관리) ──
    def _managed_patcher(self, memory, model):
        memory_management, model_patcher = memory
        patcher_cls = _patcher_class(model_patcher)
        if not isinstance(self._patcher, patcher_cls) or self._patcher.model is not model:
            import torch

            self._patcher = patcher_cls(model, memory_management.get_torch_device(), torch.device("cpu"),
                                        self._gpu_bytes(model), runtime=self)
        return self._patcher

    def _forge_load(self, patcher, device_to) -> int:
        """Forge 가 올릴 때(``partially_load``) — RAM 에서 fp16 으로 맞춘 뒤 그 CPU 텐서를 쥐고 통째로 올린다. 새로 올린 바이트."""
        import torch

        with self._lock:
            gained = 0
            if self._resident_cpu_params is None:
                model = patcher.model
                with torch.inference_mode(False):
                    self._place(model, "cpu", torch.float16)
                    cpu_params = self._cpu_params(model)
                    try:
                        self._place(model, device_to, torch.float16)
                    except BaseException:
                        self._restore_cpu(model, cpu_params)   # 일부만 올라간 채 실패해도 전부 CPU 로
                        raise
                self._resident_cpu_params = cpu_params
                self._resident_bytes = self._nbytes(cpu_params)
                gained = patcher.model_size()
            patcher.model.model_loaded_weight_memory = patcher.model_size()
            patcher.model.model_lowvram = False
            patcher.current_device = device_to
            return gained

    def _forge_unload(self, patcher) -> bool:
        """Forge 가 퇴출할 때(``detach``)와 ``release`` — 쥐어 둔 CPU 텐서를 되돌린다(복사 없음). 내렸으면 True."""
        import torch

        with self._lock:
            cpu_params = self._resident_cpu_params
            if cpu_params is None:
                return False
            with torch.inference_mode(False):
                self._restore_cpu(self._model if patcher is None else patcher.model, cpu_params)
            self._resident_cpu_params = None
            self._resident_bytes = 0
            if patcher is not None:
                patcher.model.model_loaded_weight_memory = 0
                patcher.current_device = patcher.offload_device
            return True

    def generate(self, prompt_text: str, *, requested_device: str, max_new_tokens: int, seed: int,
                 keep_on_gpu: bool = False) -> GenerationResult:
        """``keep_on_gpu``: GPU 로 돌 때 Forge 메모리 관리에 맡겨 끝나도 남긴다(자리가 필요하면 Forge 가 내린다).
        끄면(기본) 누를 때만 올렸다가 끝나면 내린다. 어느 쪽이든 같은 시드면 같은 텍스트다."""
        with self._lock:
            return self._generate(prompt_text, requested_device, max_new_tokens, seed, bool(keep_on_gpu))

    def _generate(self, prompt_text, requested_device, max_new_tokens, seed, keep_on_gpu) -> GenerationResult:
        import torch

        model, tokenizer = self._ensure_loaded()
        cuda_available, free_bytes = self._cuda_info()
        if self.resident:
            free_bytes += self._resident_bytes   # 이미 올라간 가중치는 여유로 친다(다시 올리지 않으므로)
        device, note = choose_device(requested_device, cuda_available, free_bytes)
        memory = None
        if keep_on_gpu and device == "cuda":
            memory = self._forge_memory()
            if memory is None:
                note = KEEP_UNAVAILABLE_NOTE
        if memory is None:
            self.release()   # 맡기지 않는 회차(CPU·끄기·Forge 밖)는 남은 GPU 사본을 먼저 내린다(복사 없음)
        seed = int(seed)
        if seed < 0:
            seed = random.randrange(2 ** 31)
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else EOS_ID
        start = time.perf_counter()
        if memory is not None:
            patcher = self._managed_patcher(memory, model)
            try:
                # 이미 남아 있으면 Forge 는 목록 순서만 바꾸고 아무것도 옮기지 않는다. 아니면 필요한 만큼 다른 모델을
                # 내린 뒤 partially_load → _forge_load 로 통째로 올린다.
                memory[0].load_models_gpu([patcher], force_full_load=True)
                if not self.resident:
                    raise RuntimeError("TIPO 를 GPU 에 올리지 못했습니다")
                text, finished = self._sample(model, tokenizer, prompt_text, patcher.load_device, "cuda",
                                              max_new_tokens, seed, eos_id)
            except BaseException:
                if not self.release():
                    self._empty_cache()
                raise
            return GenerationResult(text, seed, device, time.perf_counter() - start, note, finished, resident=True)
        # 쉬는 dtype 은 마지막으로 쓴 장치를 따른다 — 장치를 바꿀 때만 한 번 캐스트한다.
        # GPU 회차: RAM 에도 fp16 으로 두고(이미 fp16 이면 아무것도 안 함) 그 CPU 텐서를 쥐어 두었다가 끝나면 그대로 되돌린다.
        # CPU 회차: fp32 로 한 번 바꾸면 그대로 둔다.
        cpu_params = None
        if device == "cuda":
            self._place(model, "cpu", torch.float16)
            cpu_params = self._cpu_params(model)
        try:
            # 장치 이동은 inference_mode 밖에서 — 생성만 안에서 한다. 옮기다 실패해도(OOM) finally 가 CPU 로 되돌린다.
            self._place(model, device, torch.float16 if device == "cuda" else torch.float32)
            text, finished = self._sample(model, tokenizer, prompt_text, device, device, max_new_tokens, seed, eos_id)
        finally:
            if cpu_params is not None:
                self._restore_cpu(model, cpu_params)
                del cpu_params
                self._empty_cache()
        return GenerationResult(text, seed, device, time.perf_counter() - start, note, finished)

    def _sample(self, model, tokenizer, prompt_text, input_device, device, max_new_tokens, seed, eos_id):
        import torch

        ids = self._move_inputs(self._encode(tokenizer, prompt_text), input_device)
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
        return tokenizer.decode(new_ids, skip_special_tokens=True), finished


_SHARED: TipoRuntime | None = None


def shared_runtime() -> TipoRuntime:
    global _SHARED
    if _SHARED is None:
        _SHARED = TipoRuntime()
    return _SHARED
