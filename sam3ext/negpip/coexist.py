# sam-extra 자체 코드(GPL-3.0-only) — 상류 NegPiP 코드가 아니다. 이 폴더의 다른 파일(AGPL-3.0)과 달리 새로 쓴 파일이다.
"""내장 NegPiP 와 따로 설치된 sd-forge-negpip 가 함께 로드됐는지 본다 (Forge 를 import 하지 않는다).

둘 다 돌면 같은 훅을 두 번 건다. Anima 는 -1 마스크가 두 번 곱해져 NegPiP 가 상쇄되고, 두 번째 패치가
``model.orig_forward`` 를 덮어 첫 래퍼가 자기 자신을 부르게 된다. 그래서 따로 설치된 쪽이 보이면 내장이 쉰다.

판별 신호 두 가지 (어느 하나면 쉰다):

1. 스크립트 객체 — ``p.scripts`` 의 always-on 스크립트 중 제목이 ``NegPiP`` 인데 내장 표시(``BUILTIN_ATTR``)가 없는 것.
   Forge 는 켜진 확장의 스크립트만 러너에 넣으므로, 확장을 끄거나 지우면 사라진다. 우리 스크립트보다 뒤에 돌아
   아직 패치하지 않았어도 여기서 잡힌다(확장 폴더 이름순으로 sd-forge-negpip 가 뒤).
2. 패치 흔적 — 내장이 패치하지 않은 상태(``PATCHED`` 가 모두 False)인데 모델·DiT·cross-attn 모듈에 NegPiP 가 거는 속성
   (``orig_forward``, 클래스의 ``negpip_orig_forward``)이 남아 있다. 다른 NegPiP 가 이미 패치했다는 뜻이다(순서를 바꾼
   설치, 실패한 생성이 남긴 패치). 이 위에 패치하면 위의 자기 호출이 생긴다.

모듈 경로(``sys.modules['lib_negpip']``)는 경고 문구에만 쓴다 — 확장을 꺼도 모듈은 재시작 전까지 남으므로 판별에 쓰면
내장까지 영영 꺼진다.
"""
from __future__ import annotations

import sys

TITLE = "NegPiP"
BUILTIN_ATTR = "_sam_extra_builtin_negpip"
STANDALONE_PACKAGE = "lib_negpip"
# 상류 NegPiP 가 거는 속성 — 인스턴스(모델·DiT·SD attn2) 와 클래스(Anima SelfCrossAttention)
PATCH_ATTR = "orig_forward"
CLASS_PATCH_ATTR = "negpip_orig_forward"

_warned = False


def foreign_scripts(runner) -> list:
    """runner(p.scripts) 에서 내장이 아닌 NegPiP 스크립트 객체들."""
    scripts = getattr(runner, "alwayson_scripts", None)
    if scripts is None:
        scripts = getattr(runner, "scripts", None) or ()
    found = []
    for script in scripts:
        if getattr(script, BUILTIN_ATTR, False):
            continue
        try:
            title = script.title()
        except Exception:
            continue
        if title == TITLE:
            found.append(script)
    return found


def _diffusion_model(sd_model):
    unet = getattr(getattr(sd_model, "forge_objects", None), "unet", None)
    return getattr(getattr(unet, "model", None), "diffusion_model", None)


def foreign_patch_live(sd_model) -> bool:
    """모델에 NegPiP 모양의 패치가 살아 있는가. 내장이 패치하지 않았을 때만 부른다(그땐 남의 것이다)."""
    if hasattr(sd_model, PATCH_ATTR):   # Anima get_learned_conditioning 훅
        return True
    dit = _diffusion_model(sd_model)
    if dit is None:
        return False
    if hasattr(dit, PATCH_ATTR):        # Anima DiT forward 훅
        return True
    named_modules = getattr(dit, "named_modules", None)
    if named_modules is None:
        return False
    for _, module in named_modules():
        # SD attn2 CrossAttention 인스턴스 훅, Anima SelfCrossAttention 클래스 훅
        if hasattr(module, PATCH_ATTR) or hasattr(type(module), CLASS_PATCH_ATTR):
            return True
    return False


def _where(script) -> str:
    where = getattr(script, "filename", None)
    if where:
        return str(where)
    module = sys.modules.get(STANDALONE_PACKAGE)
    return str(getattr(module, "__file__", None) or type(script).__module__)


def standalone_reason(p, ours_patched) -> str | None:
    """내장 NegPiP 가 이번 생성에서 쉬어야 하는 이유 — 없으면 None."""
    found = foreign_scripts(getattr(p, "scripts", None))
    if found:
        return f"another NegPiP script is loaded ({_where(found[0])})"
    if not any(ours_patched) and foreign_patch_live(getattr(p, "sd_model", None)):
        return "another NegPiP patch is active on the model"
    return None


def warn_once(reason: str) -> None:
    """프로세스당 한 번만 — 매 생성마다 콘솔을 채우지 않는다."""
    global _warned
    if _warned:
        return
    _warned = True
    text = (
        f"[sam-extra] WARNING: built-in NegPiP is standing down: {reason}. sd-forge-negpip is now part of "
        "sam-extra - remove extensions/sd-forge-negpip (or disable it) and restart Forge so NegPiP is not applied twice."
    )
    try:   # 레거시 콘솔 코드페이지(cp949 등)에서 경로의 글자로 생성이 죽지 않게
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))
