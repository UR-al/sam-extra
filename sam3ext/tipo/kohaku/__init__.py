"""TIPO-v2.1 의 KohakUwU MoE 모델 코드.

configuration_kohaku.py·modeling_kohaku.py 는 KBlueLeaf/TIPO-v2.1-1B-A200M(revision
f5a318524a4ab30cdbbf51816cf406170f454e65)의 원문 그대로다. KohakUwULLM(https://github.com/KohakuBlueleaf/KohakUwULLM,
Apache-2.0)의 src/kohakuwullm/export/hf/ 파일과 같다. 고치지 말 것 — tests/test_tipo_model.py 가 SHA256 으로 고정한다.
"""
from .configuration_kohaku import KohakuConfig
from .modeling_kohaku import KohakuForCausalLM

__all__ = ["KohakuConfig", "KohakuForCausalLM"]
