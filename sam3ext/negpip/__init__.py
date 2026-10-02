"""
Copyright (C) 2025 hako-mikan
Copyright (C) 2026 Haoming02

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# NegPiP — vendored into sam-extra (forge_sam3_extension) as sam3ext/negpip.
# Upstream: https://github.com/Haoming02/sd-forge-negpip @ 0585496 (lib_negpip/__init__.py),
#           archived by its author on 2026-09-30. Licence text: sam3ext/negpip/LICENSE (AGPL-3.0).
# MODIFIED by sam-extra, 2026-09-30 (AGPL-3.0 section 5a): package renamed lib_negpip -> sam3ext.negpip;
# added PATCHED (patch state shared across Forge script reloads, see scripts/negpip.py).

try:
    from backend import shared  # noqa
except ImportError:
    IS_NEO = False
else:
    IS_NEO = True

INCOMPATIBLE_EXTENSIONS: set[str] = {
    "Forge Couple",
}

# sam-extra: 상류는 NegPiP._patched 클래스 속성([SD, Anima])이다. Forge 가 스크립트를 다시 불러오면(Reload UI) 클래스가
# 새로 생겨 패치 상태를 잊는다 — 이 목록은 패키지에 두어 다시 불러와도 같은 객체다. 스크립트는 _patched = PATCHED 로 쓴다.
PATCHED: list[bool] = [False, False]
