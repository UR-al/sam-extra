"""테스트 패키지 표지(감사 L43).

venv 에 깔린 다른 패키지(ultralytics)의 최상위 ``tests`` 가 이 폴더를 가려 ``python -m unittest tests.test_x`` 가
실패하던 것을 막는다(확장 루트에서 실행하면 이 패키지가 먼저 잡힌다). 몇몇 테스트는 서로를
``from test_anima_safe_pag import ...`` 처럼 최상위 이름으로 불러오므로, 패키지로 불러올 때도 이 폴더를
sys.path 에 둔다. ``python -m unittest discover -s tests -p "test_*.py"`` 는 이 파일과 무관하게 전과 같다.
"""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
