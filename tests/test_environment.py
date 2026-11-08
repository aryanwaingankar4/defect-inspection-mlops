import sys


def test_python_is_3_11() -> None:
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version}"
