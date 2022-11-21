from .utils import get_true_path
import pytest

@pytest.mark.parametrize("path, expected",
[
    ("file:///lrde/home2/rdi-2022/rhermary/update_morphonet/src/mlruns/0/9dd8f931b493403ab3a88f30301b9c6b/artifacts/outputs",
    "/lrde/home2/rdi-2022/rhermary/update_morphonet/src/mlruns/0/9dd8f931b493403ab3a88f30301b9c6b/artifacts/outputs"),
    ("/lrde/home2/rdi-2022/rhermary/update_morphonet/src/mlruns/0/9dd8f931b493403ab3a88f30301b9c6b/artifacts/outputs",
    "/lrde/home2/rdi-2022/rhermary/update_morphonet/src/mlruns/0/9dd8f931b493403ab3a88f30301b9c6b/artifacts/outputs"),

])
def test_get_true_path(path, expected):
    assert get_true_path(path) == expected