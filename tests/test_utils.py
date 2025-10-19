
from src.utils import normalize_text, sha256_hex
def test_normalize_nfkc():
    assert normalize_text("Ａ B\u200B C") == "A B C"
def test_sha256():
    assert len(sha256_hex("abc")) == 64
