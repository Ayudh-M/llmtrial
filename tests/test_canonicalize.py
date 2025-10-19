
from src.canonicalize import canonicalize_for_hash

def test_json_minify_and_sort():
    s = '{\n  "b": 2,  "a": [1, 2]\n}'
    assert canonicalize_for_hash(s) == '{"a":[1,2],"b":2}'

def test_sql_normalize():
    s = "SELECT  a  /* c */  ,  b  FROM\n  t  -- comment\n WHERE a = ' x  y '  "
    out = canonicalize_for_hash(s)
    assert "/*" not in out and "--" not in out
    assert "  " not in out  # collapsed

def test_number_norm_collapses_ws():
    s = "   3.1415    "
    out = canonicalize_for_hash(s)
    assert out == "3.1415"
