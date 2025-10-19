from __future__ import annotations
from typing import Dict, Any
import sqlite3, json

def judge_sql(task_prompt: str, env: Dict[str, Any]) -> Dict[str, Any]:
    # Expect inline SQLite spec in prompt:
    # {"sqlite_rows":[{"table":"t","rows":[{"col":"val"}, ...]}], "expected_checksum": <int>}
    ct = (env.get("final_solution") or {}).get("canonical_text","")
    try:
        spec = json.loads(task_prompt)
    except Exception:
        return {"passes_judge": False, "kind":"sql", "reason":"prompt lacks sqlite spec"}

    rows_spec = spec.get("sqlite_rows",[])
    if not rows_spec:
        return {"passes_judge": False, "kind":"sql", "reason":"no sqlite_rows in prompt"}

    con = sqlite3.connect(":memory:")
    try:
        # Create and fill tables
        for tbl in rows_spec:
            name = tbl["table"]
            rows = tbl.get("rows", [])
            if not rows:
                continue
            cols = list(rows[0].keys())
            con.execute(f"CREATE TABLE {name} ({', '.join([c+' TEXT' for c in cols])})")
            for r in rows:
                placeholders = ", ".join(["?"] * len(cols))
                col_list = ", ".join(cols)
                con.execute(f"INSERT INTO {name} ({col_list}) VALUES ({placeholders})", [str(r[c]) for c in cols])

        # Run the candidate SQL
        cur = con.execute(ct)
        out = cur.fetchall()
        checksum = hash(tuple(out))
        exp = spec.get("expected_checksum", None)
        if exp is None:
            return {"passes_judge": True, "kind":"sql", "rows": len(out)}
        return {"passes_judge": checksum == exp, "kind":"sql", "rows": len(out), "checksum": checksum}
    except Exception as e:
        return {"passes_judge": False, "kind":"sql", "error": str(e)}
    finally:
        con.close()
