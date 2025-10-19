from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import time
from .model_loader import load_causal_lm, generate_json_only, build_inputs, _render_chat
from .utils import parse_envelope
from .json_enforcer import validate_envelope, coerce_minimal_defaults
from .strategies import REGISTRY as STRATS
from .pseudocode import augment_system_prompt

@dataclass
class Agent:
    name: str
    role: str
    domain: str
    model_id: str
    system_prompt: str
    seed: int = 7
    max_new_tokens: int = 768
    strategy_id: str = "strategy-01"

    def __post_init__(self):
        self.tok, self.model = load_causal_lm(self.model_id, seed=self.seed)
        self.cfg = STRATS.get(self.strategy_id, STRATS["strategy-01"])
        self.system_prompt = augment_system_prompt(self.system_prompt)

    def _build_user_prompt(self, task: str, transcript: List[Dict[str, Any]]) -> str:
        # Include last peer [CONTACT] or request.to_peer and any arbiter hint
        peer_msgs = []
        for e in reversed(transcript[-6:]):  # look back a few turns
            if isinstance(e, dict):
                r = e.get("role","")
                if r and r != self.role:
                    tags = set(e.get("tags", []))
                    if "[CONTACT]" in tags or e.get("status") == "NEED_PEER":
                        req = (e.get("request") or {}).get("to_peer", "")
                        if req:
                            peer_msgs.append(f"Peer says: {req}")
                            break
                if e.get("role") == "arbiter":
                    peer_msgs.append(f"Arbiter: {e.get('public_message','')}")
                    break
        if peer_msgs:
            return task.strip() + "\n\n" + "\n".join(peer_msgs)
        return task

    def _gen_once(self, user_prompt: str, max_new_tokens: int) -> Tuple[Dict[str,Any] | None, str]:
        raw = generate_json_only(self.tok, self.model, self.system_prompt, user_prompt,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=not self.cfg.greedy,
                                 temperature=0.7 if not self.cfg.greedy else 0.0,
                                 top_p=0.95 if not self.cfg.greedy else 1.0)
        obj, err = parse_envelope(raw)
        return obj, raw

    def step(self, task: str, transcript: List[Dict[str, Any]]) -> Tuple[Dict[str,Any], str]:
        user_prompt = self._build_user_prompt(task, transcript)
        # Self-consistency k samples if configured
        candidates: List[Tuple[Dict[str,Any], str]] = []
        k = max(1, int(self.cfg.k_samples))
        for _ in range(k):
            obj, raw = self._gen_once(user_prompt, max_new_tokens=min(self.max_new_tokens, self.cfg.max_new_tokens))
            if obj:
                candidates.append((obj, raw))
                if k == 1:
                    break
        if not candidates:
            # one retry with an explicit JSON reminder
            reminder = user_prompt + "\n\nIMPORTANT: Emit a single valid JSON object matching the agreed envelope. No prose."
            obj, raw = self._gen_once(reminder, max_new_tokens=min(self.max_new_tokens, self.cfg.max_new_tokens))
            if not obj:
                fb = {
                    "role": self.role,
                    "domain": self.domain,
                    "task_understanding": "Parse error; requesting clarification.",
                    "public_message": "[CONTACT] JSON parse error; please restate or simplify.",
                    "artifact": {"type": "results", "content": {}},
                    "needs_from_peer": ["Re-send last artifact in simpler structure"],
                    "handoff_to": "peer",
                    "status": "NEED_PEER",
                    "tags": ["[CONTACT]"],
                    "request": {"to_peer": "Please restate your last message as valid JSON matching the agreed envelope."},
                    "meta": {"strategy_id": self.strategy_id},
                    "final_solution": {"canonical_text": "", "sha256": ""}
                }
                return fb, raw or ""
            candidates.append((obj, raw))
        # If multiple, pick the first SOLVED else first
        best_obj, best_raw = None, ""
        for obj, raw in candidates:
            if (obj.get("status") == "SOLVED"):
                best_obj, best_raw = obj, raw; break
        if best_obj is None:
            best_obj, best_raw = candidates[0]
        # Ensure meta/strategy_id present
        best_obj.setdefault("meta", {}).setdefault("strategy_id", self.strategy_id)
        # Estimate tokens (prompt/gen) for logging
        try:
            prompt_text = _render_chat(self.tok, self.system_prompt, user_prompt)
            prompt_tokens = len(self.tok(prompt_text).input_ids)
            gen_tokens = max(0, len(self.tok(best_raw).input_ids) - prompt_tokens)
            best_obj.setdefault("meta", {}).setdefault("token_estimate", {"prompt": prompt_tokens, "gen": gen_tokens})
        except Exception:
            pass
        # role/domain defaults
        best_obj.setdefault("role", self.role)
        best_obj.setdefault("domain", self.domain)
        # protocol keys
        best_obj.setdefault("tags", [])
        best_obj.setdefault("request", {"to_peer": None})
        fs = best_obj.get("final_solution", {}) or {}
        fs.setdefault("sha256","")
        best_obj["final_solution"] = fs
        # JSON Schema validation (soft enforcement)
        try:
            import os
            schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schemas", "envelope.schema.json")
            ok, errs = validate_envelope(best_obj, schema_path)
            if not ok:
                # try minimal coercion, then revalidate once
                best_obj = coerce_minimal_defaults(best_obj)
                ok2, errs2 = validate_envelope(best_obj, schema_path)
                if not ok2:
                    # attach validation errors to meta for observability
                    best_obj.setdefault("meta", {}).setdefault("validation_errors", errs2 or errs)
        except Exception:
            pass
        return best_obj, best_raw
