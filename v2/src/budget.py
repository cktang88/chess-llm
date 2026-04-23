"""Token + dollar accounting with a hard kill-switch.

Prefers OpenRouter's reported per-call cost when available; falls back to a
price-per-token estimate otherwise.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Fallback prices (USD per 1M tokens). Only used if upstream doesn't report cost.
DEFAULT_INPUT_PRICE_PER_M = 0.25
DEFAULT_OUTPUT_PRICE_PER_M = 2.00


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallRecord:
    ts: float
    tag: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    source: str  # "reported" or "estimated"


@dataclass
class Budget:
    cap_usd: float
    input_price: float = DEFAULT_INPUT_PRICE_PER_M
    output_price: float = DEFAULT_OUTPUT_PRICE_PER_M
    log_path: Path | None = None
    spent: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000

    def charge(self, *, tag: str, model: str, input_tokens: int, output_tokens: int,
               reported_cost: float | None = None) -> float:
        if reported_cost is not None and reported_cost >= 0:
            cost = float(reported_cost)
            source = "reported"
        else:
            cost = self._estimate(input_tokens, output_tokens)
            source = "estimated"
        with self._lock:
            self.spent += cost
            rec = CallRecord(time.time(), tag, model, input_tokens, output_tokens, cost, source)
            self.calls.append(rec)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")
            if self.spent > self.cap_usd:
                raise BudgetExceeded(f"spent ${self.spent:.4f} > cap ${self.cap_usd:.4f}")
        return cost

    def remaining(self) -> float:
        return self.cap_usd - self.spent

    def summary(self) -> dict:
        by_tag: dict[str, dict] = {}
        for c in self.calls:
            d = by_tag.setdefault(c.tag, {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
            d["calls"] += 1
            d["input"] += c.input_tokens
            d["output"] += c.output_tokens
            d["cost"] += c.cost
        return {"spent": self.spent, "cap": self.cap_usd, "n_calls": len(self.calls),
                "by_tag": by_tag}
