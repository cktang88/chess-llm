"""Token + dollar accounting with a hard kill-switch."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Best public guess for gpt-5.4-mini pricing (per 1M tokens, USD).
# Override via Budget(input_price=..., output_price=...) if pricing differs.
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


@dataclass
class Budget:
    cap_usd: float
    input_price: float = DEFAULT_INPUT_PRICE_PER_M
    output_price: float = DEFAULT_OUTPUT_PRICE_PER_M
    log_path: Path | None = None
    spent: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cost_of(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000

    def charge(self, *, tag: str, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = self.cost_of(input_tokens, output_tokens)
        with self._lock:
            self.spent += cost
            rec = CallRecord(time.time(), tag, model, input_tokens, output_tokens, cost)
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
        return {"spent": self.spent, "cap": self.cap_usd, "by_tag": by_tag}
