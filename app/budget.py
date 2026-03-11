"""
RouteIQ — Budget manager with real EWMA burn-rate tracking.

Features:
- Total balance tracking with configurable starting amount
- EWMA-based burn rate (not simple window average)
- Daily and monthly spend limits
- Runway projection
- Threshold alerts with debouncing
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import yaml

logger = logging.getLogger(__name__)

EPSILON = 1e-9
MAX_LOG_ENTRIES = 2000


class BudgetManager:
    """Tracks LLM spend, burn rate, runway, and fires threshold alerts."""

    def __init__(
        self,
        config_path: str = "conf/router.yaml",
        state_path: str = "state/state.json",
    ) -> None:
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        budget_cfg = cfg.get("budget", {})
        self._starting = float(budget_cfg.get("starting_balance_usd", 100.0))
        self._warn_pct = float(budget_cfg.get("warn_at_pct", 50))
        self._critical_pct = float(budget_cfg.get("critical_at_pct", 80))
        self._stop_pct = float(budget_cfg.get("stop_at_pct", 98))
        self._daily_limit = float(budget_cfg.get("daily_limit_usd", 0))
        self._monthly_limit = float(budget_cfg.get("monthly_limit_usd", 0))

        self._state = self._load_state()
        # EWMA state (not persisted — recalculated from log on load)
        self._ewma_rate = 0.0
        self._last_ewma_ts = 0.0

    # ── State persistence ──

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not load state, starting fresh: %s", e)
        return {
            "balance_usd": self._starting,
            "spent_usd": 0.0,
            "spend_log": [],
            "last_alert": {"50": None, "80": None, "98": None},
        }

    def save(self) -> None:
        tmp = str(self._state_path) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.error("Failed to save budget state: %s", e)

    # ── Tracking ──

    def track(self, model: str, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        now = time.time()
        entry = {"ts": now, "model": model, "cost_usd": cost_usd,
                 "tokens_in": tokens_in, "tokens_out": tokens_out}

        log: list = self._state["spend_log"]
        log.append(entry)
        if len(log) > MAX_LOG_ENTRIES:
            self._state["spend_log"] = log[-MAX_LOG_ENTRIES:]

        self._state["spent_usd"] = self._state.get("spent_usd", 0.0) + cost_usd
        self._state["balance_usd"] = self._starting - self._state["spent_usd"]

        self._update_ewma(cost_usd, now)
        self.save()

    def _update_ewma(self, cost_usd: float, now: float) -> None:
        """Real exponential weighted moving average of burn rate."""
        if self._last_ewma_ts <= 0:
            self._last_ewma_ts = now
            self._ewma_rate = 0.0
            return

        dt = now - self._last_ewma_ts
        if dt < EPSILON:
            return

        # Instantaneous rate: cost / time_delta (USD per second)
        instant_rate = cost_usd / dt

        # EWMA with alpha based on half-life of ~60 seconds
        half_life = 60.0
        alpha = 1.0 - (0.5 ** (dt / half_life))
        self._ewma_rate = alpha * instant_rate + (1.0 - alpha) * self._ewma_rate
        self._last_ewma_ts = now

    # ── Metrics ──

    def burn_rate_per_min(self) -> float:
        """EWMA burn rate in USD/minute."""
        return self._ewma_rate * 60.0

    def runway_minutes(self) -> float:
        rate = self.burn_rate_per_min()
        if rate < EPSILON:
            return float("inf")
        return self.balance_usd() / rate

    def balance_usd(self) -> float:
        return max(0.0, self._state.get("balance_usd", self._starting))

    def spent_usd(self) -> float:
        return self._state.get("spent_usd", 0.0)

    def spent_pct(self) -> float:
        if self._starting <= 0:
            return 0.0
        return (self.spent_usd() / self._starting) * 100

    def daily_spend(self) -> float:
        """Total spend in the current UTC day."""
        import datetime
        today_start = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        return sum(
            e["cost_usd"] for e in self._state.get("spend_log", [])
            if e["ts"] >= today_start
        )

    def monthly_spend(self) -> float:
        """Total spend in the current UTC month."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        return sum(
            e["cost_usd"] for e in self._state.get("spend_log", [])
            if e["ts"] >= month_start
        )

    def budget_mode(self) -> str:
        """Returns routing mode: normal → economy → critical → stopped."""
        # Check daily/monthly limits first
        if self._daily_limit > 0 and self.daily_spend() >= self._daily_limit:
            return "stopped"
        if self._monthly_limit > 0 and self.monthly_spend() >= self._monthly_limit:
            return "stopped"

        pct = self.spent_pct()
        if pct >= self._stop_pct:
            return "stopped"
        if pct >= self._critical_pct:
            return "critical"
        if pct >= self._warn_pct:
            return "economy"
        return "normal"

    def get_status(self) -> dict:
        runway = self.runway_minutes()
        burn = self.burn_rate_per_min()
        return {
            "balance_usd": round(self.balance_usd(), 4),
            "spent_usd": round(self.spent_usd(), 4),
            "spent_pct": round(self.spent_pct(), 1),
            "mode": self.budget_mode(),
            "burn_rate_usd_per_min": round(burn, 6),
            "burn_rate_usd_per_hour": round(burn * 60, 4),
            "runway_minutes": round(runway, 1) if runway != float("inf") else None,
            "runway_hours": round(runway / 60, 1) if runway != float("inf") else None,
            "runway_days": round(runway / 1440, 1) if runway != float("inf") else None,
            "daily_spend_usd": round(self.daily_spend(), 4),
            "monthly_spend_usd": round(self.monthly_spend(), 4),
            "daily_limit_usd": self._daily_limit if self._daily_limit > 0 else None,
            "monthly_limit_usd": self._monthly_limit if self._monthly_limit > 0 else None,
        }

    # ── Alerts ──

    def check_and_alert(self, alert_fn: Callable[[str], None]) -> None:
        pct = self.spent_pct()
        now = time.time()
        last = self._state.get("last_alert", {})

        levels = [
            (self._stop_pct, "98", "🚨 RouteIQ: budget exhausted! Requests stopped."),
            (self._critical_pct, "80", "🔴 RouteIQ: <20% budget left. Switching to cheap models."),
            (self._warn_pct, "50", "🟡 RouteIQ: 50% budget spent. Economy mode."),
        ]

        for threshold, key, msg in levels:
            if pct >= threshold:
                last_ts = last.get(key)
                if last_ts is None or (now - last_ts) > 3600:
                    try:
                        bal = self.balance_usd()
                        runway = self.runway_minutes()
                        runway_str = f"{runway:.0f} min" if runway != float("inf") else "∞"
                        full_msg = f"{msg}\nBalance: ${bal:.2f} | Runway: {runway_str}"
                        alert_fn(full_msg)
                        self._state.setdefault("last_alert", {})[key] = now
                        self.save()
                    except Exception as e:
                        logger.error("Alert failed: %s", e)
                break
