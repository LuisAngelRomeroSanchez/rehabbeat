"""Módulo 4: Lógica de retroalimentación Loop 1, métricas y resumen exportable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class AdaptiveFeedbackEngine:
    """Ejecuta el Loop 1 cada 10 s comparando BPM objetivo vs RPM reales."""

    initial_bpm: float = 85.0
    eval_interval_sec: float = 10.0
    warmup_sec: float = 5.0
    min_bpm: float = 60.0
    max_bpm: float = 120.0

    target_bpm: float = field(init=False)
    last_eval_sec: float = 0.0
    fatigue_events: int = 0
    time_to_fatigue: Optional[float] = None
    last_alert: Optional[str] = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.target_bpm = float(self.initial_bpm)

    def record_sample(
        self,
        elapsed_sec: float,
        rpm: float,
        angle: float,
        reps: int,
        track_name: str,
    ) -> Optional[dict]:
        """Registra un punto de telemetría y evalúa Loop 1 si pasaron 10 s."""
        self.history.append(
            {
                "tiempo_s": round(elapsed_sec, 1),
                "bpm_objetivo": round(self.target_bpm, 1),
                "rpm_reales": round(rpm, 1) if elapsed_sec >= self.warmup_sec else None,
                "angulo": round(angle, 1),
                "repeticiones": int(reps),
                "pista": track_name,
            }
        )

        if elapsed_sec - self.last_eval_sec < self.eval_interval_sec:
            return None

        self.last_eval_sec = elapsed_sec
        deviation = self.target_bpm - rpm
        lag_ratio = deviation / max(1.0, self.target_bpm)

        if lag_ratio > 0.15:
            self.fatigue_events += 1
            if self.time_to_fatigue is None:
                self.time_to_fatigue = round(elapsed_sec, 1)
            new_bpm = max(self.min_bpm, self.target_bpm - 10.0)
            event_type = "fatiga"
            self.last_alert = (
                f"Fatiga detectada ({elapsed_sec:.0f} s): RPM {rpm:.1f} vs "
                f"{self.target_bpm:.0f} BPM (-{lag_ratio * 100:.0f}%). "
                f"Bajando a {new_bpm:.0f} BPM."
            )
        else:
            new_bpm = min(self.max_bpm, self.target_bpm + 5.0)
            event_type = "progreso"
            self.last_alert = (
                f"Ritmo sostenido ({elapsed_sec:.0f} s): subiendo a {new_bpm:.0f} BPM."
            )

        self.target_bpm = new_bpm
        return {
            "event": event_type,
            "elapsed_sec": round(elapsed_sec, 1),
            "deviation": round(deviation, 1),
            "new_bpm": new_bpm,
            "message": self.last_alert,
        }

    def get_history_df(self) -> pd.DataFrame:
        """Devuelve la serie temporal de la sesión."""
        if not self.history:
            return pd.DataFrame(
                columns=["tiempo_s", "bpm_objetivo", "rpm_reales", "angulo", "repeticiones", "pista"]
            )
        return pd.DataFrame(self.history)

    def compute_precision(self) -> float:
        """Calcula precisión rítmica promedio (0-100%), ignorando ventana inicial."""
        valid = [row for row in self.history if row["rpm_reales"] is not None]
        if not valid:
            return 0.0
        errors = [
            abs(row["bpm_objetivo"] - row["rpm_reales"]) / max(1.0, row["bpm_objetivo"])
            for row in valid
        ]
        mean_err = sum(errors) / len(errors)
        return round(max(0.0, min(100.0, (1.0 - mean_err) * 100.0)), 1)

    def build_summary_df(
        self,
        patient_name: str,
        joint: str,
        max_angle: float,
        total_reps: int,
    ) -> pd.DataFrame:
        """Construye tabla resumen exportable a CSV."""
        return pd.DataFrame(
            [
                {
                    "paciente": patient_name,
                    "articulacion": joint,
                    "bpm_inicial": round(self.initial_bpm, 1),
                    "bpm_final": round(self.target_bpm, 1),
                    "repeticiones": int(total_reps),
                    "angulo_max_deg": round(max_angle, 1),
                    "precision_ritmica_pct": self.compute_precision(),
                    "eventos_fatiga": int(self.fatigue_events),
                    "tiempo_hasta_fatiga_s": (
                        self.time_to_fatigue if self.time_to_fatigue is not None else "N/A"
                    ),
                }
            ]
        )
