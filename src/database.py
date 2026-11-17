"""Módulo 1: Persistencia SQLite de pacientes, sesiones y Loop 2 (BPM sugerido)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rehab.db"


class RehabDatabase:
    """Gestiona pacientes, historial de sesiones y recomendación de BPM (Loop 2)."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    injury_type TEXT NOT NULL,
                    joint TEXT NOT NULL,
                    target_angle REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    duration_sec INTEGER NOT NULL,
                    initial_bpm REAL NOT NULL,
                    final_bpm REAL NOT NULL,
                    max_angle REAL NOT NULL,
                    total_reps INTEGER NOT NULL,
                    time_to_fatigue REAL,
                    fatigue_events INTEGER NOT NULL DEFAULT 0,
                    rhythm_precision REAL NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
                """
            )

    def create_patient(
        self,
        name: str,
        injury_type: str,
        joint: str,
        target_angle: float,
    ) -> int:
        """Crea un paciente nuevo y devuelve su ID."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("El nombre del paciente no puede estar vacío.")
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO patients (name, injury_type, joint, target_angle, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_name, injury_type.strip(), joint.lower().strip(), float(target_angle), now),
            )
            return int(cur.lastrowid)

    def list_patients(self) -> list[dict]:
        """Lista todos los pacientes ordenados por nombre."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM patients ORDER BY name ASC").fetchall()
            return [dict(r) for r in rows]

    def get_patient(self, patient_id: int) -> Optional[dict]:
        """Obtiene un paciente por ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
            return dict(row) if row else None

    def save_session(
        self,
        patient_id: int,
        duration_sec: int,
        initial_bpm: float,
        final_bpm: float,
        max_angle: float,
        total_reps: int,
        time_to_fatigue: Optional[float],
        fatigue_events: int,
        rhythm_precision: float,
    ) -> int:
        """Guarda los resultados de una sesión (Loop 2)."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sessions (
                    patient_id, created_at, duration_sec, initial_bpm, final_bpm,
                    max_angle, total_reps, time_to_fatigue, fatigue_events, rhythm_precision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(patient_id),
                    now,
                    int(duration_sec),
                    float(initial_bpm),
                    float(final_bpm),
                    float(max_angle),
                    int(total_reps),
                    float(time_to_fatigue) if time_to_fatigue is not None else None,
                    int(fatigue_events),
                    float(rhythm_precision),
                ),
            )
            return int(cur.lastrowid)

    def get_patient_sessions(self, patient_id: int) -> list[dict]:
        """Devuelve el historial de sesiones del paciente (más reciente primero)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE patient_id = ? ORDER BY id DESC",
                (patient_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def suggest_initial_bpm(self, patient_id: int, default_bpm: int = 85) -> tuple[int, str]:
        """Loop 2: sugiere BPM inicial óptimo según historial del paciente."""
        sessions = self.get_patient_sessions(patient_id)
        if not sessions:
            return default_bpm, "Primera sesión: BPM estándar inicial (85 BPM)."

        last = sessions[0]
        precision = float(last["rhythm_precision"])
        fatigue_events = int(last["fatigue_events"])
        ttf = last["time_to_fatigue"]
        dur = max(1, int(last["duration_sec"]))
        base_bpm = float(last["final_bpm"])

        if precision >= 85.0 and (fatigue_events == 0 or (ttf is not None and ttf >= dur * 0.75)):
            suggested = int(min(120, round(base_bpm + 5)))
            reason = (
                f"Última cita: precisión {precision:.1f}% y buena resistencia. "
                f"Se sugiere subir a {suggested} BPM."
            )
        elif precision < 75.0 or (ttf is not None and ttf < dur * 0.5):
            suggested = int(max(60, round(min(base_bpm, float(last["initial_bpm"]) - 5))))
            reason = (
                f"Última cita: precisión {precision:.1f}% o fatiga temprana ({ttf or 0:.0f} s). "
                f"Se sugiere iniciar en {suggested} BPM."
            )
        else:
            suggested = int(max(60, min(120, round(base_bpm))))
            reason = (
                f"Última cita: precisión {precision:.1f}% estable. "
                f"Se sugiere mantener {suggested} BPM."
            )

        return suggested, reason
