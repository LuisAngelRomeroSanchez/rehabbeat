"""Módulo 2: Tracker biomecánico con MediaPipe Pose y modo demo simulado."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "pose_landmarker_lite.task"

# Pares (izquierdo, derecho) de 3 landmarks: (origen, vértice, extremo)
JOINT_LANDMARKS = {
    "hombro": (
        (
            vision.PoseLandmark.LEFT_HIP.value,
            vision.PoseLandmark.LEFT_SHOULDER.value,
            vision.PoseLandmark.LEFT_ELBOW.value,
        ),
        (
            vision.PoseLandmark.RIGHT_HIP.value,
            vision.PoseLandmark.RIGHT_SHOULDER.value,
            vision.PoseLandmark.RIGHT_ELBOW.value,
        ),
    ),
    "rodilla": (
        (
            vision.PoseLandmark.LEFT_HIP.value,
            vision.PoseLandmark.LEFT_KNEE.value,
            vision.PoseLandmark.LEFT_ANKLE.value,
        ),
        (
            vision.PoseLandmark.RIGHT_HIP.value,
            vision.PoseLandmark.RIGHT_KNEE.value,
            vision.PoseLandmark.RIGHT_ANKLE.value,
        ),
    ),
}


def calculate_angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Calcula el ángulo en grados [0, 180] en el vértice b formado por a-b-c."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    radians = np.arctan2(cy - by, cx - bx) - np.arctan2(ay - by, ax - bx)
    angle = float(np.abs(radians * 180.0 / np.pi))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


class BiomechTracker:
    """Rastrea ángulo articular, repeticiones válidas y RPM reales."""

    def __init__(
        self,
        joint: str = "hombro",
        target_angle: float = 90.0,
        model_path: Optional[Path | str] = None,
        demo_mode: bool = False,
    ) -> None:
        joint_key = joint.lower().strip()
        if joint_key not in JOINT_LANDMARKS:
            raise ValueError(f"Articulación no soportada: {joint}. Usa 'hombro' o 'rodilla'.")

        self.joint = joint_key
        self.target_angle = float(target_angle)
        self.rest_threshold = max(25.0, self.target_angle * 0.5)
        self.demo_mode = demo_mode

        self.reps = 0
        self.max_angle = 0.0
        self.current_angle = 0.0
        self.in_rep = False
        self.rep_timestamps: list[float] = []
        self.start_time = time.monotonic()
        self._smoothed_rpm = 0.0

        self._landmarker: Optional[vision.PoseLandmarker] = None
        if not self.demo_mode:
            resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo MediaPipe en {resolved_path}"
                )
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(resolved_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def reset(self, now: Optional[float] = None) -> None:
        """Reinicia contadores de la sesión."""
        self.reps = 0
        self.max_angle = 0.0
        self.current_angle = 0.0
        self.in_rep = False
        self.rep_timestamps.clear()
        self.start_time = now if now is not None else time.monotonic()
        self._smoothed_rpm = 0.0

    def close(self) -> None:
        """Libera recursos de MediaPipe."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _update_rep_state(self, angle: float, now: float) -> None:
        """Actualiza máquina de estados de repeticiones válidas y ángulo máximo."""
        self.current_angle = angle
        if angle > self.max_angle:
            self.max_angle = angle

        if not self.in_rep and angle >= self.target_angle:
            self.in_rep = True
        elif self.in_rep and angle <= self.rest_threshold:
            self.in_rep = False
            self.reps += 1
            self.rep_timestamps.append(now)

    def get_rpm(self, now: Optional[float] = None) -> float:
        """Calcula RPM reales usando ventana deslizante de 15 s con suavizado."""
        current = now if now is not None else time.monotonic()
        window_sec = 15.0
        recent = [t for t in self.rep_timestamps if current - t <= window_sec]
        if len(recent) < 2:
            self._smoothed_rpm = 0.0
            return 0.0

        span = recent[-1] - recent[0]
        if span <= 0:
            return round(self._smoothed_rpm, 1)

        avg_period = span / (len(recent) - 1)
        raw_rpm = 60.0 / avg_period
        idle_time = current - recent[-1]
        grace = max(3.0, avg_period * 2.5)
        if idle_time > grace:
            decay = max(0.0, 1.0 - (idle_time - grace) / 5.0)
            raw_rpm *= decay

        if self._smoothed_rpm <= 0.0:
            self._smoothed_rpm = raw_rpm
        else:
            self._smoothed_rpm = 0.25 * raw_rpm + 0.75 * self._smoothed_rpm

        return round(self._smoothed_rpm, 1)

    def _build_metrics(self, angle: float, rpm: float, pose_detected: bool) -> dict:
        """Construye diccionario único de métricas compartido entre overlay y UI."""
        return {
            "angle": round(angle, 1),
            "reps": int(self.reps),
            "rpm": float(rpm),
            "max_angle": round(self.max_angle, 1),
            "pose_detected": bool(pose_detected),
        }

    def _draw_overlay(
        self,
        frame: np.ndarray,
        pts_px: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None,
        metrics: dict,
    ) -> np.ndarray:
        """Dibuja los 3 landmarks, líneas articulares y métricas sobre el video."""
        out = frame.copy()
        angle = metrics["angle"]
        rpm = metrics["rpm"]
        reps = metrics["reps"]
        max_angle = metrics["max_angle"]

        if pts_px is not None:
            p1, p2, p3 = pts_px
            color_line = (0, 220, 130) if self.in_rep else (255, 190, 50)
            cv2.line(out, p1, p2, color_line, 4, cv2.LINE_AA)
            cv2.line(out, p2, p3, color_line, 4, cv2.LINE_AA)
            for idx, pt in enumerate((p1, p2, p3)):
                radius = 9 if idx == 1 else 7
                cv2.circle(out, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, pt, radius - 2, (30, 120, 255), -1, cv2.LINE_AA)

            cv2.putText(
                out,
                f"{angle:.0f} deg",
                (p2[0] + 14, p2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Panel superior compacto
        cv2.rectangle(out, (10, 10), (310, 105), (20, 24, 32), -1)
        cv2.rectangle(out, (10, 10), (310, 105), (70, 130, 220), 1)
        cv2.putText(
            out,
            f"Articulacion: {self.joint.upper()} (Meta: {self.target_angle:.0f} deg)",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Reps: {reps}   |   RPM: {rpm:.1f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (80, 240, 150),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Angulo actual: {angle:.1f} deg   Max: {max_angle:.1f} deg",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return out

    def _pick_best_side(self, landmarks: list) -> tuple[int, int, int]:
        """Elige lado izquierdo o derecho según visibilidad promedio."""
        left_ids, right_ids = JOINT_LANDMARKS[self.joint]
        left_vis = sum(getattr(landmarks[i], "visibility", 0.0) or 0.0 for i in left_ids)
        right_vis = sum(getattr(landmarks[i], "visibility", 0.0) or 0.0 for i in right_ids)
        return left_ids if left_vis >= right_vis else right_ids

    def process_frame(
        self, frame_bgr: np.ndarray, now: Optional[float] = None
    ) -> tuple[np.ndarray, dict]:
        """Procesa un frame BGR de webcam/video y devuelve frame anotado + métricas."""
        if self._landmarker is None:
            raise RuntimeError("Tracker inicializado en demo_mode=True; usa generate_demo_frame.")

        current = now if now is not None else time.monotonic()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)

        pts_px = None
        angle = self.current_angle
        pose_detected = False

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            ids = self._pick_best_side(landmarks)
            coords = [(landmarks[i].x, landmarks[i].y) for i in ids]
            raw_angle = calculate_angle(coords[0], coords[1], coords[2])
            angle = (180.0 - raw_angle) if self.joint == "rodilla" else raw_angle
            self._update_rep_state(angle, current)
            pts_px = (
                (int(coords[0][0] * w), int(coords[0][1] * h)),
                (int(coords[1][0] * w), int(coords[1][1] * h)),
                (int(coords[2][0] * w), int(coords[2][1] * h)),
            )
            pose_detected = True

        rpm = self.get_rpm(current)
        metrics = self._build_metrics(angle, rpm, pose_detected)
        annotated = self._draw_overlay(frame_bgr, pts_px, metrics)
        return annotated, metrics

    def generate_demo_frame(
        self,
        target_bpm: float,
        pace_multiplier: float = 1.0,
        width: int = 640,
        height: int = 480,
        now: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """Genera un frame sintético con esqueleto animado para demo sin webcam."""
        current = now if now is not None else time.monotonic()
        elapsed = current - self.start_time

        # Frecuencia en Hz según BPM objetivo y multiplicador de ritmo/fatiga
        freq_hz = max(0.0, (target_bpm * pace_multiplier) / 60.0)
        wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * freq_hz * elapsed))
        min_ang = 15.0
        peak_ang = self.target_angle + 10.0
        sim_angle = min_ang + wave * (peak_ang - min_ang)

        frame = np.full((height, width, 3), (26, 30, 40), dtype=np.uint8)
        cx, cy = width // 2, height // 2

        if self.joint == "hombro":
            # p1 = cadera, p2 = hombro, p3 = codo
            p2 = (cx - 20, cy - 40)
            p1 = (cx - 20, cy + 95)
            rad = math.radians(sim_angle)
            arm_len = 120
            p3 = (
                int(p2[0] + arm_len * math.sin(rad)),
                int(p2[1] + arm_len * math.cos(rad)),
            )
            # Cabeza y torso referencial
            cv2.circle(frame, (p2[0], p2[1] - 45), 24, (90, 105, 130), 2, cv2.LINE_AA)
        else:
            # rodilla: p1 = cadera, p2 = rodilla, p3 = tobillo
            p1 = (cx - 60, cy - 60)
            p2 = (cx + 40, cy - 40)
            # 0 deg = pierna extendida hacia la derecha/abajo; flexion baja el tobillo
            base_dir = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            rad = base_dir + math.radians(sim_angle)
            shin_len = 125
            p3 = (
                int(p2[0] + shin_len * math.cos(rad)),
                int(p2[1] + shin_len * math.sin(rad)),
            )

        raw_angle = calculate_angle(p1, p2, p3)
        measured_angle = (180.0 - raw_angle) if self.joint == "rodilla" else raw_angle
        self._update_rep_state(measured_angle, current)

        rpm = self.get_rpm(current)
        metrics = self._build_metrics(measured_angle, rpm, True)
        annotated = self._draw_overlay(frame, (p1, p2, p3), metrics)
        cv2.putText(
            annotated,
            f"MODO DEMO ({pace_multiplier:.2f}x ritmo)",
            (width - 230, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated, metrics
