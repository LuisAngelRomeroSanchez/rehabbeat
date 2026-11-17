"""Módulo 3: Controlador musical adaptativo con CSV y pygame.mixer."""

from __future__ import annotations

import os
import re
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pygame

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "tracks.csv"
DEFAULT_AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def synthesize_beat_wav(bpm: float, out_path: Path, beats: int = 8, sr: int = 22050) -> Path:
    """Genera un loop WAV rítmico local al BPM indicado para demo sin descargas."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    beat_dur = 60.0 / max(40.0, float(bpm))
    total_samples = int(sr * beat_dur * beats)
    t_all = np.linspace(0.0, beat_dur * beats, total_samples, endpoint=False)

    # Base armónica suave (acorde Cmaj7 sutil)
    audio = 0.08 * np.sin(2 * np.pi * 261.63 * t_all) + 0.05 * np.sin(2 * np.pi * 329.63 * t_all)

    # Pulso rítmico por cada beat (acento en beat 0 y 4)
    pulse_len = min(int(sr * 0.14), int(sr * beat_dur * 0.8))
    t_p = np.linspace(0.0, pulse_len / sr, pulse_len, endpoint=False)
    for b in range(beats):
        start = int(b * beat_dur * sr)
        freq = 523.25 if b % 4 == 0 else 392.00
        env = np.exp(-22.0 * t_p)
        click = 0.45 * np.sin(2 * np.pi * freq * t_p) * env
        end = min(total_samples, start + pulse_len)
        audio[start:end] += click[: end - start]

    audio = np.clip(audio, -0.95, 0.95)
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return out_path


class MusicController:
    """Selecciona y reproduce la pista más cercana al BPM objetivo."""

    def __init__(
        self,
        csv_path: Path | str = DEFAULT_CSV_PATH,
        audio_dir: Path | str = DEFAULT_AUDIO_DIR,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.tracks = self._load_tracks(self.csv_path)
        self.current_track: Optional[dict] = None
        self.is_playing = False
        self.audio_available = self._init_mixer()

    @staticmethod
    def _load_tracks(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "tempo" in df.columns and "bpm" not in df.columns:
            df = df.rename(columns={"tempo": "bpm"})
        required = {"nombre", "bpm", "danceability"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en CSV de pistas: {missing}")
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
        df["danceability"] = pd.to_numeric(df["danceability"], errors="coerce")
        return df.dropna(subset=["nombre", "bpm", "danceability"]).reset_index(drop=True)

    @staticmethod
    def _init_mixer() -> bool:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            return True
        except pygame.error:
            # Fallback silencioso si no hay dispositivo de audio (ej. entorno headless)
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                return True
            except pygame.error:
                return False

    def select_closest_track(self, target_bpm: float) -> dict:
        """Elige la pista con BPM más cercano (desempate por mayor danceability)."""
        df = self.tracks.copy()
        df["diff"] = (df["bpm"] - float(target_bpm)).abs()
        best = df.sort_values(by=["diff", "danceability"], ascending=[True, False]).iloc[0]
        return {
            "nombre": str(best["nombre"]),
            "bpm": float(best["bpm"]),
            "danceability": float(best["danceability"]),
        }

    def play_for_bpm(self, target_bpm: float) -> dict:
        """Cambia y reproduce la pista más cercana al BPM objetivo si es distinta a la actual."""
        track = self.select_closest_track(target_bpm)
        same_track = (
            self.current_track is not None
            and self.current_track["nombre"] == track["nombre"]
            and self.is_playing
        )
        if same_track:
            return track

        wav_name = f"{_slugify(track['nombre'])}_{int(round(track['bpm']))}.wav"
        wav_path = synthesize_beat_wav(track["bpm"], self.audio_dir / wav_name)

        self.current_track = track
        if self.audio_available:
            try:
                pygame.mixer.music.load(str(wav_path))
                pygame.mixer.music.play(loops=-1)
                self.is_playing = True
            except pygame.error:
                self.is_playing = False
        else:
            self.is_playing = True
        return track

    def stop(self) -> None:
        """Detiene la reproducción."""
        if self.audio_available and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except pygame.error:
                pass
        self.is_playing = False
