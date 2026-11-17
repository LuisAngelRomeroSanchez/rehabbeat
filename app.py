"""RehabBeat MVP — Música adaptativa para rehabilitación ('Del Dato al Impacto')."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import streamlit as st

from src.dashboard import AdaptiveFeedbackEngine
from src.database import RehabDatabase
from src.music import MusicController
from src.tracker import BiomechTracker

st.set_page_config(page_title="RehabBeat | Del Dato al Impacto", layout="wide")


@st.cache_resource
def get_db() -> RehabDatabase:
    return RehabDatabase()


@st.cache_resource
def get_music() -> MusicController:
    return MusicController()


def init_state() -> None:
    defaults = {
        "running": False,
        "session_finished": False,
        "engine": None,
        "tracker": None,
        "session_start_ts": None,
        "last_sample_ts": -1.0,
        "last_metrics": {"angle": 0.0, "reps": 0, "rpm": 0.0, "max_angle": 0.0},
        "active_patient": None,
        "active_duration": 60,
        "last_saved_summary": None,
        "last_saved_history": None,
        "pace_multiplier": 1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def finalize_session(db: RehabDatabase, music: MusicController) -> None:
    """Detiene audio, cierra tracker, guarda resultados en SQLite (Loop 2) y prepara resumen."""
    music.stop()
    tracker: BiomechTracker | None = st.session_state.tracker
    if tracker is not None:
        tracker.close()
        st.session_state.tracker = None
    st.session_state.session_start_ts = None
    st.session_state.last_sample_ts = -1.0

    engine: AdaptiveFeedbackEngine | None = st.session_state.engine
    patient = st.session_state.active_patient
    metrics = st.session_state.last_metrics

    if engine is not None and patient is not None:
        precision = engine.compute_precision()
        db.save_session(
            patient_id=patient["id"],
            duration_sec=int(st.session_state.active_duration),
            initial_bpm=engine.initial_bpm,
            final_bpm=engine.target_bpm,
            max_angle=float(metrics.get("max_angle", 0.0)),
            total_reps=int(metrics.get("reps", 0)),
            time_to_fatigue=engine.time_to_fatigue,
            fatigue_events=engine.fatigue_events,
            rhythm_precision=precision,
        )
        st.session_state.last_saved_summary = engine.build_summary_df(
            patient_name=patient["name"],
            joint=patient["joint"],
            max_angle=float(metrics.get("max_angle", 0.0)),
            total_reps=int(metrics.get("reps", 0)),
        )
        st.session_state.last_saved_history = engine.get_history_df()

    st.session_state.running = False
    st.session_state.session_finished = True


def main() -> None:
    init_state()
    db = get_db()
    music = get_music()

    st.title("RehabBeat — Música Adaptativa para Rehabilitación")
    st.caption("Del Dato al Impacto: bucle de retroalimentación biomecánica y musical en tiempo real")

    # ---------------- SIDEBAR: MÓDULO 1 (PACIENTES Y SESIÓN) ----------------
    with st.sidebar:
        st.header("1. Paciente y Sesión")

        mode_patient = st.radio(
            "Gestión de paciente",
            ["Seleccionar existente", "Nuevo paciente"],
            horizontal=True,
            disabled=st.session_state.running,
        )

        if mode_patient == "Nuevo paciente":
            with st.form("form_nuevo_paciente"):
                p_name = st.text_input("Nombre del paciente", placeholder="Ej. Carlos Ruiz")
                p_injury = st.text_input("Tipo de lesión", placeholder="Ej. Tendinitis de hombro")
                p_joint = st.selectbox("Articulación", ["hombro", "rodilla"])
                p_angle = st.slider("Ángulo meta (°)", min_value=30, max_value=150, value=90, step=5)
                submitted = st.form_submit_button("Guardar paciente")
                if submitted:
                    try:
                        db.create_patient(p_name, p_injury or "Rehabilitación general", p_joint, p_angle)
                        st.success(f"Paciente '{p_name}' creado.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"No se pudo crear: {exc}")

        patients = db.list_patients()
        if not patients:
            st.info("Crea un paciente para comenzar.")
            return

        patient_names = {f"{p['name']} ({p['joint']})": p for p in patients}
        selected_label = st.selectbox(
            "Paciente activo",
            list(patient_names.keys()),
            disabled=st.session_state.running,
        )
        patient = patient_names[selected_label]

        # Loop 2: sugerencia automática de BPM según historial
        suggested_bpm, reason = db.suggest_initial_bpm(patient["id"], default_bpm=85)
        st.info(f"**Loop 2 (BPM sugerido):** {suggested_bpm} BPM\n\n{reason}")

        initial_bpm = st.number_input(
            "BPM inicial",
            min_value=60,
            max_value=120,
            value=int(suggested_bpm),
            step=1,
            disabled=st.session_state.running,
        )
        duration_sec = st.slider(
            "Duración de sesión (s)",
            min_value=20,
            max_value=300,
            value=60,
            step=10,
            disabled=st.session_state.running,
        )
        video_source = st.radio(
            "Fuente de video",
            ["Modo Demo (simulado)", "Webcam (en vivo)"],
            disabled=st.session_state.running,
        )

        if video_source == "Modo Demo (simulado)":
            st.session_state.pace_multiplier = st.slider(
                "Simulación ritmo paciente (Demo)",
                min_value=0.60,
                max_value=1.10,
                value=float(st.session_state.pace_multiplier),
                step=0.05,
                help="1.00 = sigue el ritmo; <0.85 = simula fatiga (>15% atraso)",
            )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Iniciar sesión", type="primary", disabled=st.session_state.running):
                st.session_state.active_patient = patient
                st.session_state.active_duration = duration_sec
                st.session_state.engine = AdaptiveFeedbackEngine(initial_bpm=float(initial_bpm))
                st.session_state.tracker = None
                st.session_state.session_start_ts = None
                st.session_state.last_sample_ts = -1.0
                st.session_state.last_metrics = {
                    "angle": 0.0,
                    "reps": 0,
                    "rpm": 0.0,
                    "max_angle": 0.0,
                }
                st.session_state.session_finished = False
                st.session_state.running = True
                st.rerun()

        with col_btn2:
            if st.button("Detener y guardar", disabled=not st.session_state.running):
                finalize_session(db, music)
                st.rerun()

    # ---------------- ÁREA PRINCIPAL ----------------
    st.markdown(
        f"**Paciente:** {patient['name']} | **Lesión:** {patient['injury_type']} | "
        f"**Articulación:** {patient['joint'].capitalize()} | **Meta:** {patient['target_angle']:.0f}°"
    )

    col_video, col_metrics = st.columns([1.2, 1.0])
    with col_video:
        video_slot = st.empty()
        track_slot = st.empty()
    with col_metrics:
        kpi_slot = st.empty()
        alert_slot = st.empty()
        chart_slot = st.empty()

    # ---------------- BUCLE EN VIVO (MÓDULOS 2 + 3 + LOOP 1) ----------------
    if st.session_state.running:
        engine: AdaptiveFeedbackEngine = st.session_state.engine
        use_demo = video_source == "Modo Demo (simulado)"
        cap = None

        if not use_demo:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.warning("No se detectó webcam disponible. Cambiando a Modo Demo automáticamente.")
                use_demo = True
                if cap is not None:
                    cap.release()
                    cap = None

        if st.session_state.tracker is None:
            st.session_state.tracker = BiomechTracker(
                joint=patient["joint"],
                target_angle=float(patient["target_angle"]),
                demo_mode=use_demo,
            )
            start_ts = time.monotonic()
            st.session_state.tracker.reset(now=start_ts)
            st.session_state.session_start_ts = start_ts
            st.session_state.last_sample_ts = -1.0

        tracker: BiomechTracker = st.session_state.tracker
        start_ts: float = st.session_state.session_start_ts
        current_track = music.play_for_bpm(engine.target_bpm)

        try:
            while st.session_state.running:
                now = time.monotonic()
                elapsed = now - start_ts
                if elapsed >= st.session_state.active_duration:
                    finalize_session(db, music)
                    st.rerun()
                    break

                if use_demo:
                    frame, metrics = tracker.generate_demo_frame(
                        target_bpm=engine.target_bpm,
                        pace_multiplier=st.session_state.pace_multiplier,
                        now=now,
                    )
                else:
                    ok, raw_frame = cap.read()
                    if not ok:
                        frame, metrics = tracker.generate_demo_frame(
                            target_bpm=engine.target_bpm,
                            pace_multiplier=1.0,
                            now=now,
                        )
                    else:
                        raw_frame = cv2.flip(raw_frame, 1)
                        frame, metrics = tracker.process_frame(raw_frame, now=now)

                st.session_state.last_metrics = metrics

                # Registrar telemetría cada ~0.5 s y evaluar Loop 1 cada 10 s
                if elapsed - st.session_state.last_sample_ts >= 0.5:
                    st.session_state.last_sample_ts = elapsed
                    event = engine.record_sample(
                        elapsed_sec=elapsed,
                        rpm=metrics["rpm"],
                        angle=metrics["angle"],
                        reps=metrics["reps"],
                        track_name=current_track["nombre"],
                    )
                    if event is not None:
                        current_track = music.play_for_bpm(engine.target_bpm)

                # Render video
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_slot.image(frame_rgb, channels="RGB", use_container_width=True)
                track_slot.info(
                    f"🎵 **Pista:** {current_track['nombre']} | "
                    f"**BPM Pista:** {current_track['bpm']:.0f} | "
                    f"**Danceability:** {current_track['danceability']:.2f}"
                )

                # Render KPIs
                with kpi_slot.container():
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Reps válidas", metrics["reps"])
                    k2.metric("RPM reales", f"{metrics['rpm']:.1f}")
                    k3.metric("BPM objetivo", f"{engine.target_bpm:.0f}")
                    k4.metric("Tiempo", f"{elapsed:.0f}/{st.session_state.active_duration}s")

                if engine.last_alert:
                    if engine.fatigue_events > 0 and "Fatiga" in engine.last_alert:
                        alert_slot.warning(engine.last_alert)
                    else:
                        alert_slot.success(engine.last_alert)

                df_hist = engine.get_history_df()
                if len(df_hist) >= 2:
                    chart_df = df_hist.set_index("tiempo_s")[["bpm_objetivo", "rpm_reales"]]
                    chart_slot.line_chart(chart_df, height=260)

                time.sleep(0.04)
        finally:
            if cap is not None:
                cap.release()

    # ---------------- MÓDULO 4: DASHBOARD RESUMEN Y EXPORTACIÓN ----------------
    if st.session_state.session_finished and st.session_state.last_saved_summary is not None:
        st.divider()
        st.subheader("4. Resumen de Sesión (Guardado en SQLite)")
        summary_df = st.session_state.last_saved_summary
        history_df = st.session_state.last_saved_history

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                "Descargar resumen CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"resumen_{patient['name'].replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )
        with c_dl2:
            if history_df is not None and not history_df.empty:
                st.download_button(
                    "Descargar telemetría RPM vs BPM (CSV)",
                    data=history_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"telemetria_{patient['name'].replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )

        if history_df is not None and not history_df.empty:
            st.line_chart(history_df.set_index("tiempo_s")[["bpm_objetivo", "rpm_reales"]], height=260)

    # Historial Loop 2 del paciente
    sessions = db.get_patient_sessions(patient["id"])
    if sessions:
        with st.expander("Historial de citas del paciente (Loop 2)", expanded=False):
            import pandas as pd

            st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
