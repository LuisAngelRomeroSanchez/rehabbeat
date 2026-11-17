# RehabBeat MVP — Del Dato al Impacto

Software de escritorio local para fisioterapeutas: música adaptativa para rehabilitación guiada por visión computacional.

## Requisitos
- Python 3.10+ (probado en Python 3.12)
- Webcam opcional (incluye **Modo Demo** simulado si no hay cámara)

## Instalación y ejecución

```bash
cd /home/lwysangel/rehab_music_mvp

# Si aún no tienes entorno virtual:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ejecutar la aplicación
.venv/bin/streamlit run app.py
```

## Estructura
- `app.py`: Interfaz Streamlit e integración en vivo.
- `src/tracker.py`: Módulo 2 — MediaPipe Pose, cálculo de ángulos, repeticiones válidas, RPM y modo demo.
- `src/music.py`: Módulo 3 — Lectura de `data/tracks.csv`, selección por BPM y reproducción local con `pygame.mixer`.
- `src/database.py`: Módulo 1 — SQLite para pacientes, sesiones y sugerencia automática de BPM (Loop 2).
- `src/dashboard.py`: Módulo 4 — Retroalimentación cada 10 s (Loop 1), gráfica RPM vs BPM y exportación CSV.
- `data/tracks.csv`: 30 pistas de ejemplo (60–120 BPM).
