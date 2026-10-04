---
title: Neural Forge
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
pinned: false
short_description: Educational webcam visual screening backend (agents + vLLM)
---

# Neural Forge – AI Visual Diagnostic (backend)

Neural Forge is an **educational**, real-time, webcam-based visual screening backend. A patient is guided step by step through an examination of one body area (eyes, face, mouth, hands, movement, posture or skin). A team of Python agents measures landmarks, colours, movements and timings, compares them with educational thresholds, and produces **screening flags with confidence scores, never diagnoses**.

> **Educational disclaimer.** Neural Forge is not a medical device. It does not diagnose, treat or rule out any condition. Webcam measurements can be wrong. Always consult a qualified health professional, and seek emergency care for sudden or severe symptoms.

## Architecture

```
 Client (browser/app)                       Hugging Face Space (one GPU container)
 ───────────────────                        ─────────────────────────────────────────────────────────────
  camera frames  ──WebSocket──▶  FastAPI (:7860, public)
  controls                         │
                                   ▼
                          SessionOrchestratorAgent  ◀── protocol YAML (steps, timing, criteria, text)
                                   │  per frame (thread pool)
                                   ▼
   FrameIngest ─▶ Quality ─▶ ColourCalibration ─▶ Shared Landmark Service ─▶ Area agents (only this area)
   (decode, fps,   (light,     (card or grey-      (MediaPipe Face/Hands       (measurements, rolling
    backpressure)   blur,       world, CIELAB)      VIDEO mode, RTMW pose,      buffers, real timestamps)
                    framing)                        each at most once/frame)
                                   │
                                   ▼  after the last step
   Aggregator (median, IQR, step records) ─▶ ReasoningAgent (Qwen3-VL-4B via vLLM :8001, httpx)
                                                 │  on demand: MedicalConsultantAgent (MedGemma 1.5 4B via vLLM :8002)
                                                 ▼
                                             SafetyAgent (rules YAML) ─▶ ReportAgent ─▶ report JSON + summary
```

Perception models: MediaPipe Face Landmarker (478 landmarks, iris, 52 blendshapes), MediaPipe Hand Landmarker, RTMW whole-body pose (rtmlib + onnxruntime-gpu), YOLOE open-vocabulary lesion finder, MedSigLIP-448 zero-shot classifier (swappable for trained heads). LLMs run in separate vLLM processes; the API never loads them.

## Examination flow

For every step of `config/protocols/<area>.yaml` the orchestrator:

1. sends `step_preview` (instruction, tips, safety note, camera setup);
2. waits for a `ready` control if the step requires repositioning;
3. gates on camera quality, sending `quality_feedback` until the view is acceptable;
4. counts down with `get_ready`, then sends `step_started`;
5. sends `step_progress` about once per second (percent complete, seconds remaining, timed prompts);
6. completes when every active agent has enough data **and** the minimum duration has passed;
7. on timeout sends `step_repeat` with specific guidance and repeats once, otherwise marks the step incomplete;
8. sends `step_complete` with a preview of the next step.

After the last step: `examination_complete` → `analysis_started` → `report_ready`. `pause`/`resume` freeze the timer, `skip_step` skips optional steps, `stop` ends early (a report is built from completed steps), and `add_area` repeats the skin "area of concern" step.

## API reference

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/v1/sessions` | Create a session. Body: `area`, `consent` (must be `true`), `patient_context` (`age`, `sex`, `symptoms`), `preferred_language` (default `en`). Returns `session_id` and localised steps. |
| WS | `/api/v1/ws/{area}/{session_id}` | Real-time examination stream. |
| GET | `/api/v1/sessions/{id}/summary` | Aggregated findings JSON. |
| GET | `/api/v1/sessions/{id}/report` | Final report (after analysis). |
| POST | `/api/v1/sessions/{id}/question` | Follow-up question answered from the session findings. |
| DELETE | `/api/v1/sessions/{id}` | Delete the session and all of its data. |
| GET | `/api/v1/protocols/{area}?language=en` | Protocol steps for an area. |
| GET | `/api/v1/health` | `starting` until both vLLM servers respond, then `ready`; models, GPU and reachability. |

Areas: `eyes`, `face`, `mouth`, `hands`, `movement`, `posture`, `skin`.

## WebSocket message protocol

All messages are JSON objects with a `type` field.

Client → server:

```json
{"type": "frame", "session_id": "…", "timestamp_ms": 1712345678901, "image_base64": "<JPEG base64>"}
{"type": "control", "action": "start | ready | pause | resume | skip_step | add_area | stop"}
{"type": "reference_card_present", "present": true}
```

Server → client: `step_preview`, `quality_feedback`, `get_ready`, `step_started`, `step_progress`, `live_reading`, `step_repeat`, `step_complete`, `examination_complete`, `analysis_started`, `report_ready`, `error`, and `keepalive`. Definitions are in `neural_forge/schemas/messages.py`.

Send frames as JPEG (recommended 640–960 px wide, 10–30 fps) with the camera's real capture timestamp; all frequency and timing analysis uses these measured timestamps. Frames are dropped when the pipeline is behind, and frames are never stored unless `NF_SAVE_REPORT_CROPS=true` (only report crops).

## Configuration

* `config/protocols/*.yaml` – patient-facing steps (multilingual text keyed by language code).
* `config/thresholds/*.yaml` – educational screening thresholds per agent.
* `config/quality.yaml` – camera gating, colour calibration, aggregation.
* `config/reasoning/` – reference ranges and sign-to-condition rules.
* `config/safety_rules.yaml` – language rules, red flags, disclaimer.
* `config/zero_shot_prompts.yaml` – MedSigLIP prompts and YOLOE detection prompts.
* `.env.example` – environment variables (`NF_` prefix for the app).

## Deployment

The project is deployed only as a Hugging Face Space with the Docker SDK on a GPU (24 GB L4 or A10G). See [`deploy/HUGGINGFACE_SETUP.md`](deploy/HUGGINGFACE_SETUP.md).

## Tests

Unit tests live in `tests/` and use synthetic landmark sequences (no model weights required): `pytest`.
