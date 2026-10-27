---
title: Neural Forge
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
pinned: false
short_description:  Health care visual screening backend (agents + vLLM)
---

# Neural Forge – AI Visual Diagnostic (backend)

Neural Forge is a real-time, camera-based clinical screening backend. A patient is guided step by step through an examination of one body area (eyes, face, mouth, hands, movement, posture or skin). A team of advanced models and  agents measures landmarks, colours, movements and timings, compares them with screening thresholds, and produces **screening findings with confidence scores** for clinician review.


## Architecture

```
 Client (browser/app)                        Space (one GPU container)
 ───────────────────                        ─────────────────────────────────────────────────────────────
  camera frames  ──WebSocket──▶  FastAPI (:7860, public)
  controls                         │
                                   ▼
                          SessionOrchestratorAgent  ◀── protocol YAML (steps, timing, criteria, text)
                                   │  per frame (thread pool)
                                   ▼
   FrameIngest ─▶ Quality ─▶ ColourCalibration ─▶ Shared Landmark Service ─▶ Area agents (only this area)
   (decode, fps,   (light,     (card or grey-      
    backpressure)   blur,       world, CIELAB)           
                    framing)                       
                                   │
                                   ▼  after the last step
   Aggregator (median, IQR, step records) ─▶ ReasoningAgent ( via vLLM :8001, httpx)
                                                 │  on demand: MedicalConsultantAgent 
                                                 ▼
                                             SafetyAgent (rules YAML) ─▶ ReportAgent ─▶ report JSON + summary
```



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
* `config/thresholds/*.yaml` – screening thresholds per agent (defaults pending clinical validation).
* `config/quality.yaml` – camera gating, colour calibration, aggregation.
* `config/reasoning/` – reference ranges and sign-to-condition rules.
* `config/safety_rules.yaml` – language rules, red flags, disclaimer.
* `config/zero_shot_prompts.yaml` – MedSigLIP prompts and YOLOE detection prompts.
* `.env.example` – environment variables (`NF_` prefix for the app).

My goal is to make my late grand mother proud i promised her i will save the world with my health tech solutions.
