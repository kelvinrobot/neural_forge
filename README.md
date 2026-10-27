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




Architectural Evaluation: Mismatch of Data Virtualization Engines (Zetaris)

During the architectural design phase, enterprise data virtualization and federation platforms (specifically Zetaris) were thoroughly evaluated. My technicality and researches has classified data virtualization engines as an architectural anti-pattern for Neural Forge due to four definitive system-level constraints:

1. Per-Frame Latency Budget Exhaustion

• Neural Forge Requirement: The pipeline ingests live video at 10–30 FPS over WebSockets, dictating a strict processing budget of 33.3ms to 100ms per frame for decoding, quality filtering, color calibration, and landmark tracking.
• The Incompatibility: Zetaris is a federated SQL query engine built to translate text or structured commands into multi-source database queries (e.g., Snowflake, Oracle, S3). Query parsing, semantic mapping, remote network round-trips, and data stitching introduce latencies measured in hundreds of milliseconds or seconds, which would instantly break the real-time frame pipeline and trigger catastrophic backpressure frame drops.

2. Payload Paradigm Mismatch

• Neural Forge Requirement: The streaming pipeline handles binary image payloads (JPEG base64 byte streams) transformed into multi-dimensional NumPy arrays and PyTorch tensors for computer vision models (YOLOE, MedSigLIP).
• The Incompatibility: Zetaris virtualizes relational, tabular, and semi-structured business data (rows, columns, JSON logs). It possesses no native mechanisms to ingest high-frequency binary media streams, run local matrix operations, or handle spatial coordinate tracking.

3. State Ephemerality vs. Cold Storage Execution

• Neural Forge Requirement: The platform acts as a self-contained, stateless execution loop. To maintain absolute clinical security and compliance, raw video frames are processed entirely in-memory and instantly dropped unless specific crops are intentionally flagged (NF_SAVE_REPORT_CROPS=true).
• The Incompatibility: Zetaris operates on the assumption that data permanently resides in static distributed databases or data lakes that need to be searched. Because Neural Forge does not store patient sessions or build a persistent database of raw frames during the live exam, there is no underlying database infrastructure for a virtualization engine to connect to or query.

4. Deterministic Local Rules vs. Dynamic Semantic Queries

• Neural Forge Requirement: Agent clinical logic, step workflows, and threshold safety guardrails are explicitly defined in local configuration matrices (config/protocols/*.yaml, config/thresholds/*.yaml). These files reside inside the runtime container for instant execution and deterministic state management.
• The Incompatibility: Zetaris is designed to solve data silo problems across disparate corporate departments. Neural Forge has zero data silos; its configuration layer is centralized, lightweight, and tightly integrated into the core application code, rendering external data federation layers entirely redundant.


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
