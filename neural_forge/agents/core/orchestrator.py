"""Session orchestrator agent: runs the patient-facing examination protocol step by step."""

import asyncio
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from neural_forge.core.exceptions import ExaminationStopped, QualityTimeout, StepSkipped
from neural_forge.core.pipeline import ExaminationPipeline
from neural_forge.core.protocol_loader import make_step_instance
from neural_forge.core.settings import Settings
from neural_forge.schemas.findings import StepRecord, StepStatus
from neural_forge.schemas.messages import (
    ControlAction,
    ErrorMessage,
    ExaminationCompleteMessage,
    GetReadyMessage,
    LiveReadingMessage,
    NextStepPreview,
    QualityFeedbackMessage,
    ServerMessage,
    StepCompleteMessage,
    StepPreviewMessage,
    StepProgressMessage,
    StepRepeatMessage,
    StepStartedMessage,
)
from neural_forge.schemas.protocols import ProtocolStep, localize, localize_list
from neural_forge.schemas.readings import QualityAssessment, Reading

SendCallable = Callable[[ServerMessage], Awaitable[None]]


@dataclass
class CollectionOutcome:
    """Result of one data-collection attempt."""

    completed: bool
    reason: str | None
    guidance: list[str]
    valid_frames: int
    duration_seconds: float
    quality_issues: list[str] = field(default_factory=list)


@dataclass
class ExaminationOutcome:
    """Result of running the whole protocol."""

    started: bool
    stopped: bool
    step_records: list[StepRecord]


class SessionOrchestratorAgent:
    """Drives previews, ready confirmation, quality gating, countdowns, progress, repeats and completion."""

    def __init__(self, pipeline: ExaminationPipeline, send: SendCallable, settings: Settings) -> None:
        self._pipeline = pipeline
        self._protocol = pipeline.protocol
        self._ingest = pipeline.frame_ingest
        self._messages = pipeline.messages
        self._language = pipeline.language
        self._send_message = send
        self._settings = settings
        self._controls: asyncio.Queue[ControlAction] = asyncio.Queue()
        self._paused = False
        self._pause_started: float | None = None
        self._paused_accumulated = 0.0
        self._records: list[StepRecord] = []
        self._encouragement_index = 0
        self._current_step: tuple[ProtocolStep, str, int] | None = None

    @property
    def step_records(self) -> list[StepRecord]:
        """Records of all step instances run so far."""
        return list(self._records)

    def submit_control(self, action: ControlAction) -> None:
        """Queue a control action received from the client."""
        self._controls.put_nowait(action)

    def set_reference_card_hint(self, present: bool) -> None:
        """Forward the client's reference card hint to the pipeline."""
        self._pipeline.set_reference_card_hint(present)

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def _text(self, step_field: dict[str, str] | None) -> str:
        return localize(step_field, self._language) or ""

    async def _send(self, message: ServerMessage) -> None:
        await self._send_message(message)

    def _next_control_nowait(self) -> ControlAction | None:
        try:
            return self._controls.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def run(self) -> ExaminationOutcome:
        """Run the full protocol and return the step records."""
        try:
            await self._wait_for_start()
        except ExaminationStopped:
            return ExaminationOutcome(started=False, stopped=True, step_records=[])
        steps = self._protocol.steps
        stopped = False
        try:
            for index, step in enumerate(steps):
                next_step = steps[index + 1] if index + 1 < len(steps) else None
                repetition = 1
                while True:
                    step_instance = make_step_instance(step.id, repetition)
                    can_add_area = step.repeatable and repetition < step.max_repetitions
                    record = await self._run_step(step, step_instance, repetition, next_step, can_add_area)
                    self._records.append(record)
                    successful = record.status in (StepStatus.COMPLETED, StepStatus.REPEATED)
                    if can_add_area and successful and await self._offer_repetition():
                        repetition += 1
                        continue
                    break
        except ExaminationStopped:
            stopped = True
            self._record_stop()
        completed = sum(1 for record in self._records if record.status == StepStatus.COMPLETED)
        repeated = sum(1 for record in self._records if record.status == StepStatus.REPEATED)
        skipped = sum(1 for record in self._records if record.status == StepStatus.SKIPPED)
        incomplete = sum(
            1 for record in self._records if record.status in (StepStatus.INCOMPLETE, StepStatus.NOT_REACHED)
        )
        await self._send(
            ExaminationCompleteMessage(
                completed_steps=completed,
                repeated_steps=repeated,
                skipped_steps=skipped,
                incomplete_steps=incomplete,
                stopped=stopped,
                message=self._messages.text("examination_stopped" if stopped else "examination_complete", self._language),
            )
        )
        return ExaminationOutcome(started=True, stopped=stopped, step_records=list(self._records))

    def _record_stop(self) -> None:
        recorded_ids = {record.step_id for record in self._records}
        if self._current_step is not None:
            step, step_instance, attempts = self._current_step
            self._records.append(
                StepRecord(
                    step_id=step.id,
                    step_instance=step_instance,
                    step_number=step.step_number,
                    title=self._text(step.title),
                    status=StepStatus.INCOMPLETE,
                    attempts=attempts,
                    reason=self._messages.text("reason_stopped_by_patient", self._language),
                )
            )
            recorded_ids.add(step.id)
            self._current_step = None
        for step in self._protocol.steps:
            if step.id not in recorded_ids:
                self._records.append(
                    StepRecord(
                        step_id=step.id,
                        step_instance=step.id,
                        step_number=step.step_number,
                        title=self._text(step.title),
                        status=StepStatus.NOT_REACHED,
                        attempts=0,
                        reason=self._messages.text("reason_not_reached", self._language),
                    )
                )

    async def _wait_for_control(self, accepted: set[ControlAction], timeout: float, step: ProtocolStep | None) -> ControlAction | None:
        deadline = self._now() + timeout
        while self._now() < deadline:
            self._ingest.drain()
            try:
                action = await asyncio.wait_for(self._controls.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            if action == ControlAction.STOP:
                raise ExaminationStopped()
            if action in accepted:
                return action
            if step is not None:
                await self._handle_common_control(action, step)
                if self._paused:
                    deadline += 0.2
        return None

    async def _wait_for_start(self) -> None:
        action = await self._wait_for_control({ControlAction.START}, self._settings.start_timeout_seconds, None)
        if action is None:
            raise ExaminationStopped()

    async def _wait_for_ready(self, step: ProtocolStep) -> None:
        action = await self._wait_for_control({ControlAction.READY}, self._settings.ready_timeout_seconds, step)
        if action is None:
            raise ExaminationStopped()

    async def _offer_repetition(self) -> bool:
        action = await self._wait_for_control(
            {ControlAction.ADD_AREA, ControlAction.READY, ControlAction.START, ControlAction.SKIP_STEP},
            self._settings.repeatable_offer_timeout_seconds,
            None,
        )
        return action == ControlAction.ADD_AREA

    async def _handle_common_control(self, action: ControlAction, step: ProtocolStep) -> None:
        if action == ControlAction.STOP:
            raise ExaminationStopped()
        if action == ControlAction.PAUSE and not self._paused:
            self._paused = True
            self._pause_started = self._now()
        elif action == ControlAction.RESUME and self._paused:
            if self._pause_started is not None:
                self._paused_accumulated += self._now() - self._pause_started
            self._paused = False
            self._pause_started = None
        elif action == ControlAction.SKIP_STEP:
            if step.skippable:
                raise StepSkipped()
            await self._send(
                ErrorMessage(
                    code="step_not_skippable",
                    message=self._messages.text("step_not_skippable", self._language),
                    recoverable=True,
                )
            )

    async def _process_controls(self, step: ProtocolStep) -> None:
        while True:
            action = self._next_control_nowait()
            if action is None:
                return
            await self._handle_common_control(action, step)

    async def _send_preview(self, step: ProtocolStep, step_instance: str, repetition: int, attempt: int) -> None:
        await self._send(
            StepPreviewMessage(
                step_id=step.id,
                step_instance=step_instance,
                step_number=step.step_number,
                total_steps=self._protocol.total_steps,
                title=self._text(step.title),
                instruction=self._text(step.instruction),
                tips=localize_list(step.tips, self._language),
                safety_note=localize(step.safety_note, self._language),
                camera_setup=step.camera_setup,
                requires_patient_ready=step.requires_patient_ready,
                skippable=step.skippable,
                repetition=repetition,
                attempt=attempt,
            )
        )

    def _next_step_preview(self, next_step: ProtocolStep | None) -> NextStepPreview | None:
        if next_step is None:
            return None
        return NextStepPreview(
            step_id=next_step.id,
            step_number=next_step.step_number,
            title=self._text(next_step.title),
            instruction=self._text(next_step.instruction),
        )

    async def _run_step(
        self,
        step: ProtocolStep,
        step_instance: str,
        repetition: int,
        next_step: ProtocolStep | None,
        can_add_area: bool,
    ) -> StepRecord:
        self._current_step = (step, step_instance, 0)
        await self._send_preview(step, step_instance, repetition, attempt=1)
        outcome: CollectionOutcome | None = None
        status = StepStatus.INCOMPLETE
        attempts = 0
        for attempt in (1, 2):
            attempts = attempt
            self._current_step = (step, step_instance, attempt)
            try:
                if step.requires_patient_ready:
                    await self._wait_for_ready(step)
                await self._wait_for_quality(step, step_instance)
                await self._countdown(step)
                outcome = await self._collect(step, step_instance)
            except StepSkipped:
                self._pipeline.end_step(step, step_instance, completed=False)
                record = StepRecord(
                    step_id=step.id,
                    step_instance=step_instance,
                    step_number=step.step_number,
                    title=self._text(step.title),
                    status=StepStatus.SKIPPED,
                    attempts=attempt,
                    reason=self._messages.text("reason_skipped_by_patient", self._language),
                )
                await self._send(
                    StepCompleteMessage(
                        step_id=step.id,
                        step_instance=step_instance,
                        status=StepStatus.SKIPPED,
                        message=self._messages.text("step_skipped", self._language),
                        next_step=self._next_step_preview(next_step),
                        can_add_area=False,
                    )
                )
                self._current_step = None
                return record
            except QualityTimeout as timeout:
                outcome = CollectionOutcome(
                    completed=False,
                    reason=self._messages.text("reason_quality_timeout", self._language),
                    guidance=timeout.guidance,
                    valid_frames=0,
                    duration_seconds=0.0,
                    quality_issues=timeout.issues,
                )
            if outcome.completed:
                status = StepStatus.COMPLETED if attempt == 1 else StepStatus.REPEATED
                break
            if attempt == 1:
                guidance = list(outcome.guidance) + localize_list(step.repeat_guidance, self._language)
                await self._send(
                    StepRepeatMessage(
                        step_id=step.id,
                        attempt=2,
                        reason=outcome.reason or self._messages.text("repeat_reason_default", self._language),
                        guidance=list(dict.fromkeys(guidance)),
                    )
                )
                await self._send_preview(step, step_instance, repetition, attempt=2)
        if status in (StepStatus.COMPLETED, StepStatus.REPEATED):
            message = self._messages.encouragement(self._encouragement_index, self._language)
            self._encouragement_index += 1
        else:
            message = self._messages.text("step_incomplete", self._language)
        record = StepRecord(
            step_id=step.id,
            step_instance=step_instance,
            step_number=step.step_number,
            title=self._text(step.title),
            status=status,
            attempts=attempts,
            reason=None if status in (StepStatus.COMPLETED, StepStatus.REPEATED) else (outcome.reason if outcome else None),
            duration_seconds=outcome.duration_seconds if outcome else 0.0,
            valid_frames=outcome.valid_frames if outcome else 0,
            quality_issues=outcome.quality_issues if outcome else [],
        )
        await self._send(
            StepCompleteMessage(
                step_id=step.id,
                step_instance=step_instance,
                status=status,
                message=message,
                next_step=self._next_step_preview(next_step),
                can_add_area=can_add_area and status in (StepStatus.COMPLETED, StepStatus.REPEATED),
            )
        )
        self._current_step = None
        return record

    async def _wait_for_quality(self, step: ProtocolStep, step_instance: str) -> None:
        deadline = self._now() + self._settings.quality_wait_timeout_seconds
        consecutive_good = 0
        last_feedback = -math.inf
        last_quality: QualityAssessment | None = None
        required = self._pipeline.quality.confirmation_frames
        while True:
            await self._process_controls(step)
            if self._paused:
                self._ingest.drain()
                await asyncio.sleep(0.1)
                deadline += 0.1
                continue
            if self._now() > deadline:
                issues = last_quality.issues if last_quality is not None else ["no_frames"]
                guidance = last_quality.guidance if last_quality is not None else self._pipeline.quality.guidance_for(["no_frames"])
                raise QualityTimeout(issues=issues, guidance=guidance)
            raw_frame = await self._ingest.next_frame(timeout=0.2)
            if raw_frame is None:
                continue
            result = await self._pipeline.process_frame(raw_frame, step, step_instance, collect=False)
            last_quality = result.quality
            if result.quality.usable:
                consecutive_good += 1
                if consecutive_good >= required:
                    await self._send(
                        QualityFeedbackMessage(
                            step_id=step.id,
                            acceptable=True,
                            issues=[],
                            guidance=[self._messages.text("quality_ok", self._language)],
                            brightness=result.quality.brightness,
                            blur_variance=result.quality.blur_variance,
                            subject_present=result.quality.subject_present,
                            estimated_distance_m=result.quality.estimated_distance_m,
                        )
                    )
                    return
            else:
                consecutive_good = 0
                if self._now() - last_feedback >= self._settings.quality_feedback_interval_seconds:
                    await self._send_quality_feedback(step, result.quality)
                    last_feedback = self._now()

    async def _send_quality_feedback(self, step: ProtocolStep, quality: QualityAssessment) -> None:
        await self._send(
            QualityFeedbackMessage(
                step_id=step.id,
                acceptable=quality.usable,
                issues=quality.issues,
                guidance=quality.guidance,
                brightness=quality.brightness,
                blur_variance=quality.blur_variance,
                subject_present=quality.subject_present,
                estimated_distance_m=quality.estimated_distance_m,
            )
        )

    async def _countdown(self, step: ProtocolStep) -> None:
        total = step.get_ready_seconds
        for remaining in range(total, 0, -1):
            await self._send(GetReadyMessage(step_id=step.id, countdown_seconds=total, seconds_remaining=remaining))
            tick_end = self._now() + 1.0
            while self._now() < tick_end:
                await self._process_controls(step)
                if self._paused:
                    tick_end += 0.05
                self._ingest.drain()
                await asyncio.sleep(0.05)

    def _active_prompt(self, step: ProtocolStep, elapsed: float) -> tuple[int, str | None]:
        active_index = -1
        for index, prompt in enumerate(step.prompts):
            if prompt.at_seconds <= elapsed:
                active_index = index
        if active_index < 0:
            return -1, None
        return active_index, localize(step.prompts[active_index].text, self._language)

    async def _send_live_readings(self, step: ProtocolStep, readings: list[Reading], last_sent: dict[tuple[str, str], float]) -> None:
        now = self._now()
        sent = 0
        for reading in readings:
            key = (reading.agent, reading.metric)
            if now - last_sent.get(key, -math.inf) < self._settings.live_reading_interval_seconds:
                continue
            last_sent[key] = now
            await self._send(
                LiveReadingMessage(
                    step_id=step.id,
                    agent=reading.agent,
                    metric=reading.metric,
                    value=reading.value,
                    unit=reading.unit,
                    confidence=reading.confidence,
                    description=reading.description,
                )
            )
            sent += 1
            if sent >= 5:
                return

    async def _collect(self, step: ProtocolStep, step_instance: str) -> CollectionOutcome:
        self._pipeline.begin_step(step, step_instance)
        self._ingest.drain()
        self._paused_accumulated = 0.0
        start = self._now()
        _, first_prompt = self._active_prompt(step, 0.0)
        await self._send(
            StepStartedMessage(
                step_id=step.id,
                step_instance=step_instance,
                step_number=step.step_number,
                min_duration_seconds=step.min_duration_seconds,
                max_duration_seconds=step.max_duration_seconds,
                prompt=first_prompt,
            )
        )
        valid_frames = 0
        last_progress = -math.inf
        last_feedback = -math.inf
        last_frame_time = start
        prompt_index = 0 if first_prompt is not None else -1
        live_last_sent: dict[tuple[str, str], float] = {}
        issue_counter: Counter[str] = Counter()
        criteria = step.completion_criteria
        while True:
            await self._process_controls(step)
            now = self._now()
            if self._paused:
                self._ingest.drain()
                paused_elapsed = now - start - self._paused_accumulated - (now - (self._pause_started or now))
                if now - last_progress >= self._settings.progress_interval_seconds:
                    await self._send(
                        StepProgressMessage(
                            step_id=step.id,
                            percent_complete=0.0,
                            seconds_elapsed=round(max(0.0, paused_elapsed), 2),
                            seconds_remaining=round(max(0.0, step.min_duration_seconds - paused_elapsed), 2),
                            paused=True,
                        )
                    )
                    last_progress = now
                await asyncio.sleep(0.1)
                continue
            raw_frame = await self._ingest.next_frame(timeout=0.1)
            now = self._now()
            if raw_frame is not None:
                last_frame_time = now
                result = await self._pipeline.process_frame(raw_frame, step, step_instance, collect=True)
                if result.quality.usable:
                    valid_frames += 1
                    if result.readings:
                        await self._send_live_readings(step, result.readings, live_last_sent)
                else:
                    issue_counter.update(result.quality.issues)
                    if now - last_feedback >= self._settings.quality_feedback_interval_seconds:
                        await self._send_quality_feedback(step, result.quality)
                        last_feedback = now
            elif now - last_frame_time > self._settings.frame_starvation_seconds:
                await self._send(
                    ErrorMessage(code="no_frames", message=self._messages.text("no_frames", self._language), recoverable=True)
                )
                last_frame_time = now
            now = self._now()
            elapsed = now - start - self._paused_accumulated
            new_prompt_index, prompt_text = self._active_prompt(step, elapsed)
            prompt_changed = new_prompt_index != prompt_index
            prompt_index = new_prompt_index
            progress = self._pipeline.step_progress(step, step_instance, valid_frames)
            time_fraction = min(1.0, elapsed / step.min_duration_seconds)
            if prompt_changed or now - last_progress >= self._settings.progress_interval_seconds:
                if elapsed < step.min_duration_seconds:
                    seconds_remaining = step.min_duration_seconds - elapsed
                else:
                    seconds_remaining = 0.0 if progress.complete else step.max_duration_seconds - elapsed
                await self._send(
                    StepProgressMessage(
                        step_id=step.id,
                        percent_complete=round(100.0 * min(progress.fraction, time_fraction), 1),
                        seconds_elapsed=round(elapsed, 2),
                        seconds_remaining=round(max(0.0, seconds_remaining), 2),
                        current_prompt=prompt_text,
                        paused=False,
                    )
                )
                last_progress = now
            top_issues = [issue for issue, _ in issue_counter.most_common(3)]
            if elapsed >= step.min_duration_seconds and progress.complete:
                self._pipeline.end_step(step, step_instance, completed=True)
                return CollectionOutcome(True, None, [], valid_frames, elapsed, top_issues)
            if elapsed >= step.max_duration_seconds:
                if criteria.complete_on_max_duration and valid_frames >= criteria.min_valid_frames:
                    self._pipeline.end_step(step, step_instance, completed=True)
                    return CollectionOutcome(True, None, [], valid_frames, elapsed, top_issues)
                self._pipeline.end_step(step, step_instance, completed=False)
                reason = " ".join(progress.reasons) if progress.reasons else self._messages.text("repeat_reason_default", self._language)
                guidance = self._pipeline.quality.guidance_for(top_issues)
                return CollectionOutcome(False, reason, guidance, valid_frames, elapsed, top_issues)
