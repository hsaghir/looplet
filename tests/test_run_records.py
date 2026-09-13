from __future__ import annotations

from looplet import LifecycleEvent, RunEnvelope
from looplet.events import EventPayload
from looplet.run_records import ArtifactRef, RunEvent, event_from_payload


def test_event_from_payload_is_json_safe() -> None:
    envelope = RunEnvelope(run_id="run-1")
    payload = EventPayload(
        event=LifecycleEvent.POST_TOOL_USE,
        step_num=2,
        tool_result={"ok": True},
        run_envelope=envelope.to_dict(),
    )

    event = event_from_payload(payload, envelope=envelope, sequence=3, timestamp=4.5)

    assert event.kind == "post_tool_use"
    assert event.sequence == 3
    assert event.timestamp == 4.5
    assert event.payload["tool_result"] == {"ok": True}
    assert event.to_dict()["envelope"] == {"run_id": "run-1"}


def test_artifact_ref_round_trips() -> None:
    ref = ArtifactRef("trace", "provenance", "/tmp/trace", "looplet.saved-artifact", 1)
    assert ArtifactRef.from_dict(ref.to_dict()) == ref


def test_run_event_round_trips() -> None:
    event = RunEvent(
        envelope=RunEnvelope(run_id="run-1"),
        sequence=1,
        timestamp=2.0,
        kind="done",
        payload={"summary": "ok"},
        artifact_refs=(ArtifactRef("a", "trace", "/tmp/a"),),
    )
    assert RunEvent.from_dict(event.to_dict()) == event
