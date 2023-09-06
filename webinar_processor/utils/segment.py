import json

from pywhispercpp.model import Segment

# Assuming the Segment class is already defined and imported

def segment_to_dict(segment: Segment) -> dict:
    return {
        "t0": segment.t0,
        "t1": segment.t1,
        "text": segment.text
    }

def segment_from_dict(data: dict) -> Segment:
    return Segment(data["t0"], data["t1"], data["text"])

def serialize_segments(segments: list[Segment]) -> str:
    return json.dumps([segment_to_dict(segment) for segment in segments], indent=4, ensure_ascii=False)

def deserialize_segments(data: str) -> list[Segment]:
    segments_dict = json.loads(data)
    return [segment_from_dict(segment) for segment in segments_dict]