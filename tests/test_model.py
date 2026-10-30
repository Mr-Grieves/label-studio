"""model.py: annotations into training pairs, and the backfill. S3 and the prescan conversion are stubbed."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import square

import tsam.model
from tsam.model import LS_HEIGHT, LS_WIDTH, NewModel, TSAMHandler

LABEL_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="polygon" toName="image"><Label value="Skin"/></PolygonLabels>
</View>"""


def polygon(label: str, points: list) -> dict:
    return {"type": "polygonlabels", "value": {"polygonlabels": [label], "points": points}}


def test_annotation_pairs_are_one_per_instance():
    handler = SimpleNamespace(  # post-scan pixels map straight onto the 256 frame
        locate=lambda task_data, url: ("prescan.png", "blob.json", 1),
        to_model_frame=lambda points, blob_key, ts: points / [LS_WIDTH, LS_HEIGHT] * 256,
    )
    trainer = SimpleNamespace(label_structure={"Skin": 0, "Tendon": 1, "FDT": 1})
    results = [  # points in percent of the image
        polygon("Tendon [SAM]", square(10, 10, 10)),
        polygon("Tendon", square(15, 15, 10)),  # overlaps the first: the same instance
        polygon("FDT", square(60, 60, 10)),  # Tendon too, but apart: another instance
        polygon("Skin", square(0, 0, 5)),
        polygon("Unmapped", square(30, 30, 10)),
        {"type": "rectanglelabels", "value": {"rectanglelabels": ["Skin"]}},
    ]
    pairs = TSAMHandler.annotation_pairs(handler, {}, "s3://image", results, trainer)
    assert sorted((p["structure"], len(p["polygons"])) for p in pairs) == [(0, 1), (1, 1), (1, 2)]
    assert {p["image"] for p in pairs} == {"prescan.png"}
    np.testing.assert_allclose(pairs[0]["polygons"][0][0], [25.6, 25.6])


@pytest.mark.parametrize("limit, expected", [(2, [3, 4]), (None, [0, 1, 2, 3, 4])])
def test_backfill_feeds_the_latest_annotations_oldest_first(monkeypatch, limit, expected):
    tasks = [  # annotation ids in order of creation, spread over tasks
        {"data": {}, "annotations": [{"id": 3, "created_at": "2026-10-04"}, {"id": 0, "created_at": "2026-10-01"}]},
        {"data": {}, "annotations": [{"id": 4, "created_at": "2026-10-05"}]},
        {"data": {}, "annotations": [{"id": 1, "created_at": "2026-10-02"}, {"id": 2, "created_at": "2026-10-03"}]},
    ]
    fed = []
    monkeypatch.setattr(tsam.model, "ls_get", lambda path: {"label_config": LABEL_CONFIG, "title": "test"})
    monkeypatch.setattr(tsam.model, "annotated_tasks", lambda project_id: tasks)
    monkeypatch.setattr(NewModel, "trainer", SimpleNamespace(config=SimpleNamespace(project_id=1), run_id="run"))
    monkeypatch.setattr(NewModel, "fit", lambda self, event, data: fed.append((event, data["annotation"]["id"])))
    NewModel.backfill(limit)
    assert fed == [("ANNOTATION_CREATED", i) for i in expected]
