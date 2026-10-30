"""Replay a Label Studio project's existing annotations through online training, as if each had just been submitted.

A test of the whole system on real labels: the project's annotations are fetched from the Label Studio API and fed,
oldest first, through the backend's own `fit()`, exactly as Label Studio's webhook would. Each update finishes before
more annotations are fed, so every update sees `samples_per_update` annotations, as it would at labelling speed. The
run is logged like a live one, under the config's run name with "_replay" appended.

    python -m tsam.replay --config tsam/configs/foot-ankle.yaml
"""

import argparse
import logging
import os
import time

import requests

from tsam.config import load_config
from tsam.model import NewModel

logger = logging.getLogger(__name__)


def get(path: str, **params) -> dict:
    """GET from the Label Studio API at $LABEL_STUDIO_URL, with $LABEL_STUDIO_API_KEY."""
    url = os.environ["LABEL_STUDIO_URL"].rstrip("/") + path
    response = requests.get(url, params, headers={"Authorization": f"Token {os.environ['LABEL_STUDIO_API_KEY']}"})
    response.raise_for_status()
    return response.json()


def annotated_tasks(project_id: int) -> list[dict]:
    """Every task of the project with its annotations, a page at a time."""
    tasks, page = [], 1
    while True:
        try:
            batch = get("/api/tasks", project=project_id, fields="all", page_size=100, page=page)["tasks"]
        except requests.HTTPError as e:  # past the last page
            if e.response.status_code == 404:
                return tasks
            raise
        tasks += [task for task in batch if task["annotations"]]
        page += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The online-training config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

    config = load_config(args.config)
    config.run_name += "_replay"
    NewModel.configure(config)
    project = get(f"/api/projects/{config.project_id}")
    model = NewModel(project_id=str(config.project_id), label_config=project["label_config"])

    submitted = sorted(
        ((annotation, task) for task in annotated_tasks(config.project_id) for annotation in task["annotations"]),
        key=lambda pair: pair[0]["created_at"],
    )
    logger.info(f"Replaying {len(submitted)} annotations of project {config.project_id} ({project['title']})")
    for annotation, task in submitted:
        model.fit("ANNOTATION_CREATED", {"annotation": annotation, "task": task})
        while model.trainer.updating:
            time.sleep(1)
    queued = len(model.trainer.queue)
    logger.info(f"Done: {model.trainer.step} updates; {queued} annotations left queued, short of a full update")
    model.trainer.mlflow.set_terminated(model.trainer.run_id)


if __name__ == "__main__":
    main()
