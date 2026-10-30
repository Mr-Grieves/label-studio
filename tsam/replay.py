"""Replay all of a Label Studio project's existing annotations through online training, then exit.

A new run's backfill (model.py) with no limit: the project's annotations are fed, oldest first, through the backend's
own `fit()`, exactly as Label Studio's webhook would, in updates of `samples_per_update`. The run is logged like a live
one, under the config's run name with "_replay" appended.

    python -m tsam.replay --config tsam/configs/foot-ankle.yaml
"""

import argparse
import logging
import time

from tsam.config import load_config
from tsam.model import NewModel

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The online-training config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

    config = load_config(args.config)
    config.run_name += "_replay"
    config.trainer.backfill_limit = None
    NewModel.configure(config)
    trainer = NewModel.trainer
    while trainer.updating or len(trainer.queue) >= config.trainer.samples_per_update:
        time.sleep(1)
    queued = len(trainer.queue)
    logger.info(f"Done: {trainer.step} updates; {queued} annotations left queued, short of a full update")
    trainer.mlflow.set_terminated(trainer.run_id)


if __name__ == "__main__":
    main()
