# Overnight notes: online training build (2026-10-29/30)

Decisions I made without you, for review. Most important first.

## Decisions

0. **`learning_rate` is now 1e-5, not 1e-4** (`configs/foot-ankle.yaml`). In short replays of the real Foot-Ankle annotations (2 updates of 32 annotations each, local MLflow), 1e-4 lowered test IoU on every structure and both updates were rolled back. Every rollback returns to the base model with a fresh optimizer state, so each update would repeat those first steps and be rejected again: the run would never move. 1e-5 stayed within ~0.003 of baseline; 3e-5 within ~0.007.

   | lr | baseline | update 1 | update 2 |
   |---|---|---|---|
   | 1e-5 | 0.718 | 0.716 accepted | 0.716 accepted |
   | 3e-5 | 0.718 | 0.711 accepted | 0.712 accepted |
   | 1e-4 | 0.718 | 0.698 rolled back | 0.696 rolled back |

1. **A training pair is one label per image, all its polygons combined**, not one polygon. That's how isonoai's `SAMDataset` trains T-SAM (on the `2055-train-sam-boxes-only` branch) and how the test tfrecord stores labels (`{label}/pre_scan_mask`), so the base model, online training and the test all mean the same thing by a box. The test set leaves out masks under 10 pixels, as `SAMDataset` does. Training doesn't check for them: none of the 7,592 Foot-Ankle training pairs is that small. (`data.py`, `TSAMHandler.annotation_pairs` in `model.py`)
2. **Losses ported exactly from isonoai**: the mask loss is isonoai's `dice_coefficient` (V-Net soft dice over the whole batch, squared denominator, smoothing 1), and the IoU loss is mse against `mask_iou` of the thresholded mask, as `SAMTrainer` does. (`online.py`)
3. **`Fluid` is skipped**: it isn't in `structures.mapping`, so (as agreed for unmapped labels) it's neither trained on nor tested. On Foot-Ankle that's 128 of 7,834 polygons, and some test pairs.
4. **Test boxes come from each test mask with no jitter; IoU is on the raw thresholded logits** (as agreed). Test image embeddings are computed once at startup and kept: 157 images, about 0.5 GB of memory.
5. **Rollback uses plain arrays, not `tf.train.Checkpoint`**: the accepted state is the training decoder's weights plus the optimizer's variables, copied into numpy, and restoring after a rejected update is assigning them back. *Changed 10-30, at your request:* the MLflow checkpoint is no longer that list as `state.npz`. It's now the accepted T-SAM logged as isonoai logs its models (`model/`), plus `checkpoint/` (`offsets.json`, `buffer.json`, `progress.json`), both overwritten at each accepted update. A resumed run loads its own `model/` and starts the optimizer afresh. isonoai loads `model/` with `mlflow.tensorflow.load_model` once `isonoai.models.sam` is imported; for that, the SAM classes are pickled by value (`online.py`), since isonoai has no `tsam`.
6. **MLflow through `MlflowClient` with an explicit run ID**, not `mlflow.start_run()`: updates run on a background thread, and the fluent API's active run is per thread. The config is logged whole as `config.yaml` (mlflow `log_dict`), rather than as flattened params.
7. **Relative test-set paths are under `$DATA_DIR`** (default `/data`, isonoai's convention). On your Mac that's `~/data`: `make run DATA_DIR=~/data`, and the debug config sets it. `make docker-run` mounts `DATA_DIR` at `/data`.
8. **`main.py` takes `--config` (or `$CONFIG`) instead of `--mlflow-id`**; the base model is `model_config.base_run_id`. Makefile, README, `.vscode/launch.json` and `.dockerignore` (now includes `tsam/configs/*.yaml`) follow.
9. **The prediction's model version is the online-training run ID** (`tsam-<run id>`), so a prediction says which run's model made it.
10. **`fit()` takes `ANNOTATION_CREATED` and `ANNOTATION_UPDATED`** from this config's project only. An updated annotation arrives as new pairs; its old ones stay in the replay buffer (as agreed: no edit handling).
11. **Replay (`tsam/replay.py`, `make replay`)**: fetches the project's annotations through the Label Studio API (the bulk export endpoint returns 404 on app.heartex.com, so it pages through `/api/tasks`), sorts them by `created_at`, and feeds each through the backend's own `fit()` as a webhook would. It waits for each update before feeding more, so every update sees exactly `samples_per_update` annotations. The run is named `<run_name>_replay`. Annotations left over at the end (fewer than a full update) aren't trained on.

## Findings

- **The pipeline is checked end to end**, offline: config → test set (157 images, 926 pairs) → baseline → queued annotations → background update → test → accept/rollback → MLflow metrics and checkpoint → resume from `continue_run_id`.
- **Training learns**: 30 steps on the same 64 pairs raise their IoU 0.713 → 0.814 (no augmentation), or 0.714 → 0.743 with augmentation and box jitter, so augmentation keeps images and masks aligned.
- **Real Foot-Ankle annotations convert correctly to prescan**: overlaid on the prescan image the polygons sit on the anatomy, and base T-SAM scores 0.74 IoU on them with exact boxes (0.72 on the test set).
- **Base T-SAM's test IoU is 0.718.**
- **Resuming from `continue_run_id` works**: a new trainer restores exactly the accepted weights, the replay buffer, the step and the best score, then continues the same run's step count. (Its startup test is logged at the resumed step, so that step has two `test_iou` points.)

## Overnight runs (your MLflow, experiment `sam`)

Both replay all 1,288 Foot-Ankle annotations (1,265 tasks, created 2026-08-18 to 2026-10-21), oldest first, 32 per update (~40 updates):

- `tsam_boxes_foot_ankle_online_replay` (run `eaac344569fb4e8591e860e1a7bb8505`): the config as it is, lr 1e-5.
- `tsam_boxes_foot_ankle_online_lr3e-5_replay`: the same at lr 3e-5, for comparison (config copy outside the repo).

Both ran to the end without errors: 40 updates each (7,592 new pairs, as many replayed), ~1 h 15 min per run (~1 m 50 s per update, nearly all of it S3 fetches, augmentation and the frozen encoder). The last 6 annotations were left queued, short of a full update.

| | baseline | best | last accepted | updates rolled back |
|---|---|---|---|---|
| **lr 1e-5** (`eaac3445…`) | 0.718 | 0.723 (update 37) | **0.722** (update 40) | 0 of 40 |
| lr 3e-5 (`5f1d66cc…`) | 0.718 | 0.725 (update 25) | 0.720 (update 39) | 4 of 40 (updates 4–6, 40) |

Test IoU per structure (baseline → last accepted):

| | 1e-5 | 3e-5 |
|---|---|---|
| Skin | 0.725 → **0.742** | 0.725 → 0.744 |
| Subcutaneous_Layer | 0.628 → 0.625 | 0.628 → 0.612 |
| Muscle | 0.763 → 0.761 | 0.763 → 0.758 |
| Fat | 0.571 → 0.566 | 0.571 → 0.567 |
| Ligament | 0.609 → 0.613 | 0.609 → 0.614 |
| Tendon | 0.632 → 0.638 | 0.632 → 0.646 |
| Bone | 0.890 → 0.892 | 0.890 → 0.889 |

What I read from it:

- **Online training is safe, but on this test set it's a small gain**: +0.004 test IoU at 1e-5 after all 1,288 annotations. Skin is the only clear gain (+0.017); Tendon and Ligament gain a little, and Fat and Subcutaneous_Layer lose a little. Training loss falls steadily (0.124 → 0.086 at 1e-5), so the model is learning the project's labels; little of that carries over to the test set.
- **1e-5 is the better setting**: it climbs slowly and is never rolled back. 3e-5 first drops to 0.705 (update 6; Fat −0.036, Ligament −0.028, Skin −0.022), where three updates in a row are rolled back. It then recovers and peaks higher (0.725), but swings more (Muscle 0.736 and Tendon 0.589 at its rolled-back update 40), and ends no better.
- **Acceptance did its job in the 3e-5 run**: the four rolled-back updates are the four that fell more than 0.01 below the best score. With 1e-5, the threshold never came into play.

**On Foot-Ankle's own annotations the gain is small too.** I scored the base model and both final models on the same pairs they were trained on (a one-off script in my scratchpad; it wrote nothing to MLflow).

| exact boxes, mean IoU | pairs | base | 1e-5 | 3e-5 |
|---|---|---|---|---|
| the 7,592 pairs trained on | 7,592 | 0.722 | 0.729 | 0.731 |
| Skin | 1,280 | 0.745 | 0.765 | 0.773 |
| Subcutaneous_Layer | 1,259 | 0.630 | 0.646 | 0.645 |
| Muscle | 502 | 0.670 | 0.667 | 0.681 |
| Fat | 387 | 0.594 | 0.594 | 0.587 |
| Ligament | 966 | 0.586 | 0.590 | 0.593 |
| Tendon | 1,013 | 0.660 | 0.663 | 0.671 |
| Bone | 2,185 | 0.885 | 0.885 | 0.883 |
| the last 6 annotations (never trained on) | 39 | 0.765 | 0.766 | 0.775 |

Even on labels it has seen (once new, then about once more as replay), the model moves under 1 point. So the small test gain isn't the test set disagreeing with the project. At these learning rates, online training is simply conservative.

The base model already scores about the same on this project as on the test set (0.722 vs 0.718), so Foot-Ankle may not be the distribution-shift case online training is meant for. 1e-4 adapts faster, but in the short test it cost about 0.02 test IoU and was rolled back.

This leaves a choice that's yours to make, and I changed nothing about it: keep 1e-5 as a safe, slow setting, or trade test IoU for adaptation. A project with real shift (a new probe or anatomy) would show which one matters.

### Small things I changed or noticed after the runs

- **`replay.py` now ends its MLflow run when it's done** (`set_terminated`). Both runs were left as RUNNING; I marked them FINISHED. A live backend's run stays open, as it should.
- **Both runs are re-logged in the new checkpoint format** (decision 5), from their old `state.npz`, which is still in their `checkpoint/` folders but no longer used. Loading `model/` back from your MLflow server gives exactly the accepted weights, so `continue_run_id` works with either run.
- **Not changed (rare): a rolled-back update's pairs still go into the replay buffer, and the step counter still advances.** The checkpoint is only written at an accepted update, so if the backend restarts after rollbacks, it resumes without those pairs. It then logs its next update at a step number that's already used. That's what would happen if the 3e-5 run were continued now (checkpoint at step 39, step 40 already logged).
