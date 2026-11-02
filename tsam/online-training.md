# Plan: Online Learning

## Goal

One of the main limitations of T-SAM is distribution shift: if we start a project on a new part of the body that T-SAM has never been trained on, it will struggle to accurately segment certain structures. The goal here is to adapt the currently running instance of T-SAM to whatever project it is running on. Through online learning of the labels as they get accepted, we plan to finetune T-SAM to get better and better at segmenting anatomy for this project.

## What will we need before we start?

- [x] A labelled the test set. This should be done by hand, and should be highly accurate. Once complete, export it for hold-out evaluation of the adaptive model.
- [x] A trained T-SAM model to start from. This model should be pulled fresh for each project, and fine-tuned (mask-decoder only, prompt encoder maybe, image encoder frozen) on batches of data as they come in.
- [x] A training set with defined labelling config, and an instance of this backend connected and running.

## How should this backend work?

- All the code for training will reside here, not in `isonoai`. This will port over the minimal amount of code set required to get this working.
- We will need to implement the following:
  - [ ] A yaml configuration file, similar to our training configs where we can define everything we need.
  - [ ] A tester that evaluates the current T-SAM model on the holdout test set defined in the config. Testing will be done by drawing the minimum bounding box around every label in an image and calculating its IoU. This should be tracked per structure so we can see which anatomy it is most struggling on.
    - The test set is a tfrecord of prescan images and labels only. Boxes are the tight box around each mask in the model's 256 prescan frame, built as training builds them but without jitter. Each test pair's accepted masks (see Structures and accepted masks) are drawn once, with a fixed seed, so every test is alike. IoU is on the model's raw output (thresholded logits), against the whole mask as isonoai's SAMTester scores it, not the backend's clean-up: comparisons are all relative.
    - The image encoder is frozen and test images aren't augmented, so their embeddings are computed once at startup and kept. Each re-test is then only decoder passes.
  - [ ] Code to setup the model for training. Don't need any callbacks. Just an optimizer and loss function definition. Losses: dice on the mask (per instance, averaged, so a small instance counts as much as a large one, as in the test IoU), and mse on the predicted IoU (weight 0.1), which keeps the score sent back with each prediction in step with the masks.
  - [ ] MLFlow logging. The instance should only create one new mlflow run per project (unless we are running multiple tests). The idea being that you would be able to pull up the run for "tsam_foot_ankle_online" and see how the finetuned model is progressing over the weeks of labelling time. The main metrics we will be logging are the `test_iou`'s of the model over time, but also training loss and other common metrics/parameters.
  - [ ] A basic training step that gets called every time the next batch of labels is ready. This should run every label in the samples collected through the model. Number of passes defined in config.
    - Labels arrive only through the API: Label Studio's webhook on annotation submit/update calls the backend's `fit()`, which queues the new samples and returns straight away. A batch is ready once `samples_per_update` new annotations are queued; training runs in a background thread and takes that many.
    - As soon as a label arrives it is converted to the prescan frame (its polygon, and later its box), and everything from then on works on prescan data only.
    - A sample is one image+instance pair: (image, the mask of one instance of a structure, a box from that mask, its structure, the frame's accepted masks). An instance is a connected component of a label's polygons on the image, as a user boxes it in Label Studio; test pairs are likewise the components of the test tfrecord's `{label}/pre_scan_mask`s. Components under 10 pixels, and labels in no structure, are left out. `samples_per_update` counts submitted annotations, i.e. images.
    - An update is `steps_per_update` gradient steps, however many new annotations it has. Each step draws `minibatch_size` annotations at random, with replacement, from the update's new ones and the replay buffer, `replay.ratio` replayed per new one, and a pair from each: a random instance, with a random share of the others accepted. So early updates see their few annotations many times, and later ones a growing variety. Each pair is augmented afresh (its image with its mask and accepted masks), then its box made from the augmented mask, with jitter. As isonoai trains, the mask is scored less the accepted masks, which the model can't give it.
  - [ ] A replay buffer of every training sample seen so far, mixed into each training step (see below).
  - [ ] Basic augmentation of training samples, new and replayed (see below).
  - [ ] Each box's structure, and the masks already accepted on the image, as the model takes them (see below).
  - [ ] Checkpointing and resuming (see below).

## Replay buffer

Training on each new batch of 16-32 labels alone would make the model chase whatever the latest images looked like: it would forget earlier exams and structures, and each update would be as noisy as its small batch. The replay buffer keeps every sample the model has trained on, and each training step mixes some of them back in.

- **What it holds:** one record per annotation this project has trained on: the image (its prescan S3 key), and each instance's structure and polygons in the model's 256 prescan frame. Masks are rasterized from the polygons when the record is used, so records stay small. Boxes, the drawn instance and its accepted masks are made then too, so a replayed annotation gives a fresh pair each time.
- **How it is used:** each minibatch has `replay.ratio` replayed annotations per new one (e.g. 1.0: half of it replayed), drawn uniformly from the buffer. The first update, with an empty buffer, trains on new annotations only.
- **What gets added:** after a training step, the new batch's records are added to the buffer, whether or not the updated model is accepted. A rejected update means the weights got worse, not that the labels were wrong, so later steps still learn from them.
- **Persistence:** the buffer is saved with each accepted checkpoint, so a restarted backend carries on where it left off (see Checkpointing and resuming).
- **No cached embeddings:** pairs are augmented (see below), so every pair goes through the frozen image encoder. The prescan images are cached in memory instead (256 KB each): fetching a step's images from S3 took several seconds, leaving the GPU idle.

Config key: `replay.ratio`.

## Augmentation

The model learns from a small, slowly growing set, and the replay buffer shows it the same images again and again, so without augmentation the decoder overfits to the exact pixels of the images labelled so far. It only needs to be basic, applied afresh to every pair (new or replayed) in every update:

- **Box jitter:** each box coordinate shifted by up to `box_jitter` pixels, as in the original training. It costs nothing, since it's applied to the prompt only.
- **Intensity:** small random changes in brightness and contrast, for differences in gain and depth between exams.
- **Geometry:** small random scaling and shifting, and left-right flips if the anatomy allows it. The mask, and so the box, are transformed with the image.

Should use low effort keras augmentation pipeline. Test images are never augmented.

## Structures and accepted masks

isonoai's T-SAM takes two prompts besides the box, and learns an embedding for each in its prompt encoder, so online training fine-tunes them with the rest of it:

- **The box's structure:** one of `SAM_STRUCTURES`, groups of labels that look alike in ultrasound (e.g. "Fibrous": tendons, ligaments, nerves). `tsam/structures.py` is a copy of isonoai's, plus our projects' label names: add a new project's labels there, to the structure they look like. A label's tool suffix (`Bone [SAM]`) and instance suffix (`Carotid_1`) are ignored. The structures themselves and their order must stay isonoai's (the trainer checks the model's list at startup). Labels in no structure aren't trained on, and their boxes are predicted as "Anything".
- **The accepted masks:** the masks already accepted on the image, as a map of their structures. The model takes them into account, and its mask may not overlap them (its logits there are -1e4). In training they are a random share of the frame's other instances, drawn as isonoai draws them: each with the same chance, itself uniform in [0, 1], but never one the drawn instance is over half inside. When predicting they are the polygons already on the image, which the `accepted-context.js` plugin adds to each request: Label Studio sends only the smart boxes.

## Checkpointing and resuming

- **What a checkpoint holds:** the accepted T-SAM, logged as isonoai logs its models (`model/`, `mlflow.tensorflow`), so `runs:/<run id>/model` loads like any isonoai run's model; and in `checkpoint/`, the replay buffer, the step and the best test IoU.
- **Examples:** `examples/deployed/` shows the accepted T-SAM on the first 20 test images, each its labels (left) beside the masks predicted from their boxes (right), rewritten with every checkpoint; `examples/base/` shows the same for the base T-SAM, logged once when a new run starts.
- **Rollback** after a rejected update is from memory, and puts the optimizer's state back too, so the next step starts from the accepted model, not the rejected one. The optimizer's state isn't part of the logged model, so a resumed run starts it afresh.
- **Where:** logged as an MLflow artifact of the project's run each time an update is accepted, so a run ID is all that's needed to resume.
- **Resuming:** if `continue_run_id` is set in the config, the backend loads that run's model instead of `base_run_id`'s, logs to that run instead of creating one, restores the rest from its `checkpoint/`, and carries on from the step stored in it, so metrics continue the same step count. It skips logging parameters, since the run already has them. It still tests the restored model once at startup and logs that, to confirm the restore and compare against the run's last logged score.

## Workflow

1. We read the config
   - A config without `trainer_config` (e.g. `configs/frozen.yaml`) only serves the loaded model, frozen: no MLflow run, test, backfill or training.
2. We load the pretrained T-SAM
3. If `continue_run_id` is set, we resume that run: load its model and checkpoint (replay buffer, step) and continue from there. Otherwise we create a new MLflow run and log parameters; then we backfill: the project's latest `backfill_limit` annotations (null: all) are fed through `fit()`, oldest first, as if they had just been submitted.
4. We run the model on the test set: for a new run this is the original T-SAM's baseline; for a resumed run, a check that the restore worked. Its score is the best accepted score so far.
5. As labels come in through the API, they are queued; once a batch is ready, we run a training step on it plus a replay sample, then re-test.
6. If test-performance does not degrade beyond the acceptance threshold compared to the best accepted score so far (not the previous model's, or small drops could add up over many updates), we accept the new model and save a checkpoint of it. Otherwise we roll back to the latest accepted state, optimizer state included.
7. If a new model is accepted, we switch over to using that in production instead (in a threadsafe way)
