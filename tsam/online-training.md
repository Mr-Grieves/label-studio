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
    - The test set is a tfrecord of prescan images and labels only. Boxes are the tight box around each mask in the model's 256 prescan frame, built as training builds them but without jitter. IoU is on the model's raw output (thresholded logits), not the backend's clean-up: comparisons are all relative.
    - The image encoder is frozen and test images aren't augmented, so their embeddings are computed once at startup and kept. Each re-test is then only decoder passes.
  - [ ] Code to setup the model for training. Don't need any callbacks. Just an optimizer and loss function definition. Losses: dice on the mask, and mse on the predicted IoU (weight 0.1), which keeps the score sent back with each prediction in step with the masks.
  - [ ] MLFlow logging. The instance should only create one new mlflow run per project (unless we are running multiple tests). The idea being that you would be able to pull up the run for "tsam_foot_ankle_online" and see how the finetuned model is progressing over the weeks of labelling time. The main metrics we will be logging are the `test_iou`'s of the model over time, but also training loss and other common metrics/parameters.
  - [ ] A basic training step that gets called every time the next batch of labels is ready. This should run every label in the samples collected through the model. Number of passes defined in config.
    - Labels arrive only through the API: Label Studio's webhook on annotation submit/update calls the backend's `fit()`, which queues the new samples and returns straight away. A batch is ready once `samples_per_update` new annotations are queued; training runs in a background thread and takes everything queued.
    - As soon as a label arrives it is converted to the prescan frame (its polygon, and later its box), and everything from then on works on prescan data only.
    - A sample is one image+label pair: (image, one label's mask, a box from that mask, its structure), the label's mask being all its polygons on that image combined, exactly as isonoai's `SAMDataset` trains T-SAM and as the test tfrecord stores labels (`{label}/pre_scan_mask`). `samples_per_update: 32` counts submitted annotations, i.e. images; every mapped label on them becomes a pair, so 32 annotations give ~190 pairs on Foot-Ankle.
    - An update is a single pass: the new pairs plus `replay.ratio` times as many drawn from the replay buffer, shuffled together and run through once, `minibatch_size` pairs per gradient step. No pair is seen twice in an update, and each is augmented once (its image with its mask), then its box made from the augmented mask, with jitter.
  - [ ] A replay buffer of every training sample seen so far, mixed into each training step (see below).
  - [ ] Basic augmentation of training samples, new and replayed (see below).
  - [ ] A learned offset per structure, so the model knows what kind of anatomy each box holds (see below).
  - [ ] Checkpointing and resuming (see below).

## Replay buffer

Training on each new batch of 16-32 labels alone would make the model chase whatever the latest images looked like: it would forget earlier exams and structures, and each update would be as noisy as its small batch. The replay buffer keeps every sample the model has trained on, and each training step mixes some of them back in.

- **What it holds:** one record per pair this project has trained on: the image (its prescan S3 key), the structure, and the label's polygons in the model's 256 prescan frame. Masks are rasterized from the polygons when the record is used, so records stay small. Boxes are made from the mask then too, with the same jitter as training, so a replayed sample gets a fresh box each time.
- **How it is used:** each update draws `replay.ratio` times as many pairs as it has new ones (e.g. 1.0: as many replayed pairs as new ones), uniformly from the buffer, and shuffles them in with the new pairs.
- **What gets added:** after a training step, the new batch's records are added to the buffer, whether or not the updated model is accepted. A rejected update means the weights got worse, not that the labels were wrong, so later steps still learn from them.
- **Persistence:** the buffer is saved with each accepted checkpoint, so a restarted backend carries on where it left off (see Checkpointing and resuming).
- **No cached embeddings:** pairs are augmented (see below), so every pair goes through the frozen image encoder (around 0.1 s per image on CPU). An update of ~320 new + ~320 replayed pairs is then about a minute on CPU, well within the time it takes to label the next 32 images.

Config key: `replay.ratio`.

## Augmentation

The model learns from a small, slowly growing set, and the replay buffer shows it the same images again and again, so without augmentation the decoder overfits to the exact pixels of the images labelled so far. It only needs to be basic, applied afresh to every pair (new or replayed) in every update:

- **Box jitter:** each box coordinate shifted by up to `box_jitter` pixels, as in the original training. It costs nothing, since it's applied to the prompt only.
- **Intensity:** small random changes in brightness and contrast, for differences in gain and depth between exams.
- **Geometry:** small random scaling and shifting, and left-right flips if the anatomy allows it. The mask, and so the box, are transformed with the image.

Should use low effort keras augmentation pipeline. Test images are never augmented.

## Structure conditioning

Each project has a short, fixed list of structures (say 6), each with its own typical shape, but SAM itself takes no notion of what a box contains: the box's label is thrown away. The model is told which structure it is looking for through a learned offset per structure:

- **A table of vectors, one row per structure** (e.g. 6 x 192), created as zeros. The project's structure list comes from its labelling config, mapping labels by name without their tool suffix (`Bone [SAM]`, `Bone [Polygon]` -> `Bone`).
- **Where it applies:** between the prompt encoder and the mask decoder, the box's row is added to the two box-corner embeddings. The saved T-SAM model is untouched: this is a tensor add in the backend's own code.
- **Starting at zero:** before any training it is exactly the original T-SAM. Each row then learns from its own structure's samples, so the decoder can learn, for example, to go for the bright cortical line in a bone box and the fibrillar band in a tendon box.
- **Every sample carries its structure:** new and replayed samples alike, so replay trains each sample's own row. Test boxes use their structure too, and per-structure test IoU shows which structures benefit.
- **Labels not in the mapping** get no offset: plain T-SAM when predicting, and they aren't trained on.
- **Saved alongside the model:** the logged T-SAM can't hold the table, so every checkpoint stores it as `checkpoint/offsets.json`, by structure name.

## Checkpointing and resuming

- **What a checkpoint holds:** the accepted T-SAM, logged as isonoai logs its models (`model/`, `mlflow.tensorflow`), so `runs:/<run id>/model` loads like any isonoai run's model; and in `checkpoint/`, the structure offsets, the replay buffer, the step and the best test IoU.
- **Rollback** after a rejected update is from memory, and puts the optimizer's state back too, so the next step starts from the accepted model, not the rejected one. The optimizer's state isn't part of the logged model, so a resumed run starts it afresh.
- **Where:** logged as an MLflow artifact of the project's run each time an update is accepted, so a run ID is all that's needed to resume.
- **Resuming:** if `continue_run_id` is set in the config, the backend loads that run's model instead of `base_run_id`'s, logs to that run instead of creating one, restores the rest from its `checkpoint/`, and carries on from the step stored in it, so metrics continue the same step count. It skips logging parameters, since the run already has them. It still tests the restored model once at startup and logs that, to confirm the restore and compare against the run's last logged score.

## Workflow

1. We read the config
2. We load the pretrained T-SAM
3. If `continue_run_id` is set, we resume that run: load its model and checkpoint (structure offsets, replay buffer, step) and continue from there. Otherwise we create a new MLflow run, log parameters, and start with the structure offsets at zero.
4. We run the model on the test set: for a new run this is the original T-SAM's baseline; for a resumed run, a check that the restore worked. Its score is the best accepted score so far.
5. As labels come in through the API, they are queued; once a batch is ready, we run a training step on it plus a replay sample, then re-test.
6. If test-performance does not degrade beyond the acceptance threshold compared to the best accepted score so far (not the previous model's, or small drops could add up over many updates), we accept the new model and save a checkpoint of it. Otherwise we roll back to the latest accepted state, optimizer state included.
7. If a new model is accepted, we switch over to using that in production instead (in a threadsafe way)
