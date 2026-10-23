/**
 * medsam2: when newly propagated masks are accepted, delete the older propagated masks they replace.
 *
 * With auto-accept on, every MedSAM2 run (each corrective click) is accepted straight away as a
 * fresh set of masks, one per frame. Without this, each frame would collect one mask per run. When
 * new masks are accepted, this deletes, on the same frames and with the same label, the propagated
 * masks from earlier runs: ones accepted earlier in this session, and ones loaded from a saved
 * annotation that came from a prediction and were never edited.
 *
 * Never touched: hand-drawn regions, polygons, edited predictions, other labels, other frames.
 *
 * Label Studio emits no event when a prediction is accepted, so the masks are polled.
 */
(() => {
  const POLL_MS = 250;

  // Plugins re-run per annotation; drop the previous run's timer.
  clearInterval(window.__medsam2ReplacePropagatedTimer);

  let annotationId = null; // the annotation `accepted` belongs to
  const accepted = new Set(); // ids of the masks seen accepted so far

  window.__medsam2ReplacePropagatedTimer = setInterval(() => {
    const annotation = LSI.annotation;
    if (!annotation) return;
    if (annotation.id !== annotationId) {
      annotationId = annotation.id;
      accepted.clear();
    }

    const masks = annotation.regions.filter((r) => r.type === "brushregion");
    const fresh = masks.filter((r) => r.fromSuggestion && !accepted.has(r.id));
    if (!fresh.length) return;

    // A frame and label, so a mask only replaces its own label's masks on its own frame.
    const slot = (r) => `${r.labelName}\u0000${r.item_index ?? 0}`;
    const freshSlots = new Set(fresh.map(slot));
    const freshIds = new Set(fresh.map((r) => r.id));
    const stale = masks
      .filter(
        (r) =>
          !freshIds.has(r.id) &&
          freshSlots.has(slot(r)) &&
          r.origin !== "prediction-changed" && // edited: yours now
          (accepted.has(r.id) || r.origin === "prediction"),
      )
      .map((r) => r.id);
    for (const r of fresh) accepted.add(r.id);

    for (const id of stale) {
      // Looked up again: deleting a region can replace the others' nodes in Label Studio's state tree.
      const region = annotation.regions.find((r) => r.id === id);
      if (region) annotation.deleteRegion(region);
    }
  }, POLL_MS);
})();
