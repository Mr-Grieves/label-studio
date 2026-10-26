/**
 * medsam2: when newly propagated masks are accepted, delete the previous run's masks.
 *
 * With auto-accept on, every MedSAM2 run (each corrective click) is accepted straight away as a
 * fresh set of masks, one per frame. Without this, each frame would collect one mask per run. When
 * a new run is accepted, this deletes every other mask with the same label: the backend returns its
 * masks with the click's label on a hidden <BrushLabels>, which nobody draws with, so every mask
 * with that label is a propagated one. The whole previous run goes, not just the frames the new run
 * covers, so frames it left empty (the structure gone) don't keep an old mask.
 *
 * (Label Studio saves accepted predictions with origin "manual", so origin can't tell them apart.)
 *
 * Never touched: polygons, masks with other labels (including your own hand-drawn brush labels).
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

    const freshLabels = new Set(fresh.map((r) => r.labelName));
    const freshIds = new Set(fresh.map((r) => r.id));
    const stale = masks.filter((r) => !freshIds.has(r.id) && freshLabels.has(r.labelName)).map((r) => r.id);
    for (const r of fresh) accepted.add(r.id);

    for (const id of stale) {
      // Looked up again: deleting a region can replace the others' nodes in Label Studio's state tree.
      const region = annotation.regions.find((r) => r.id === id);
      if (region) annotation.deleteRegion(region);
    }
  }, POLL_MS);
})();
