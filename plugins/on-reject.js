/**
 * effsam: delete the smart clicks when a predicted mask is rejected.
 *
 * Label Studio's "reject all" deletes them itself (rejectAllSuggestions ->
 * deleteAllDynamicregions), but rejecting one mask with its ✗ (rejectSuggestion) only drops the
 * mask. The clicks would then also prompt the next mask: Label Studio sends the backend every
 * smart keypoint with the clicked one's label.
 *
 * There's no event for a rejection, so the pending predictions are polled: when they go from some
 * to none and none of them became a region (which accepting does), they were rejected.
 *
 * A new click whose prediction comes back empty also clears the pending predictions, so it looks
 * the same and deletes the clicks too. The backend always returns a mask, so that shouldn't happen.
 */
(() => {
  const POLL_MS = 250;

  // Plugins re-run per annotation; drop the previous run's timer.
  clearInterval(window.__effsamOnRejectTimer);

  let annotationId = null; // the annotation `pending` belongs to
  let pending = new Set(); // ids of the predictions pending at the last poll

  window.__effsamOnRejectTimer = setInterval(() => {
    const annotation = LSI.annotation;
    if (!annotation) return;
    if (annotation.id !== annotationId) {
      annotationId = annotation.id;
      pending = new Set();
    }

    const current = new Set(annotation.suggestions.keys());
    const gone = [...pending].filter((id) => !current.has(id));
    pending = current;
    if (current.size || !gone.length) return; // still pending, or nothing was

    // Accepting keeps the prediction's id for the new region (ids may carry an "#<annotation>" suffix).
    const base = (id) => String(id).split("#")[0];
    const goneIds = new Set(gone.map(base));
    const accepted = annotation.regions.some((r) => goneIds.has(base(r.id)));
    if (!accepted) annotation.deleteAllDynamicregions();
  }, POLL_MS);
})();
