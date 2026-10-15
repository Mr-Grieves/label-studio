/**
 * effsam: colour smart clicks by polarity -- foreground clicks one colour,
 * background (alt-)clicks another.
 *
 * Label Studio already records the polarity on the region (keypoint.negative, sent
 * to the backend as is_positive) but draws every keypoint in its label's colour.
 * This repaints the keypoint's Konva circle (region.shapeRef). Label Studio
 * resets the fill when it re-renders the point (e.g. on select/hover), so it is
 * re-applied on a short timer.
 */
(() => {
  const POSITIVE = "#00c853";
  const NEGATIVE = "#ff1744";
  const POLL_MS = 200;

  // Plugins re-run per annotation; drop the previous run's timer.
  clearInterval(window.__effsamClickColoursTimer);

  window.__effsamClickColoursTimer = setInterval(() => {
    for (const region of LSI.regions ?? []) {
      if (region.type !== "keypointregion" || !region.dynamic || !region.shapeRef) continue;
      const colour = region.negative ? NEGATIVE : POSITIVE;
      if (region.shapeRef.fill() === colour) continue;
      region.shapeRef.fill(colour);
      region.shapeRef.getLayer()?.batchDraw();
    }
  }, POLL_MS);
})();
