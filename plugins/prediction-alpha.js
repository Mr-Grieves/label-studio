/**
 * effsam: a slider for the opacity of the predicted (not yet accepted) mask, to
 * check it against the image while clicking.
 *
 * Each brush region is drawn on its own Konva layer, and Label Studio sets that
 * layer's canvas opacity once, when it mounts (BrushRegion.setLayerRef). Every
 * click brings a new suggestion with a new layer, so the slider's value is
 * re-applied on a short timer.
 *
 * The backend's RLE alpha (ALPHA in backends/effsam/model.py) multiplies with
 * this; set it to 255 so the slider's 1.0 means fully opaque.
 */
(() => {
  const PANEL_ID = "effsam-prediction-alpha";
  const STORAGE_KEY = "effsam:prediction-alpha";
  const POLL_MS = 200;

  // Plugins re-run per annotation; drop the previous run's panel and timer.
  document.getElementById(PANEL_ID)?.remove();
  clearInterval(window.__effsamAlphaTimer);

  let alpha = 0.6;
  try {
    alpha = Number(localStorage.getItem(STORAGE_KEY) ?? alpha);
  } catch {}

  // TODO: place it in Label Studio's toolbar rather than floating over the page.
  const panel = document.createElement("div");
  panel.id = PANEL_ID;
  panel.style.cssText =
    "position:fixed;right:16px;bottom:72px;z-index:1000;padding:8px 12px;" +
    "background:rgba(0,0,0,.7);color:#fff;border-radius:8px;font:12px sans-serif;";
  panel.innerHTML = `
    <label>Prediction opacity
      <input type="range" min="0" max="1" step="0.05" value="${alpha}" style="vertical-align:middle">
    </label>`;
  document.body.appendChild(panel);

  panel.querySelector("input").addEventListener("input", (e) => {
    alpha = Number(e.target.value);
    try {
      localStorage.setItem(STORAGE_KEY, String(alpha));
    } catch {}
    apply();
  });

  function apply() {
    const suggestions = LSI.annotation?.suggestions;
    if (!suggestions) return;
    for (const region of suggestions.values()) {
      // TODO: check that suggestion masks get a layerRef like accepted ones do.
      const canvas = region.type === "brushregion" && region.layerRef?.canvas?._canvas;
      if (canvas && canvas.style.opacity !== String(alpha)) canvas.style.opacity = alpha;
    }
  }

  window.__effsamAlphaTimer = setInterval(apply, POLL_MS);
})();
