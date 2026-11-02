(() => {
  // Label Studio sends the backend only the smart boxes with each prediction request (its "context"). T-SAM also takes
  // the masks already accepted on the image, so this adds the annotation's polygons to the context, as "accepted":
  // results as Label Studio saves them.

  // Plugins re-run per annotation; wrap fetch once.
  if (window.__tsamFetch) return;
  window.__tsamFetch = window.fetch;

  window.fetch = (input, init) => {
    const url = String(input?.url ?? input);
    if (url.includes("/interactive-annotating") && typeof init?.body === "string") {
      const body = JSON.parse(init.body);
      const accepted = (LSI.annotation?.serializeAnnotation() ?? []).filter((r) => r.type === "polygonlabels");
      init = { ...init, body: JSON.stringify({ ...body, context: { ...body.context, accepted } }) };
    }
    return window.__tsamFetch(input, init);
  };
})();
