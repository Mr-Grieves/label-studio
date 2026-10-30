/**
 * effsam: tidy up a predicted mask once it's accepted (with its check mark, or "accept all").
 *
 *  1. It loses any pixels already covered by other regions on the image (masks or polygons, any
 *     label): regions mustn't overlap, and the model doesn't know about the others.
 *  2. It becomes a polygon with the same label, which is easier to edit than a mask: the outer
 *     boundary of its largest piece, simplified as in isonoai's get_contours_from_mask
 *     (portal/label_studio/scripts/add_predictions.py).
 *  3. Once no other prediction is pending, the smart clicks are deleted. Label Studio sends the
 *     backend every smart keypoint with the clicked one's label, so leftover clicks would also
 *     prompt the next mask. ("Accept all" deletes them itself; the check mark doesn't.)
 *
 * Needs a <PolygonLabels> on the same image with a label for each <BrushLabels> label: its name
 * without the tool suffix ("Effusion [SAM]" -> "Effusion").
 * Without it, or if the polygon would have under MIN_POLYGON_POINTS points, the mask is left as
 * it was -- overlap included -- and the console says why.
 */
(() => {
  const POLL_MS = 250;
  // As isonoai's PostprocessConfig: simplification tolerance = EPSILON * sqrt(perimeter in pixels)
  const EPSILON = 0.08;
  const MIN_POLYGON_POINTS = 3;

  // Plugins re-run per annotation; drop the previous run's timer.
  clearInterval(window.__effsamOnAcceptTimer);

  const seen = new Set(); // accepted masks already handled

  // Accepting adds the mask with annotation.areas.set(), which may not fire
  // "entityCreate", so poll for newly accepted masks instead.
  window.__effsamOnAcceptTimer = setInterval(() => {
    const annotation = LSI.annotation;
    if (!annotation) return;

    const accepted = annotation.regions.filter(
      (r) => r.type === "brushregion" && r.fromSuggestion && !seen.has(r.id),
    );
    if (!accepted.length) return;

    for (const region of accepted) {
      seen.add(region.id);
      try {
        tidyAcceptedMask(annotation, region);
      } catch (e) {
        console.error("effsam: couldn't tidy the accepted mask", e);
      }
    }

    if (annotation.suggestions.size) return; // another mask is still pending
    annotation.deleteAllDynamicregions();
  }, POLL_MS);

  function tidyAcceptedMask(annotation, region) {
    const image = region.object;
    const label = region.labelName;
    const target = polygonLabelFor(annotation, image, label);
    if (!target) {
      console.warn(`effsam: no <PolygonLabels> on "${image.name}" has a label for "${label}"; kept the mask as is`);
      return;
    }

    const width = image.naturalWidth;
    const height = image.naturalHeight;
    const mask = rleToMask(rleOf(region), width, height);

    // 1. Pixels other regions already cover aren't this one's.
    const others = annotation.regions.filter((r) => r !== region && r.object === image);
    const occupied = occupiedPixels(others, width, height);
    for (let i = 0; i < mask.length; i++) if (occupied[i]) mask[i] = 0;

    // 2. The mask -> a polygon (in percent of the image, like every Label Studio region).
    const points = maskToPolygon(mask, width, height);
    if (!points) {
      console.warn(`effsam: the accepted "${label}" mask is too small for a polygon; kept the mask as is`);
      return;
    }
    annotation.createResult(
      { points: points.map(([x, y]) => [(100 * x) / width, (100 * y) / height]), closed: true },
      { polygonlabels: [target.label] },
      target.control,
      image,
    );
    annotation.deleteRegion(region);
  }

  /**
   * The <PolygonLabels> tag and label value for a mask's label: its name without the tool suffix
   * ("Effusion [SAM]" -> "Effusion"). Null if there's none.
   */
  function polygonLabelFor(annotation, image, label) {
    const name = label.replace(/\s*\[[^\]]*\]\s*$/, "");
    for (const tag of annotation.names.values()) {
      if (tag.type !== "polygonlabels" || (tag.toname && tag.toname !== image.name)) continue;
      if (tag.findLabel?.(name)) return { control: tag, label: name };
    }
    return null;
  }

  /** The other regions' pixels, as one 0/1 mask. Polygons and masks; other region types are ignored. */
  function occupiedPixels(regions, width, height) {
    const occupied = new Uint8Array(width * height);
    for (const region of regions) {
      if (region.type === "polygonregion") {
        const points = region.serialize()?.value?.points ?? [];
        fillPolygon(
          occupied,
          points.map(([x, y]) => [(x / 100) * width, (y / 100) * height]),
          width,
          height,
        );
      } else if (region.type === "brushregion") {
        const mask = rleToMask(rleOf(region), width, height);
        for (let i = 0; i < mask.length; i++) occupied[i] |= mask[i];
      }
    }
    return occupied;
  }

  /** A mask region's RLE. Masks drawn or edited with the brush only get one by serializing. */
  function rleOf(region) {
    return region.rle?.length ? region.rle : region.serialize()?.value?.rle;
  }

  /**
   * Label Studio's mask RLE -> a 0/1 mask, (height x width) row-major.
   *
   * A port of label_studio_sdk.converter.brush.decode_rle: the RLE packs an RGBA image (4 values
   * per pixel) bit by bit; a pixel is in the mask where its alpha isn't 0.
   */
  function rleToMask(rle, width, height) {
    let pos = 0; // in bits
    const read = (n) => {
      let value = 0;
      for (let k = 0; k < n; k++, pos++) value = value * 2 + ((rle[pos >> 3] >> (7 - (pos & 7))) & 1);
      return value;
    };

    const num = read(32);
    const wordSize = read(5) + 1;
    const rleSizes = [read(4) + 1, read(4) + 1, read(4) + 1, read(4) + 1];
    if (num !== width * height * 4) {
      throw new Error(`the mask has ${num / 4} pixels, but the image is ${width}x${height}`);
    }

    const mask = new Uint8Array(width * height);
    let i = 0;
    while (i < num) {
      const isRun = read(1);
      const end = i + 1 + read(rleSizes[read(2)]);
      if (isRun) {
        const value = read(wordSize);
        for (; i < end; i++) if ((i & 3) === 3 && value) mask[i >> 2] = 1;
      } else {
        for (; i < end; i++) if (read(wordSize) && (i & 3) === 3) mask[i >> 2] = 1;
      }
    }
    return mask;
  }

  /** Set the pixels whose centres are inside the polygon (pixel coordinates; even-odd rule). */
  function fillPolygon(out, points, width, height) {
    for (let y = 0; y < height; y++) {
      const cy = y + 0.5;
      const crossings = [];
      for (let a = 0, b = points.length - 1; a < points.length; b = a++) {
        const [x1, y1] = points[a];
        const [x2, y2] = points[b];
        if (y1 > cy !== y2 > cy) crossings.push(x1 + ((cy - y1) * (x2 - x1)) / (y2 - y1));
      }
      crossings.sort((p, q) => p - q);
      for (let k = 0; k + 1 < crossings.length; k += 2) {
        const from = Math.max(0, Math.ceil(crossings[k] - 0.5));
        const to = Math.min(width - 1, Math.floor(crossings[k + 1] - 0.5));
        for (let x = from; x <= to; x++) out[y * width + x] = 1;
      }
    }
  }

  /**
   * The outer boundary of the mask's largest piece, simplified, as [x, y] pixel centres; null if
   * it has fewer than MIN_POLYGON_POINTS points. Like isonoai's get_contours_from_mask: OpenCV's
   * external contour, then skimage's approximate_polygon with tolerance EPSILON * sqrt(perimeter).
   */
  function maskToPolygon(mask, width, height) {
    const piece = largestComponent(mask, width, height);
    if (!piece) return null;

    const contour = traceBoundary(piece, width, height);
    if (contour.length < MIN_POLYGON_POINTS) return null;

    const simplified = simplifyClosed(contour, EPSILON * Math.sqrt(perimeter(contour)));
    if (simplified.length < MIN_POLYGON_POINTS) return null;
    return simplified.map(([x, y]) => [x + 0.5, y + 0.5]);
  }

  /** The largest 8-connected piece of a 0/1 mask, as its own 0/1 mask; null if the mask is empty. */
  function largestComponent(mask, width, height) {
    const labels = new Int32Array(mask.length);
    const stack = new Int32Array(mask.length); // each pixel is pushed at most once
    let best = 0;
    let bestSize = 0;
    let next = 0;
    for (let seed = 0; seed < mask.length; seed++) {
      if (!mask[seed] || labels[seed]) continue;
      next++;
      let top = 0;
      let size = 0;
      stack[top++] = seed;
      labels[seed] = next;
      while (top) {
        const p = stack[--top];
        size++;
        const px = p % width;
        const py = (p - px) / width;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const qx = px + dx;
            const qy = py + dy;
            if (qx < 0 || qy < 0 || qx >= width || qy >= height) continue;
            const q = qy * width + qx;
            if (mask[q] && !labels[q]) {
              labels[q] = next;
              stack[top++] = q;
            }
          }
        }
      }
      if (size > bestSize) {
        best = next;
        bestSize = size;
      }
    }
    if (!best) return null;
    return Uint8Array.from(labels, (l) => (l === best ? 1 : 0));
  }

  // The 8 neighbours, clockwise on screen (y points down), starting west.
  const DIRS = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];

  /** Moore-neighbour tracing of a single 8-connected piece's outer boundary, as [x, y] pixels. */
  function traceBoundary(piece, width, height) {
    const inside = (x, y) => x >= 0 && y >= 0 && x < width && y < height && piece[y * width + x] === 1;
    const s = piece.indexOf(1);
    const start = [s % width, Math.floor(s / width)];
    const contour = [start];
    let [x, y] = start;
    let back = 0; // the first pixel in scan order has background to its west

    for (let steps = 0; steps < 4 * piece.length; steps++) {
      let found = -1;
      for (let k = 1; k <= 8; k++) {
        const d = (back + k) % 8;
        if (inside(x + DIRS[d][0], y + DIRS[d][1])) {
          found = d;
          break;
        }
      }
      if (found < 0) break; // a single pixel

      const nx = x + DIRS[found][0];
      const ny = y + DIRS[found][1];
      // Back at the start, about to retrace the first step: the boundary is closed.
      if (contour.length > 1 && x === start[0] && y === start[1] && nx === contour[1][0] && ny === contour[1][1]) {
        contour.pop(); // the start, pushed again on arrival
        break;
      }
      // The next search starts from the last background neighbour checked, seen from the new pixel.
      const [dx, dy] = DIRS[(found + 7) % 8];
      const bx = x + dx;
      const by = y + dy;
      x = nx;
      y = ny;
      back = DIRS.findIndex(([ex, ey]) => ex === bx - x && ey === by - y);
      contour.push([x, y]);
    }
    return contour;
  }

  function perimeter(points) {
    let length = 0;
    for (let a = 0, b = points.length - 1; a < points.length; b = a++) {
      length += Math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]);
    }
    return length;
  }

  /** Douglas-Peucker on a closed contour, like skimage.measure.approximate_polygon. */
  function simplifyClosed(points, tolerance) {
    const closed = [...points, points[0]];
    const keep = new Uint8Array(closed.length);
    keep[0] = keep[closed.length - 1] = 1;
    const stack = [[0, closed.length - 1]];
    while (stack.length) {
      const [a, b] = stack.pop();
      let farthest = -1;
      let maxDistance = tolerance;
      for (let i = a + 1; i < b; i++) {
        const d = distanceToSegment(closed[i], closed[a], closed[b]);
        if (d > maxDistance) {
          farthest = i;
          maxDistance = d;
        }
      }
      if (farthest >= 0) {
        keep[farthest] = 1;
        stack.push([a, farthest], [farthest, b]);
      }
    }
    return closed.filter((_, i) => keep[i]).slice(0, -1);
  }

  function distanceToSegment([px, py], [ax, ay], [bx, by]) {
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSquared)) : 0;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }
})();
