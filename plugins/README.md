# Label Studio Enterprise plugins for effsam

Small JavaScript plugins for the labeling screen, used with the effsam backend's
interactive clicks. Each file is self-contained.

| File | What it does |
|---|---|
| `on-accept.js` | When a predicted mask is accepted: cuts out pixels other regions already cover, turns it into a polygon, and deletes the smart clicks so they don't prompt the next mask |
| `on-reject.js` | When a predicted mask is rejected with its ✗, deletes the smart clicks (Label Studio's "reject all" already does) |
| `prediction-alpha.js` | A slider for the opacity of the predicted (not yet accepted) mask |
| `click-colours.js` | Colours foreground clicks green and background (alt-)clicks red |

## Installing

Plugins are an Enterprise feature and have to be enabled for the organization first:
*Project → Settings → Labeling Interface → Request Access* (or ask the account
manager). Then paste the plugin code into the project's plugin field on that same
settings page. If a project takes a single script, paste the files one after the
other -- each is wrapped in its own function, so they don't clash.

## Labeling config

`on-accept.js` turns accepted masks into polygons. The model's masks are labelled
with the clicked keypoint's label, so the config needs:

- `<KeyPointLabels smart="true">` and `<BrushLabels>` with the same label values
  (`Effusion [SAM]`), so Label Studio can show the mask while you click. The brush
  labels sit in a hidden view: they're only there to display predictions.
- a `<PolygonLabels>` label for each of them: the same value, or the same name
  suffixed ` [Polygon]` (`Effusion [SAM]` -> `Effusion [Polygon]`).

```xml
<KeyPointLabels name="clicks" toName="image" smart="true">
  <Label value="Effusion [SAM]" background="#e5484d"/>
</KeyPointLabels>
<PolygonLabels name="polygon-labels" toName="image">
  <Label value="Effusion [Polygon]" background="#e5484d"/>
</PolygonLabels>
<View style="display:none">
  <BrushLabels name="brush-labels" toName="image">
    <Label value="Effusion [SAM]" background="#e5484d"/>
  </BrushLabels>
</View>
```

## How they work

They rely on Label Studio frontend internals, not just the documented `LSI` API
(`LSI.annotation`, `LSI.regions`), so they may need adjusting after a Label Studio
upgrade:

- `annotation.suggestions`, `annotation.deleteAllDynamicregions()`,
  `annotation.createResult()`, `annotation.names`, `region.rle`
- `region.dynamic` (made with a smart tool), `region.negative` (alt-click),
  `region.fromSuggestion` (an accepted prediction)
- `region.layerRef` (a brush's Konva layer), `region.shapeRef` (a keypoint's Konva circle)

Each one re-applies its change on a short timer, because Label Studio re-renders
regions and doesn't emit an event when a suggestion is accepted.

## Not yet verified

These are skeletons, written against Label Studio 1.23's frontend bundle and not
yet run:

- whether plugins re-run per annotation (the files assume so, and clean up after
  the previous run);
- whether suggestion masks get a `layerRef` like accepted masks do;
- whether `LSI.on("entityCreate", ...)` fires on accept, which would replace the
  timer in `on-accept.js`;
- whether `annotation.createResult()` takes the polygon's points in percent, as
  saved results store them.
