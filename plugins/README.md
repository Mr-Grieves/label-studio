# Label Studio Enterprise plugins

JavaScript plugins for the labeling screen. Each file is self-contained and wrapped in its own
function: paste it into the project's plugin field (*Project → Settings → Labeling Interface*;
plugins have to be enabled for the organization first). Several can be pasted one after another.

| File | For | What it does |
|---|---|---|
| `replace-propagated.js` | the medsam2 backend, with auto-accept on | When newly propagated masks are accepted, deletes the older propagated masks of the same label on the same frames, so each frame keeps only the newest |

They use Label Studio frontend internals beyond the documented `LSI` API
(`annotation.regions`, `annotation.deleteRegion()`, `region.fromSuggestion`, `region.origin`,
`region.item_index`), so a Label Studio upgrade may need them adjusted. They poll on a timer
(Label Studio emits no event when a prediction is accepted), and run nothing at load time but
starting that timer: console calls there have stopped plugins from loading in our Enterprise
project.

## Cine projects

The `on-accept.js` plugin (mask -> polygon, overlap removal; on the tsam branch) isn't frame-aware
yet: on a frame list it would treat every frame's regions as overlapping. Don't use it on cine
projects.
