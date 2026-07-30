# Region Review (Phase 5)

Open the project workspace and choose **Regions** in the persistent navigator,
or use **Workspace → Show Region Review**. The image receives a yellow
`review_region_shapes` layer and a translucent `review_region_coverage` layer.
The coverage layer counts only saved reviewer polygons; it is not a QC score or
biological measurement.

1. Draw a polygon in the Region shapes layer and select it.
2. In Region Review, select the domain and an explicit action.
3. Optionally add a reason and notes, then choose **Apply action to selected
   shape**.

The action is autosaved to both `review/review_state.json` and
`review/review_regions.geojson`, and an audit event is appended to
`review/review_events.csv`. Use the saved-region list to remove a mistaken
annotation; removal is also recorded and rewrites the GeoJSON atomically.

The available actions record domain exclusion, object-queue, detailed-review,
ignore, rerun recommendation, and unresolved intent. They do not alter masks,
predictions, or finalized analysis in this phase; finalization applies their
meaning later. Region geometry is stored at full image coordinates even when
`--display-downsample` is active.
