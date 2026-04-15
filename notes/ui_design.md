## UI elements / config

input / metadata overview
- set input path
- (load metadata button)
- panel (for results)
- (show layout)

[*split up] layout preview:
- checkbox show shape data
- checkbox preview image data

feature explorer
- params:
	- scale
	- sigma
	- normalisation
	- transform type
	- method
	- max keypoints
	- inlier threshold
	- max trials
	- ransac iterations
- update button
- output:
	- colourbar
	- metrics: quality, #inliers, etc
- select registration operation
- run pair registration button

pair registration
X show overview button, or overview/pair view selection
X checkbox preview image data
- pair selection (combobox)
- (store button)
- output:
	- colourbar
	- metrics
- set transform output path
- run global registration button

fused preview
- select fuse method
- show preview button
- select export file type
- select ome-zarr version
- select output tile size
- compression
- #pyramid sizes
- select thumbnail pixel size to export thumbnail (leave blank for full size export)
- set output path
- export fused button
