ClotML customer geometries inbox
==================================

demo_stenosis_vessel.pt is a real stenosis vessel from the training cohort (patient041),
included so the app has something to predict on the first time you open it -- a quick way
to sanity-check that the model is behaving as expected before trying your own geometry.
Delete it any time; it only needs its matching mesh at
data/raw/biochem_anchors/patient041.msh to keep working.

Put your own vessel files HERE (this folder) too, then launch:

  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_customer_predict.ps1

Supported files
---------------
  .pt   - existing ClotML graph (recommended for first try)
  .msh  - Gmsh mesh with physical line tags (see below)
  .nas  - Nastran mesh exported by our pipeline (same tags)

Required mesh tags (Gmsh physical groups)
-----------------------------------------
  Inlet      = 101
  Outlet_1   = 102
  Walls      = 103
  Fluid_Domain = 201  (optional volume/surface tag)

For .msh / .nas you also need a sidecar JSON with the same stem
(e.g. vessel_a.msh + vessel_a.json) containing centerline_pts,
centerline_tangents, d_bar, and unit="m". Meshes from our parametric
vessel generator already include this. Untagged STL is not supported yet.

In the app
----------
  - Click Open folder to open this directory in Explorer (easiest place to drop files)
  - Or click Browse -- it starts here in this Geometries folder
  - Pick Inbox and select a file, set Re (default 450) and Hours, then Run prediction
  - Drag the Timeline slider after a run completes
