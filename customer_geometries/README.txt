ClotML customer geometries
==========================

demo_stenosis_vessel.pt is a real stenosis vessel from the training cohort (comsol041),
included as a known-good example to sanity-check the model before trying your own geometry.
Delete it any time; it only needs its matching mesh at
data/raw/biochem_anchors/comsol041.msh to keep working.

Using your own vessel
---------------------
In the app, set Geometry > Source to "Upload geometry" and choose your file.

  .msh / .nas  - a 2D triangle mesh of the vessel (linear or quadratic elements).
                 No tags or extra files needed.
  .pt          - an existing ClotML graph.

When you choose a mesh, the app previews it and finds the walls, the inlet and the outlet
from the mesh outline:
  - The inlet is shown in green and the outlet in purple. If they are the wrong way round,
    click "Swap ends".
  - If the app can't tell which ends are open (for example an outline with extra sharp
    corners), it asks you to click the inlet on the preview, then the outlet.
  - Units (m, cm or mm) are detected from the vessel's size; override them with
    "Mesh units" if the detected length looks wrong.

Meshes from our own parametric vessel generator (tagged, with a same-stem .json) still load
exactly as before.
