import numpy as np
from dust3r.viz import SceneViz

rng = np.random.default_rng(0)
points = rng.normal(scale=0.8, size=(4000, 3)).astype(np.float32)
colors = np.clip((points + 2.4) * 50, 0, 255).astype(np.uint8)

camera_pose = np.eye(4, dtype=np.float32)
camera_pose[2, 3] = 5.0

viz = SceneViz()
viz.add_pointcloud(points, colors)
viz.add_camera(
    pose_c2w=camera_pose,
    focal=500,
    color=(255, 255, 255),
    cam_size=0.5,
)
viz.show(point_size=4)

# to run use 
# apt-get update && apt-get install -y libglu1-mesa
# PYTHONPATH=/mast3r_ign/dust3r${PYTHONPATH:+:$PYTHONPATH} python demo_duster_viz.py