# Matching street-level images with ortho or aerial images
## Build docker
```
docker compose -f docker-compose-cuda.yml run --build --rm --remove-orphans mast3r-demo
```

```
#on the host 
xhost +local:docker
docker compose -f docker-compose-cuda.yml run --build --rm   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   mast3r-demo


# add this to the code 
import matplotlib
matplotlib.use("TkAgg")

```

Most likely it is a memory/shape scaling issue: at load_image(..., size=512) the model creates much larger tensors, so GPU RAM use grows quickly (often near-quadratic with resolution). At 128, it stays under limits, so the failure disappears.

Run it once and capture the real traceback:


docker compose -f docker-compose-cuda.yml run --build --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix mast3r-demo 2>&1 | tee /tmp/mast3r_run.log
Then check for the exact failure:


grep -Ei 'out of memory|cuda|cudnn|cublas|runtimeerror|shape|size mismatch' /tmp/mast3r_run.log
And monitor GPU memory while running:


watch -n 0.5 nvidia-smi
None



```
python demo_dust3r_ga.py --weights ./docker/files/checkpoints/checkpoint-aerial-mast3r.pth image_size 518 --device cuda
```

The very first test to find matches between ground level and aerial images are not very convincing.
The problem does not seems to be very hard to resolve when roadmarking ar visible in both images. 
some ideas to explore: 
- inverse perspective rectification on ground level images before injecting into Mast3r 
- at the same tome to remove the distortion -> ori operation to implement
- check if 3D reconstrction from Mast3r is accessible , it yes it can be used to generate orthos and then match them to aerial orthos (probably without AI)
- test AI based marking extractions and matching 










&
123
ROSE
PAPA
MAMAN
ARIO