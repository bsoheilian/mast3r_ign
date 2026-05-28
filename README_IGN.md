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

