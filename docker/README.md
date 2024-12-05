# Docker images
This directory contains two folders:
1. rtabmap, with a Dockerfile to build the docker image which contains rtabmap and rtabmap_ros
2. usyd, with a Dockerfile to build the docker image to run the USyd dataset

You can build a docker image by running the following command inside the folder that contains a Dockerfile:
```
docker build --progress=plain -t *image_name* . &> build.log
```
By replacing *image_name* with the name you want, it creates a build.log inside the folder that shows the progress of the image building.
At the end, the image will be available in docker.

Now you have to run the container.
Before run the container, run this command in your bash
```
xhost +local:docker
```
to provide access to the windows manager from within the container

To create and run the container (only the first time), we used this command

```
docker run -it --privileged \
--net=host \
--gpus all \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-v /mnt/e/USyd:/data \
-v /home/matteo:/wsl \
--name *container_name* \
*image_name*
```
where `-v /mnt/e/USyd:/data` and `-v /home/matteo:/wsl` specifies folders you want to mount inside docker. So `/mnt/e/USyd` is mapped as `/data` inside docker and `/home/matteo` as `/wsl`. You need to change them in order to read data from your folders outside docker (which contain the data).
For the others, please refer to docker documentation.

The command will create a docker container named \*container_name\*.

After this, you can run a docker bash using the command 
```
docker exec -it *container_name* bash
```

For USyd, you can do the same exact thing, but you need to add `--ipc host` at the container builder command:

```
docker run -it \
  --privileged \
  --net=host \
  -e DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  --ipc host  \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v /mnt/e/USyd:/data \
  --name usyd \
  usyd
```
