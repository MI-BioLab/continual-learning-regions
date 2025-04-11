# continual-learning-regions
Repository for continual learning of regions to enhance loop closure detection and relocalization in the context of robotic SLAM.

Click below to watch the video that shows how the clustering algorithm works on USyd dataset
[![frame](https://github.com/user-attachments/assets/704a762e-b76d-4225-9f3e-dae1553e19ed)](https://drive.google.com/file/d/14X8AfpSdNe13rGBi5ihB3xy8rvMf4Fa1/view)

Click below to watch the video that shows how the system works on sequence 00 of the KITTI odometry dataset
[![frame](https://github.com/user-attachments/assets/d59a7720-c69b-4d6e-9983-52976b8bc586)](https://drive.google.com/file/d/1QtxVlosS_fk-6h8WiWTgpbqnUUIE4jA6/view)

Click below to watch the video that shows how the system works on the 12 sequences of USyd dataset

[![frame](https://github.com/user-attachments/assets/1cf97370-5f0c-4dca-9a22-1a9bc816210c)](https://drive.google.com/file/d/1ZsCGBZPAi8MNjn4CuCMiUOuRGOjWu5kN/view)


# Set up the environment
1. Download the repository
2. Create a conda environment called **clr** (stands for continual learning regions) using the yaml file in the folder with the command `conda env create -f environment.yaml`
3. Activate the environment with the command `conda activate clr`
4. Inside the avalanche folder run the command `pip install -e .` to install avalanche adapted for the experiments

# Download the data
## USyd
You can download USyd from [here](https://ieee-dataport.org/open-access/usyd-campus-dataset) and follow the instruction from [here](https://gitlab.acfr.usyd.edu.au/its/dataset_metapackage) to use it. We suggest to use docker, you can find an image inside the *docker* folder in this repository.  

## KITTI
You can download KITTI from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). To run KITTI with rtabmap_ros, you can create the rosbags using this [package](https://github.com/tomas789/kitti2bag).

## OpenLoris-Scene
You can download OpenLoris-Scene from this [page](https://lifelong-robotic-vision.github.io/dataset/scene.html). To run the experiments you need both the packages and the rosbags.

## St.Lucia Multiple Times of Day
You can download the ten sequences from [here](https://github.com/arrenglover/openfabmap/wiki/Datasets).

# Run the experiments
Inside the *experiments* folder you can change the settings by manipulating the files inside the *config* folder. To run the experiments you can change the *main.py* inside the *src* folder and run it with the command `python src/main.py`

# Places365
Inside the folder places365 you can find the weights of Resnet18 and Resnet50 models pretrained on Places365. In our experiments, only Resnet18 weights were used (*places365/resnet18_places365/resnet18_places365.pt*).
These models were ported into Pytorch from Cafè, if something does not work correctly, please contact us.

# Setup RTAB-Map
To run RTAB-Map, we used [**Docker**](https://www.docker.com/). Inside the *docker* folder there is a *rtabmap* folder which contains a Dockerfile you can use to build a docker image. See the *docker/README.md* for a step by step guide to build and run the docker image. In docker *rtabmap* folder is under */root/SLAM/programs* (or *~/SLAM/programs*), while *rtabmap_ros* is already inside *~/catkin_ws/src*.

After run the container, you have to install libtorch C++ (we used libtorch 1.13.1, with CUDA 11.6).
In our experiments, we never used GPU, so if you only want to reproduce the experiments, we suggest you to install libtorch-cpu.

You need to check if your GPU and nvidia drivers are compatible with CUDA 11.6.
* You can install [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux) and [cuDNN 8.9.7](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 11.x.
* you can download libtorch-cpu [here](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.1%2Bcpu.zip).
* you can download libtorch with CUDA 11.6 support from [here](https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.13.1%2Bcu116.zip)

Install everything inside the docker container.

After libtorch installation you need to add to your .bashrc file (in */root/.bashrc* if you used docker): <br/>
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libtorch/lib/`

If you also installed CUDA, add to your .bashrc file also:
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

In your bash run the command <br/>
```source /path/to/.bashrc```

Now in you bash run the command <br/>
```ldconfig```

Now, you can install rtabmap using the following commands:
```
cd path/to/rtabmap
mkdir build
cd build
cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 .. 
make -j4 
make install
```

Then you need to build catkin_ws with rtabmap_ros using the following commands:
```  
cd ~/catkin_ws 
catkin_make -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 -DRTABMAP_SYNC_MULTI_RGBD=ON -DRTABMAP_SYNC_USER_DATA=ON -j4
``` 
If you have problems in building catkin_ws, try this:

```  
cd ~/catkin_ws/src/rtabmap_ros 
. /opt/ros/melodic/setup.bash
cd ~/catkin_ws 
catkin_make -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 -DRTABMAP_SYNC_MULTI_RGBD=ON -DRTABMAP_SYNC_USER_DATA=ON -j4
```

# Run RTAB-Map
To run RTAB-Map using ros, move into the *catkin_ws* folder and run the command 
```
roslaunch rtabmap_examples *launcher_name.launch*
```

The launcher files are in the folder *rtabmap_ros/rtabmap_examples/launch/* <br/>
**Attention**: in case of USyd, you need to use *usyd_mono.launch* (that uses only the frontal camera) **NOT** *usyd_dataset.launch* (that uses the three cameras). 

