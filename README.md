# continual-learning-regions
Repository for continual learning of regions to enhance loop closure detection and relocalization in the context of robotic SLAM.

# Set up the environment
1. Download the repository
2. Create a conda environment called **clr** (stands for continual learning regions) using the yaml file in the folder with the command `conda env create -f environment.yaml`
3. Activate the environment with the command `conda activate clr`
4. Inside the avalanche folder run the command `pip install -e .` to install avalanche adapted for the experiments

# Download the data

# Run the experiments
Inside the *experiments* folder you can change the settings by manipulating the files inside the *config* folder. To run the experiments you can change the *main.py* inside the *src* folder and run it with the command `python src/main.py`

# Run RTAB-Map
To run RTAB-Map, we suggest to use [**Docker**](https://www.docker.com/). Inside the *docker* folder there is a Dockerfile you can use to build a docker image. See the *docker/README.md* for a step by step guide to build and run the docker image. In docker *rtabmap* folder is under */root/* (or *~/*), while *rtabmap_ros* is already inside *catkin_ws/src*.

If you don't want to use docker, you need to:
1. install ROS (we used Ubuntu 18 and [ROS melodic](https://wiki.ros.org/melodic/Installation/Ubuntu))
2. you can inspect the file *docker/Dockerfile* to see the packages required by RTAB-Map and install them by hand 
3. copy the *rtabmap* folder where you want (you can leave it in this folder)
4. copy the *rtabmap_ros* folder inside your *catkin_ws/src* folder

After run docker, or have installed everything in local, you have to install libtorch C++ (we used libtorch 1.13.1, with Cuda 11.6).
In our experiments, we never used GPU, so we suggest to install libtorch-cpu. 

* **ONLY if you want to use CUDA** install [Cuda 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Linux) and [cuDNN 8.9.7](https://developer.nvidia.com/rdp/cudnn-archive) for Cuda 11.x.
* **if you want libtorch-cpu** download it from [here](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.1%2Bcpu.zip).
* **if you want libtorch-gpu** download it from [here](https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.13.1%2Bcu116.zip)

After libtorch installation you need to add to your .bashrc file (in */root/.bashrc* if you used docker): <br/>
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libtorch/lib/`

If you also installed Cuda, add to your .bashrc file also:
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

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
