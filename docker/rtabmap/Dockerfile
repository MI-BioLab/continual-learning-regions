FROM osrf/ros:melodic-desktop-full

WORKDIR /root

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update 
RUN apt-get install -y nano
RUN apt-get install -y wget
RUN apt-get install -y x11-xserver-utils
RUN apt-get install -y mesa-utils
RUN apt-get install -y software-properties-common
RUN apt-get install -y unzip
RUN apt-get install -y git
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc 
RUN source ~/.bashrc
RUN mkdir -p ~/catkin_ws/src
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && cd ~/catkin_ws && catkin_make -j4"
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc
RUN echo "export ROS_HOSTNAME=localhost" >> ~/.bashrc 
RUN echo "export ROS_MASTER_URI=http://localhost:11311" >> ~/.bashrc
RUN source ~/.bashrc
RUN mkdir -p ~/SLAM/libraries
RUN mkdir -p ~/SLAM/programs

# sqlitebrowser   
RUN apt-get install -y sqlitebrowser   

# libsuitesparse
RUN apt-get install -y libsuitesparse-dev

# octomap-rviz
RUN apt-get install -y ros-melodic-octomap-rviz-plugins

# velodyne
RUN sudo apt-get install -y ros-melodic-velodyne

# imu
RUN apt-get install -y ros-melodic-imu-filter-madgwick

# costmap_2d
RUN apt-get install -y ros-melodic-costmap-2d

# fiducial_msgs
RUN apt-get install -y ros-melodic-fiducial-msgs

# move_base_msgs
RUN apt-get install -y ros-melodic-move-base-msgs

# find_object_2d
RUN apt-get install -y ros-melodic-find-object-2d

# remove opencv 3.2
RUN apt-get remove -y libopencv*

#gdb
RUN apt-get install -y libc6-dbg gdb valgrind 

# opencv-contrib
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 wget -O opencv_contrib-4.2.0.tar.gz https://github.com/opencv/opencv_contrib/archive/refs/tags/4.2.0.tar.gz && \
                 tar -xzvf opencv_contrib-4.2.0.tar.gz && \
                 rm -rf opencv_contrib-4.2.0.tar.gz"

# opencv
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 wget -O opencv-4.2.0.tar.gz https://github.com/opencv/opencv/archive/refs/tags/4.2.0.tar.gz && \
                 tar -xzvf opencv-4.2.0.tar.gz && \
                 rm -rf opencv-4.2.0.tar.gz && \
                 cd opencv-4.2.0 && \
                 mkdir build && \
                 cd build && \
                 cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=~/SLAM/libraries/opencv_contrib-4.2.0/modules .. && \
                 make -j4 && \
                 make install"

# g2o
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 wget -O 20230223_git.tar.gz https://github.com/RainerKuemmerle/g2o/archive/refs/tags/20230223_git.tar.gz && \
                 tar -xzvf 20230223_git.tar.gz && \
                 rm -rf 20230223_git.tar.gz && \
                 cd g2o-20230223_git && \
                 sed -E -i 's/cmake_minimum_required\(VERSION [0-9]+\.[0-9]+\)/cmake_minimum_required(VERSION 3.10)/' CMakeLists.txt && \
                 mkdir build && \
                 cd build && \
                 cmake -DBUILD_WITH_MARCH_NATIVE=OFF -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_USE_OPENGL=OFF .. && \
                 make -j4 && \
                 make install"

# opengv
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 git clone https://github.com/laurentkneip/opengv.git && \
                 cd opengv && \
                 sed -i '39s/^/# /' 'CMakeLists.txt' && \
                 mkdir build && \
                 cd build && \
                 cmake .. && \
                 make -j4 && \
                 make install"

# libnabo
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 wget -O libnabo-1.0.7.tar.gz https://github.com/norlab-ulaval/libnabo/archive/refs/tags/1.0.7.tar.gz && \
                 tar -xzvf libnabo-1.0.7.tar.gz && \
                 rm -rf libnabo-1.0.7.tar.gz && \
                 cd libnabo-1.0.7 && \
                 SRC_DIR=\`pwd\` && \
                 BUILD_DIR=\${SRC_DIR}/build && \
                 mkdir -p \${BUILD_DIR} && cd \${BUILD_DIR} && \
                 cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \${SRC_DIR} && \
                 make -j4 && \
                 make install"

# libpointmatcher
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 wget -O libpointmatcher-1.3.1.tar.gz https://github.com/norlab-ulaval/libpointmatcher/archive/refs/tags/1.3.1.tar.gz && \
                 tar -xzvf libpointmatcher-1.3.1.tar.gz && \
                 rm -rf libpointmatcher-1.3.1.tar.gz && \
                 cd libpointmatcher-1.3.1 && \
                 mkdir build && \
                 cd build && \
                 cmake .. && \
                 make -j4 && \
                 make install"

# gtsam
RUN /bin/bash -c "add-apt-repository ppa:borglab/gtsam-release-4.0 && apt update && apt install -y libgtsam-dev libgtsam-unstable-dev"

# nlohmann
RUN /bin/bash -c "cd ~/SLAM/libraries && \
                 git clone https://github.com/nlohmann/json.git && \
                 cd json && \
                 mkdir build && \
                 cd build && \
                 cmake .. && \
                 make install"



# RUN /bin/bash -c "cd ~/SLAM/programs/rtabmap && \
#                  mkdir build && \
#                  cd build && \
#                  cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 .. && \
#                  make -j4 && \
#                  make install"

# cv_bridge rebuilded with opencv 4
RUN /bin/bash -c "cd ~/catkin_ws/src && \
                 git clone https://github.com/fizyr-forks/vision_opencv.git && \
                 cd vision_opencv && \
                 git checkout opencv4 && \
                 cd ~/catkin_ws/ && \
                 . /opt/ros/melodic/setup.bash && \
                 catkin_make -j4"

RUN /bin/bash -c "cd ~/ && \
                 git clone https://github.com/MI-BioLab/continual-learning-regions.git && \
                 cp -r continual-learning-regions/rtabmap ~/SLAM/programs && \
                 cp -r continual-learning-regions/rtabmap_ros ~/catkin_ws/src && \
                 rm -rf ~/continual-learning-regions"

# RUN /bin/bash -c "cd ~/catkin_ws/src/rtabmap_ros && \
#                  . /opt/ros/melodic/setup.bash && \
#                  cd ~/catkin_ws && \
#                  catkin_make -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 -DRTABMAP_SYNC_MULTI_RGBD=ON -DRTABMAP_SYNC_USER_DATA=ON -j4"

RUN ldconfig

RUN apt-get -y update 
RUN apt-get install -y xterm 
RUN echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libLLVM-10.so.1" >> ~/.bashrc

