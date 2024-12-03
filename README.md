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