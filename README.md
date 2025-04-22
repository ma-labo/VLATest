# VLATest: Testing and Evaluating Vision-Language-Action Models for Robotic Manipulation

This repository includes the replication package for ``VLATest: Testing and Evaluating Vision-Language-Action Models for Robotic Manipulation``

The codebase is modified based on [SimplerEnv](https://github.com/simpler-env/SimplerEnv) and [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim/tree/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37)

## Data Availability

Our generated testing scenes is provided under ``data/`` in json files. To reproduce our experiment results, one can proceed to the following installation and replication guides. 


## Installation

Prerequisites:
- CUDA version >=12.
- An NVIDIA GPU.
- Python >= 3.10

Clone this repo:
```
git clone https://github.com/ma-labo/VLATest.git
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd VLATest/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd VLATest
pip install -e .
```

Install development support

```
sudo apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
sudo apt-get install libglvnd-dev
```

```
sudo apt install ffmpeg
```

```
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

### RT-1 Inference Setup

Download RT-1 Checkpoint:
```
# First, install gsutil following https://cloud.google.com/storage/docs/gsutil_install

# Make a checkpoint dir:
mkdir VLATest/checkpoints

# RT-1-X
cd VLATest
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
mv rt_1_x_tf_trained_for_002272480_step checkpoints
rm rt_1_x_tf_trained_for_002272480_step.zip

# RT-1-400k
cd VLATest
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 .
mv rt_1_tf_trained_for_000400120 checkpoints

# RT-1-58k
cd VLATest
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000058240 .
mv rt_1_tf_trained_for_000058240 checkpoints

# RT-1-1k
cd VLATest
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000001120 .
mv rt_1_tf_trained_for_000001120 checkpoints      
```

### Octo Inference Setup

Install Octo:
```
pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # or jax[cuda12_pip] if you have CUDA 12

cd VLATest
git clone https://github.com/octo-models/octo/
cd octo
git checkout 653c54acde686fde619855f2eac0dd6edad7116b  # we use octo-1.0
pip install -e .
# You don't need to run "pip install -r requirements.txt" inside the octo repo; the package dependencies are already handled in the simpler_env repo
# Octo checkpoints are managed by huggingface, so you don't need to download them manually.
```

## Replication Package

To reproduce experiment results with our generated testing scenes (``data/``):

### RQ1
```
cd experiments
./run_exp_base_performance.sh
```

### RQ2
```
cd experiments
./run_exp_100_grasp.sh
./run_exp_100_move_near.sh
./run_exp_100_put_on.sh
./run_exp_100_put_in.sh
```

### RQ3
```
cd experiments
./run_random_lighting.sh
```

### RQ4
```
cd experiments
./run_random_camera.sh
```

### RQ5
```
cd experiments
./run_exp_base_performance_ycb.sh
```

The experiment results will be generated within ``results/``

## Data Generation

To generate new testing scenes:

### RQ1
```
cd experiments
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 1000 --ro
PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 1000 --ro
PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 1000 --ro
PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 1000 --ro
```

### RQ2
```
cd experiments
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 100 --obstacles 0
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 100 --obstacles 1
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 100 --obstacles 2
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 100 --obstacles 3
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 100 --obstacles 4

PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 100 --obstacles 0
PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 100 --obstacles 1
PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 100 --obstacles 2
PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 100 --obstacles 3
PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 100 --obstacles 4

PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 100 --obstacles 0
PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 100 --obstacles 1
PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 100 --obstacles 2
PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 100 --obstacles 3
PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 100 --obstacles 4

PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 100 --obstacles 0
PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 100 --obstacles 1
PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 100 --obstacles 2
PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 100 --obstacles 3
PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 100 --obstacles 4
```

### RQ3 & RQ4

These RQs reuses the data from RQ1. You need to extract the indexes of successful tasks in RQ1.

### RQ5
```
cd experiments
PYTHONPATH=~/VLATest python3 test_generation.py -t grasp -n 1000 --ro --ycb
PYTHONPATH=~/VLATest python3 test_generation.py -t move -n 1000 --ro --ycb
PYTHONPATH=~/VLATest python3 test_generation.py -t put-on -n 1000 --ro --ycb
PYTHONPATH=~/VLATest python3 test_generation.py -t put-in -n 1000 --ro --ycb
```

### RQ6
This RQ reuses the data from RQ1.

## Citation

If you found our paper/code useful in your research, please consider citing:

```
@article{wang2025vlatest,
 author = {Wang, Zhijie and Zhou, Zhehua and Song, Jiayang and Huang, Yuheng and Shu, Zhan and Ma, Lei},
 title = {VLATest: Testing and Evaluating Vision-Language-Action Models for Robotic Manipulation},
 journal = {Proceedings of ACM Software Engineering},
 year = {2025},
 volume = {2},
 issue = {FSE},
 articleno = {FSE073},
 month = jul
} 
```

## Acknowledgement

- [SimplerEnv](https://github.com/simpler-env/SimplerEnv)
- 
- [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim/tree/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37)