# pick_data_gen
```bash
git clone https://github.com/memmelma/pick_data_gen.git
cd pick_data_gen
pip install -e .
```

## install modules (eval, logging, data gen)
```bash
pip mujoco shapely wandb meshcat trimesh
```

## (optional) install curobo
```bash
sudo apt install nvidia-cuda-toolkit
mamba create -n mujoco_mp python=3.10
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

mamba install -c conda-forge 'gcc[version=">=6,<11"]' 'gxx[version=">=6,<11"]'
export CUDA_HOME=$CONDA_PREFIX
mamba install nvidia/label/cuda-11.8.0::cuda
<!-- mamba install cudatoolkit=11.8 -c nvidia -->
mamba install nvidia/label/cuda-11.8.0::cuda-nvcc
mamba install nvidia/label/cuda-11.8.0::cuda-cudart

export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
```

## install diffuser actor (policy)
```bash
git clone https://github.com/memmelma/3d_diffuser_actor.git
cd 3d_diffuser_actor
conda create -f environment.yaml

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/dgl-1.1.3%2Bcu118-cp38-cp38-manylinux1_x86_64.whl
pip install diffusers==0.11.1 transformers==4.30.2 huggingface-hub==0.25.2
cd 3d_diffuser_actor
pip install -e .
```

## install robomimic (dataloader)
```bash
git clone git@github.com:memmelma/robomimic_pcd.git
cd robomimic_pcd
pip install -e .
```
### (hyak) load cuda 11.8
```bash
module load cuda/11.8.0
```