# pick_data_gen



Start by setting up the conda env and installing the [base package](#base-package). Then install either [data generation](#data-generation) or [policy learning](#policy-learning).

WARNING: Due to conflicting cuda versions, it is recommended to create separate conda envs for [data generation w/ curobo] and [policy learning w/ 3dda]!

ATTENTION: On hyak you have to load the required cuda version on a compute node!
```bash
module load cuda/11.8.0
```

## base package
```bash
git clone https://github.com/memmelma/problem_reduction.git
cd problem_reduction
git submodule update --init --recursive
```

```bash
ENV_NAME=[!REPLACE ME!]
mamba create -n $ENV_NAME python=3.10
mamba activate $ENV_NAME
pip install -e .
```

## data generation

### install curobo (takes up to 20min)
```bash
ROOT_DIR=$(pwd)
sudo apt install nvidia-cuda-toolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

mamba install -c conda-forge 'gcc[version=">=6,<11"]' 'gxx[version=">=6,<11"]' -y
export CUDA_HOME=$CONDA_PREFIX
mamba install nvidia/label/cuda-11.8.0::cuda -y
mamba install nvidia/label/cuda-11.8.0::cuda-nvcc -y
mamba install nvidia/label/cuda-11.8.0::cuda-cudart -y

cd third_party/curobo
pip install -e . --no-build-isolation
cd $ROOT_DIR
```

### install transformers
```bash
pip install transformers==4.30.2
```

### example cmd
```bash
python scripts/datagen/generate.py --identifier debug
```

## policy learning

### install diffuser actor (policy)
```bash
ROOT_DIR=$(pwd)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install diffusers==0.11.1 transformers==4.30.2 huggingface-hub==0.25.2
pip install openai openai-clip
# downgrade numpy to ensure diffusers compatiblity
pip install "numpy<2"

cd third_party/3d_diffuser_actor
pip install -e .
cd $ROOT_DIR
```

### install robomimic (dataloader)
```bash
ROOT_DIR=$(pwd)
cd third_party/robomimic_pcd
pip install -e .
cd $ROOT_DIR
```

### example cmd
```bash
python scripts/threedda/run_3dda.py --dataset data/pick_and_place_1000_3_objs_va_high_cam.hdf5 --augment_rgb --augment_pcd --obs_crop --name debug --history 2 --horizon 8 --fps_subsampling_factor 5 --num_epochs 1500 --eval_every_n_epochs 1 --epoch_every_n_steps 1
```

# VLM

### install VILA1.5
```bash
ROOT_DIR=$(pwd)
conda create -n vila python=3.10 -y
conda activate vila

conda install -c nvidia cuda-toolkit -y
pip install --upgrade pip  # enable PEP 660 support
# this is optional if you prefer to system built-in nvcc.

cd third_party/VILA
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -e .
pip install -e ".[train]"

pip install git+https://github.com/huggingface/transformers@v4.36.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cd $ROOT_DIR
pip install -e .
```

### example cmd
```bash
python scripts/vila/inference/server_vlm.py --model_path memmelma/vila_3b_blocks_path_mask_fast
```

# real robot - Franka

### install controller (Markus' robits)
```bash
ROOT_DIR=$(pwd)
cd third_party/robtis_fork
pip install -e .
cd $ROOT_DIR
```

### install robot env and perception
```bash
ROOT_DIR=$(pwd)
cd third_party/franka_sim2real
pip install -e .
cd $ROOT_DIR
```