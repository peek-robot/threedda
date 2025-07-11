# pick_data_gen



Start by setting up the conda env and installing the [base package](#base-package). Then install either [data generation](#data-generation) or [policy learning](#policy-learning).

WARNING: Due to conflicting cuda versions, it is recommended to create separate conda envs for [data generation w/ curobo] and [policy learning w/ 3dda]!

ATTENTION: On hyak you have to load the required cuda version on a compute node!
```bash
module load cuda/11.8.0
```

## base package
```bash
ENV_NAME=[!REPLACE ME!]
mamba create -n $ENV_NAME python=3.10
mamba activate $ENV_NAME
git clone https://github.com/memmelma/pick_data_gen.git
cd pick_data_gen
git submodule update --init --recursive
pip install -e .
```

## data generation

### install curobo (takes up to 20min)
```bash
ROOT_DIR=$(pwd)
sudo apt install nvidia-cuda-toolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

mamba install -c conda-forge 'gcc[version=">=6,<11"]' 'gxx[version=">=6,<11"]'
export CUDA_HOME=$CONDA_PREFIX
mamba install nvidia/label/cuda-11.8.0::cuda
mamba install nvidia/label/cuda-11.8.0::cuda-nvcc
mamba install nvidia/label/cuda-11.8.0::cuda-cudart

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
python scripts/threedda/run_3dda.py --dataset data/pick_1000_1_objs_va_high_cam.hdf5 --augment_rgb --augment_pcd --obs_crop --name debug --history 2 --horizon 8 --fps_subsampling_factor 5 --num_epochs 1500 --eval_every_n_epochs 1
```

# VLM
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