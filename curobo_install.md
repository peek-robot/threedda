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