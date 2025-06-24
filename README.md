# pick_data_gen
cd pick_data_gen
pip install -e .

## install modules (eval, logging, data gen)
pip mujoco shapely wandb meshcat trimesh

## install diffuser actor (policy)
git clone https://github.com/memmelma/3d_diffuser_actor.git
cd 3d_diffuser_actor
conda create -f environment.yaml

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/dgl-1.1.3%2Bcu118-cp38-cp38-manylinux1_x86_64.whl
pip install diffusers==0.11.1 transformers==4.30.2 huggingface-hub==0.25.2
cd 3d_diffuser_actor
pip install -e .

## install robomimic (dataloader)
git clone git@github.com:memmelma/robomimic_pcd.git
cd robomimic_pcd
pip install -e .

### load cuda 11.8
module load cuda/11.8.0
