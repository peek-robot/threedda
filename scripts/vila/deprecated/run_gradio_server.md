# on compute node run

start the model server
```
cd /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/VILA/
conda activate vila
MODEL_PATH=/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/checkpoints/finetuned/nvila/vila_3b_all_path
python -W ignore server.py --conv-mode vicuna_v1 --model-path $MODEL_PATH
```

then start gradio server on the same node
```
cd /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/VILA/
conda activate vila
python gradio_inference.py
```


<!-- (tunnel-env) mmemmel@batch-block5-00741:/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila$ npx tmole 8000 -->