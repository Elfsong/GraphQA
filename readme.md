# GraphQA

![Examples](/data/img/example.png "Example")

# Step 1. Set up environment
```shell
conda env create -f environment.yml
```

_FYI: This environment doesn't work properly with Apple Silicon Chips_

# Step 2. Dataset preparation
```shell
mkdir ./data/train
mkdir ./data/validation
python dataset_generate.py --worker_num 8 --split train --datasize 10000
python dataset_generate.py --worker_num 8 --split validation --datasize 1000
```

# Step 3. Train the model
```shell
python train.py
```

# Step 4. Evaluate the model
```shell
# TODO
python eval.py
```
