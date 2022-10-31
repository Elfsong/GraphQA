# Step 1. Set up environment
```shell
conda env create -f environment.yml
```

_FYI: This environment doesn't work properly with Apple Silicon Chips_

# Step 2. Dataset preparation
```shell
python dataset_generate.py
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