# Set up environment
conda env create -f environment.yml

FYI: This environment doesn't work properly with Apple Silicon Chips.

# Train the model
python train.py

# Evaluate the model
python eval.py