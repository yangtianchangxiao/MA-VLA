# Setup 2025 SOTA VLA Benchmarks for Multi-Morphology Evaluation

# 1. Create evaluation environment
conda activate AET_FOR_RL
cd /home/cx/AET_FOR_RL/vla/evaluation

# 2. Install benchmark dependencies
pip install libero gymnasium mujoco
pip install transformers datasets accelerate
pip install flash-attn wandb tensorboard

# 3. Setup LIBERO benchmark
cd LIBERO
pip install -e .

# 4. Setup OpenVLA comparison
cd ../OpenVLA
pip install -e .

# 5. Run morphology evaluation
cd ..
python morphology_vla_adapter.py

# 6. Compare against SOTA baselines
# OpenVLA-OFT: 97.1% success rate (target to beat)
# Pi0, JAT, GPT-4o: comparison baselines