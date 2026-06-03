CONDA   ?= $(HOME)/miniconda3/bin/conda
ENV     ?= rseg
RUN     = $(CONDA) run -n $(ENV)
DATA    ?= ./data/LoveDA
CKPT    ?= ./checkpoints
LOG     ?= ./logs

# ------------------------------------------------------------------ #
#  Environment
# ------------------------------------------------------------------ #
.PHONY: env
env:
	$(CONDA) create -n $(ENV) python=3.11 -y
	$(RUN) pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
	$(RUN) pip install numpy scikit-learn scipy omegaconf pillow tqdm torchgeo -q
	@echo "Environment '$(ENV)' ready."

# For GPU machines replace cpu wheel with the appropriate cu* wheel:
.PHONY: env-gpu
env-gpu:
	$(CONDA) create -n $(ENV) python=3.11 -y
	$(RUN) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
	$(RUN) pip install numpy scikit-learn scipy omegaconf pillow tqdm torchgeo rasterio -q

# ------------------------------------------------------------------ #
#  Data
# ------------------------------------------------------------------ #
.PHONY: download verify
download:
	$(RUN) python -c "\
from torchgeo.datasets import LoveDA; \
import os; os.makedirs('$(DATA)', exist_ok=True); \
LoveDA(root='$(DATA)', split='train', download=True); \
LoveDA(root='$(DATA)', split='val',   download=True)"

verify:
	$(RUN) python tests/verify_loveda.py

# ------------------------------------------------------------------ #
#  Training — quick presets
# ------------------------------------------------------------------ #

# Fast CPU test: ViT-S/14, 224px, 1 warmup + 1 round
.PHONY: train-fast
train-fast:
	PYTHONUNBUFFERED=1 $(RUN) python -u train.py \
	  data.root=$(DATA) \
	  data.batch_size=8 \
	  data.num_workers=4 \
	  data.image_size=224 \
	  model.backbone=dinov2_vits14 \
	  model.frozen_backbone=true \
	  self_training.warmup_epochs=1 \
	  self_training.n_rounds=1 \
	  self_training.epochs_per_round=1 \
	  logging.save_dir=$(CKPT)

# Full CPU run: ViT-S/14, 224px, default rounds
.PHONY: train-cpu
train-cpu:
	PYTHONUNBUFFERED=1 $(RUN) python -u train.py \
	  data.root=$(DATA) \
	  data.batch_size=8 \
	  data.num_workers=4 \
	  data.image_size=224 \
	  model.backbone=dinov2_vits14 \
	  logging.save_dir=$(CKPT)

# Full GPU run: ViT-B/14, 448px
.PHONY: train-gpu
train-gpu:
	PYTHONUNBUFFERED=1 $(RUN) python -u train.py \
	  data.root=$(DATA) \
	  data.batch_size=16 \
	  data.num_workers=8 \
	  data.image_size=448 \
	  model.backbone=dinov2_vitb14 \
	  logging.save_dir=$(CKPT)

# ------------------------------------------------------------------ #
#  Evaluation
# ------------------------------------------------------------------ #
.PHONY: eval
eval:
	$(RUN) python evaluate.py \
	  data.root=$(DATA) \
	  data.image_size=224 \
	  model.backbone=dinov2_vits14

# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #
.PHONY: test
test:
	$(RUN) python tests/smoke_test.py

# ------------------------------------------------------------------ #
#  Misc
# ------------------------------------------------------------------ #
.PHONY: clean
clean:
	rm -rf $(CKPT) $(LOG) __pycache__ */__pycache__ */*/__pycache__

.PHONY: help
help:
	@echo ""
	@echo "Targets:"
	@echo "  make env          -- create conda env + install deps (CPU)"
	@echo "  make env-gpu      -- create conda env + install deps (GPU/CUDA 12.1)"
	@echo "  make download     -- download LoveDA dataset"
	@echo "  make verify       -- verify dataset structure"
	@echo "  make train-fast   -- quick end-to-end test (CPU, ~15 min)"
	@echo "  make train-cpu    -- full training on CPU  (ViT-S, 224px)"
	@echo "  make train-gpu    -- full training on GPU  (ViT-B, 448px)"
	@echo "  make eval         -- evaluate best checkpoint"
	@echo "  make test         -- run smoke test (no internet needed)"
	@echo "  make clean        -- remove checkpoints and caches"
	@echo ""
	@echo "Overrides:  DATA=/path/to/LoveDA  CKPT=./checkpoints  ENV=rseg"
	@echo ""
