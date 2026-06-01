# lerobot-handson — convenience targets
#
# `make smoke` runs the whole exercise path offline on CPU (no GPU/net/Miyabi).
# Override PYTHON to point at an environment that has lerobot installed, e.g.:
#   make smoke PYTHON=/path/to/venv/bin/python
# (also put that venv's bin on PATH so lerobot-train / lerobot-eval resolve).

PYTHON ?= python

.PHONY: smoke synth convert clean

smoke:
	PYTHON=$(PYTHON) bash tools/smoke_test.sh

synth:
	$(PYTHON) tools/make_synthetic_dataset.py --format lerobot --root .smoke/synthetic

convert:
	$(PYTHON) tools/make_synthetic_dataset.py --format raw --out .smoke/raw
	$(PYTHON) 02_convert/convert_sample.py --raw .smoke/raw --root .smoke/converted

clean:
	rm -rf .smoke
