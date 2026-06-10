# lerobot-handson — convenience targets

.PHONY: clean

# Remove local scratch output (synthetic data / CPU rehearsal runs under .smoke/).
clean:
	rm -rf .smoke
