# Define the Python interpreter to use
PYTHON = python
PIP = pip

DATA ?= fmnist
GR ?= 3
NC ?= 1

# Define the default target
all: run


run:
	cd system && \
	${PYTHON} main.py -data $(DATA) -m cnn -algo FedAvg -gr $(GR) -did 0 -go cnn -nc $(NC)


config:
	@if [ -d "venv" ]; then \
		echo "Venv exists"; \
	else \
		${PYTHON} -m venv venv; \
		. venv/bin/activate && \
        	echo "venv activated"; \
		${PIP} install -r requirements.txt;\
	fi
	. venv/bin/activate; \
	mkdir -p logs results

clean:
	@echo "Clearing the generated venv and other folders";\
	rm -r logs results venv
