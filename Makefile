PIP = pip

requirements: install-requirements

install-requirements:
	$(PIP) install -r requirements.txt

clean:
	find . \( -name "*.pyc" -o -name "*.pyo" \) -exec rm -f {} +
	find . -name "__pycache__" -exec rm -rf {} +

.PHONY: requirements install-requirements clean
