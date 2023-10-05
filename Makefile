PIP = pip

requirements:
	$(PIP) install -r requirements.txt

clean:
	find . -name "*.pyc" -exec rm -f {} \;
	find . -name "*.pyo" -exec rm -f {} \;
	find . -name "__pycache__" -exec rm -rf {} \;

.PHONY: requirements
