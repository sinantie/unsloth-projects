# Introduction
This is a simple repo that provides a `uv`-enabled environment for running unsloth V+L models on consumer GPUs.

# Installation

* Install `uv` using the standalone installer:
```
curl -LsSf https://astral.sh/uv/install.sh | sh`
```

* Create a virtual environment folder:
```
uv venv
```

* Install the depdendencies. You can either do it by syncing manually
```
uv sync
```

or just run the project that will first install all dependencies and download the model that is included in `main.py`:
```
uv run main.py
```

# Running the code

* Either execute the code using the main entry point of your python project, e.g., :
```
uv run main.py
```
* or open the included Jupyter Notebook on VS Code, making sure you use the kernel pointing to the virtual environment included in this project (`.venv/bin/python`). 
You might need to establish a Remote SSH connection to the machine, dependening on where the GPUs are hosted.
