# Python Environment Setup with `uv`

This guide will help you set up a Python environment using `uv`, a fast Python package installer and resolver.

## 1. Install `uv`

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Alternative: Using pip
```bash
pip install uv
```

## 2. Initialize and Sync Your Environment  

Create and populate a virtual environment with your dependencies:  

```bash
# To ensure a safe setup, `uv sync` should work fine.
uv venv .venv --python 3.10
uv sync
# Check if the virtual environment is created with the name `.venv`.  
# If not, explicitly create it using the first command.
```

This will:
- Create a virtual environment in `.venv/`
- Install packages from `requirements.txt` and/or `pyproject.toml`

## 3. Activate the Virtual Environment

### macOS/Linux
```bash
source .venv/bin/activate
```

### Windows
```powershell
.venv\Scripts\activate
```

## 4. Managing Packages During Development

### Add a Package
```bash
uv add package_name
```

### Remove a Package
```bash
uv remove package_name
```

### Update Requirements Files
After adding or removing packages, update your requirements file:

```bash
# Generate/update requirements.txt from the current environment
uv pip freeze > requirements.txt
```

## Additional Information

For more details on Python environment management with `uv`, check out the comprehensive guide at [FloCode: Python Environments Again with uv](https://flocode.substack.com/p/044-python-environments-again-uv).
