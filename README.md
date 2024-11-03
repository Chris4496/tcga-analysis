# Setting up the TCGA Analysis Project
Set up the project environment

## Prerequisites
- Python 3.x installed
- Git installed
- Terminal/Command Prompt access

## Installation Steps

### 1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd tcga-analysis
```

### 3. Install Dependencies
```bash
poetry install
```

### 4. Activate Virtual Environment
```bash
poetry shell
```

## Common Commands

| Command | Description |
|---------|------------|
| `poetry add package_name` | Add new dependencies |
| `poetry update` | Update all dependencies |
| `poetry show` | List all dependencies |
| `exit` | Deactivate virtual environment |

## Troubleshooting

### Poetry Installation Fails on Windows
Try the Windows-specific installer:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### SSL Certificate Errors
Install certificates:
```bash
pip install --upgrade certifi
```

### Virtual Environment Issues
Remove and reinstall the virtual environment:
```bash
poetry env remove python
poetry install
```