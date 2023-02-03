# fedcmp

A repo to show the visual comparison between different federated learning algorithms

# Techinal Details

## Runtime Environment

- **OS**: Windows (Tested on Windows 10)
- Python 3.10.9
- `pip --verison`:

```bash
pip 23.0 from fedcmp\mnist\fedavg\venv\lib\site-packages\pip (python 3.10)
```

## Projec Structure

```bash
root
├───dataset
│   ├───algorithm
│   │   ├───driver codes
│   │   ├───venv
│   │   ├───requirements.txt
```

### Example

```bash
fedcmp (root)
├───mnist (dataset)
│   ├───fedavg (algorithm)
│   │   ├───client (client code)
│   │   │   ├───client.py
│   │   │   └───__init__.py
│   │   ├───server (server code)
│   │   │   ├───server.py
│   │   │   └───__init__.py
│   │   ├───venv (virtual environment)
│   │   ├───main.py (main script)
│   │   └───requirements.txt (dependencies)
│   └───fedprox (algorithm)

```

# Steps to run

## 1. Install dependencies

First of all **activate the virtual environment**. Then run the following command:

```bash
pip install -r requirements.txt
pip install flwr["simulation"]
```

## 2. Run the main script

```bash
python main.py
```
