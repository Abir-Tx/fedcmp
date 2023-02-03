# fedcmp

A repo to show the visual comparison between different federated learning algorithms.

## Algorithms planned to be implemented

<details>
<summary>Click to expand</summary>

- [x] FedAvg
- [ ] FedProx
- [ ] PerFedAvg
- [ ] QFed
- [ ] DPSGD
</details>
<br>

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

# Credits

- Used the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) from TensorFlow Datasets
- Used the [Flower](https://flower.dev/) library for federated learning
- Used the [fedavg_mnist](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/publications/fedavg_mnist) example code for the FedAvg algorithm implementation. The code is modified to work with the MNIST dataset. Couldn't fork the specific directory so I copied the code and modified it.
