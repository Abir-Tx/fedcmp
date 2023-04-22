# fedcmp

A repo to show the visual comparison between different federated learning algorithms. This is the official repo for our paper on An Analysis of **Personalized Federated Learning Algorithms with an Improvement of Intuitive Protected Model Training**

## Algorithms planned to be implemented

<details>
<summary>Click to expand</summary>

- [x] FedAvg
- [x] FedProx
- [x] FedProto
- [x] FedBABU
- [x] APPLE

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

## MNIST FedAvg

### 1. Install dependencies

First of all **activate the virtual environment**. Then run the following command:

```bash
pip install -r requirements.txt
pip install flwr["simulation"]
```

### 2. Run the main script

```bash
python main.py
```

## MNIST FedProx

### 1. Install dependencies

First of all **activate the virtual environment**. Then run the following command:

```bash
pip install -r requirements.txt
pip install -U flwr["simulation"]

```

### 2. Run the main script

```bash
python main.py num_epochs=5 num_rounds=1000 iid=True
```

## FMNIST

The FMNIST dataset and the algorithms operating on the dataset is implemented inside the `fashionMnist` directory. The base code & inspiration is taken from the [PFL-NON-IID](https://github.com/TsingZ0/PFL-Non-IID) repo. The code is modified to work with our project structure with added support for logging, visualization, and more. Currently the algorithms implemented inside the `fashionMnist` directory are:

- FedAvg
- FedProto
- FedBABU
- APPLE

The directory structure is not same as the `mnist` directory. The `fashionMnist` directory is structured as follows:

```bash
fashionMnist
|--- dataset
|--- logs
|--- results
|--- system
|  |--- models
|  |--- utils
|  |--- main.py
|--- venv
|--- requirements.txt

```

### 1. Install dependencies

There is `requirements.txt` file for all the algorithms inside the `fashionMnist` directory. First of all **activate the virtual environment**. Then run the following command:

```bash
pip install -r requirements.txt
```

### 2. Run the main script

The main script has to be run from the **system** directory. The main script is `main.py` and it is located inside the `fashionMnist` directory. The main script has the following arguments:

```bash
python main.py -data fmnist -m cnn -algo FedAvg -gr 10 -did 0 -go cnn -nc 1
```

# Credits

- Used the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) from TensorFlow Datasets
- Used the [Flower](https://flower.dev/) library for federated learning
- Used the [fedavg_mnist](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/publications/fedavg_mnist) example code for the FedAvg algorithm implementation. The code is modified to work with the MNIST dataset. Couldn't fork the specific directory so I copied the code and modified it.
- Used the [PFL-NON-IID](https://github.com/TsingZ0/PFL-Non-IID) repo for the FMNIST dataset and the algorithms operating on the dataset.
