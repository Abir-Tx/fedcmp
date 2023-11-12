# fedcmp

A repo to show the visual comparison between different federated learning algorithms. This is the official repo for our paper on An Analysis of **Personalized Federated Learning Algorithms with an Intuitive Protected Model Training**

## Algorithms planned to be implemented

<details>
<summary>Click to expand</summary>

- [x] FedAvg
- [x] FedProto
- [x] FedBABU
- [x] APPLE

</details>
<br>

# Techinal Details

## Runtime Environment

- **OS**: Windows (Tested on Windows 10 21H2), Linux (Arch Linux 2023.4.1)
- Python 3.10.9
- `pip --verison`:

## Projec Structure

## FMNIST

```bash
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

# Steps to run

## Manually

The FMNIST dataset and the algorithms operating on the dataset is implemented inside `this` directory. The base code & inspiration is taken from the [PFL-NON-IID](https://github.com/TsingZ0/PFL-Non-IID) repo. The code is modified to work with our project structure with added support for logging, visualization, and more. Currently the algorithms implemented inside the `this` directory are:

- FedAvg
- FedProto
- FedBABU
- APPLE

### 1. Install dependencies

There is `requirements.txt` file for all the algorithms inside `this` directory. First of all **activate the virtual environment**. Then run the following command:

```bash
pip install -r requirements.txt
```

### 2. Run the main script

The main script has to be run from the **system** directory. The main script is `main.py` and it is located inside `this` directory. The main script has the following arguments:

```bash
python main.py -data fmnist -m cnn -algo FedAvg -gr 10 -did 0 -go cnn -nc 1
```

## Using Make

I have added a `Makefile` to run the algorithms. The `Makefile` is located inside `this` directory. The `Makefile` has the following commands:

- `make run`: Run the main script with the default arguments
- `make config`: Makes sure that all the needed directories are created & packages are installed and then check if the `venv` is activated or not. If not, then activate the `venv`.
- `make clean`: Clean the `venv` and the `__pycache__` directories & other unwanted files/directories.

The `make run` command runs the main script with the following arguments:

```bash
make run DATA=Cifar10 GR=3 NC=1 ALGO=FedAvg
```

Here the `DATA` argument is the dataset to use, the `GR` argument is the number of global rounds to run the algorithm, the `NC` argument is the number of clients to use, and the `ALGO` argument is the algorithm to use. The `DATA` argument is required, the `GR` argument is optional and the default value is 3, the `NC` argument is optional and the default value is 1, and the `ALGO` argument is optional and the default value is FedAvg.

#### Arguments description

- `-data`: The dataset to use. Currently only `fmnist` is supported.
- `-m`: The model to use. Currently only `cnn` is supported.
- `-algo`: The algorithm to use. Currently only `FedAvg`, `FedProto`, `FedBABU`, and `APPLE` are supported.
- `-gr`: The number of global rounds to run the algorithm.
- `-did`: The device id. This is used to differentiate between different devices. This is used for logging purposes.
- `-go`: The global optimizer to use. Currently only `cnn` is supported.
- `-nc`: The number of clients to use. Currently the maximum number of clients supported is 20. (Tested on core-i9 machine)

So to run different algorithms on the FMNIST dataset, you just have to change the `-algo` argument. For example, to run the FedProto algorithm, you have to run the following command:

```bash
python main.py -data fmnist -m cnn -algo FedProto -gr 10 -did 0 -go cnn -nc 1
```

#### Getting the results

The generated images and the execution logs are saved in separate directories. The images are saved in the `results` directory and the logs are saved in the `logs` directory.

# Credits

- Used the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) from TensorFlow Datasets
- Used the [Flower](https://flower.dev/) library for federated learning
- Used the [fedavg_mnist](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/publications/fedavg_mnist) example code for the FedAvg algorithm implementation. The code is modified to work with the MNIST dataset. Couldn't fork the specific directory so I copied the code and modified it.
- Used the [PFL-NON-IID](https://github.com/TsingZ0/PFL-Non-IID) repo for the FMNIST dataset and the algorithms operating on the dataset.

# Authors

<a href="https://github.com/abir-tx"><img src="https://avatars.githubusercontent.com/u/28858998?v=4" width="50" height="50" title="Mushfiqur Rahman Abir"/></a> <a href="https://github.com/karit7"><img src="https://avatars.githubusercontent.com/u/120469589?v=4" width="50" height="50" title="Md. Tanzib Hosain" /></a>

# License

[MIT](https://choosealicense.com/licenses/mit/)
