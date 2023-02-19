# FedProx MNIST

The following baseline replicates the experiments in _Federated Optimization in Heterogeneous Networks_ (Li et al., 2018), which proposed the FedProx algorthim.

**Paper Authors:**

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar and Virginia Smith.

**[Link to paper.](https://arxiv.org/abs/1812.06127)**

## Training Setup

### CNN Architecture

The CNN architecture is detailed in the paper and used to create the **FedProx MNIST** baseline.

| Layer | Details                                                |
| ----- | ------------------------------------------------------ |
| 1     | Conv2D(1, 32, 5, 1, 1) <br/> ReLU, MaxPool2D(2, 2, 1)  |
| 2     | Conv2D(32, 64, 5, 1, 1) <br/> ReLU, MaxPool2D(2, 2, 1) |
| 3     | FC(64 _ 7 _ 7, 512) <br/> ReLU                         |
| 5     | FC(512, 10)                                            |

### Training Paramaters

| Description      | Value                  |
| ---------------- | ---------------------- |
| loss             | cross entropy loss     |
| optimizer        | SGD with proximal term |
| learning rate    | 0.03 (by default)      |
| local epochs     | 5 (by default)         |
| local batch size | 10 (by default)        |

## Running experiments

The `config.yaml` file containing all the tunable hyperparameters and the necessary variables can be found under the `conf` folder.
[Hydra](https://hydra.cc/docs/tutorials/) is used to manage the different parameters experiments can be ran with.

To run using the default parameters, just enter `python main.py`, if some parameters need to be overwritten, you can do it like in the following example:

```sh
python main.py num_epochs=5 num_rounds=1000 iid=True
```

Results will be stored as timestamped folders inside either `outputs` or `multiruns`, depending on whether you perform single- or multi-runs.
