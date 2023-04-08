#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import datetime

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverper import FedPer
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.serverproto import FedProto
from flcore.servers.serverrod import FedROD

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *

from utils.result_utils import average_data
from utils.result_utils import generate_graph
from utils.mem_utils import MemReporter


warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32



logger = logging.getLogger("fedcmpLogger")
logger.setLevel(logging.DEBUG)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"fedcmpLogger_{timestamp}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Disable propagation and set propagation to False
logger.propagate = False


def run(args):
    time_list = []
    train_losses = []  # list to store training losses
    test_accuracies = []  # list to store test accuracies
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":  # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(
                    1 * 28 * 28, num_classes=args.num_classes
                ).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(
                    args.device
                )

        elif model_str == "cnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(
                    in_features=1, num_classes=args.num_classes, dim=1024
                ).to(args.device)
            else:
                args.model = FedAvgCNN(
                    in_features=3, num_classes=args.num_classes, dim=10816
                ).to(args.device)
        elif model_str == "harcnn":
            if args.dataset == "har":
                args.model = HARCNN(
                    9,
                    dim_hidden=1664,
                    num_classes=args.num_classes,
                    conv_kernel_size=(1, 9),
                    pool_kernel_size=(1, 2),
                ).to(args.device)
            elif args.dataset == "pamap":
                args.model = HARCNN(
                    9,
                    dim_hidden=3712,
                    num_classes=args.num_classes,
                    conv_kernel_size=(1, 9),
                    pool_kernel_size=(1, 2),
                ).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        else:
            raise NotImplementedError

        server.train()  # get the train loss and test accuracy

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(
        dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times
    )

    generate_graph(
        dataset=args.dataset,
        algorithm=args.algorithm,
        goal=args.goal,
        times=args.times,
    )

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "-go", "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-data", "--dataset", type=str, default="mnist")
    parser.add_argument("-nb", "--num_classes", type=int, default=10)
    parser.add_argument("-m", "--model", type=str, default="cnn")
    parser.add_argument("-lbs", "--batch_size", type=int, default=10)
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.005,
        help="Local learning rate",
    )
    parser.add_argument("-ld", "--learning_rate_decay", type=bool, default=False)
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument("-gr", "--global_rounds", type=int, default=2000)
    parser.add_argument("-ls", "--local_steps", type=int, default=1)
    parser.add_argument("-algo", "--algorithm", type=str, default="FedAvg")
    parser.add_argument(
        "-jr",
        "--join_ratio",
        type=float,
        default=1.0,
        help="Ratio of clients per round",
    )
    parser.add_argument(
        "-rjr",
        "--random_join_ratio",
        type=bool,
        default=False,
        help="Random ratio of clients per round",
    )
    parser.add_argument(
        "-nc", "--num_clients", type=int, default=2, help="Total number of clients"
    )
    parser.add_argument(
        "-pv", "--prev", type=int, default=0, help="Previous Running times"
    )
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        "-eg", "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )
    parser.add_argument(
        "-dp", "--privacy", type=bool, default=False, help="differential privacy"
    )
    parser.add_argument("-dps", "--dp_sigma", type=float, default=0.0)
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items")
    parser.add_argument("-ab", "--auto_break", type=bool, default=False)
    parser.add_argument("-dlg", "--dlg_eval", type=bool, default=False)
    parser.add_argument("-dlgg", "--dlg_gap", type=int, default=100)
    parser.add_argument("-bnpc", "--batch_num_per_client", type=int, default=2)
    # practical
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )
    parser.add_argument(
        "-tsr",
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when /training locally",
    )
    parser.add_argument(
        "-ssr",
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when sending global model",
    )
    parser.add_argument(
        "-ts",
        "--time_select",
        type=bool,
        default=False,
        help="Whether to group and select clients at each round according to time cost",
    )
    parser.add_argument(
        "-tth",
        "--time_threthold",
        type=float,
        default=10000,
        help="The threthold for droping slow clients",
    )
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument(
        "-bt",
        "--beta",
        type=float,
        default=0.0,
        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer",
    )
    parser.add_argument(
        "-lam", "--lamda", type=float, default=1.0, help="Regularization weight"
    )
    parser.add_argument(
        "-mu", "--mu", type=float, default=0, help="Proximal rate for FedProx"
    )
    parser.add_argument(
        "-K",
        "--K",
        type=int,
        default=5,
        help="Number of personalized training steps for pFedMe",
    )
    parser.add_argument(
        "-lrp",
        "--p_learning_rate",
        type=float,
        default=0.01,
        help="personalized learning rate to caculate theta aproximately using K steps",
    )
    # FedFomo
    parser.add_argument(
        "-M",
        "--M",
        type=int,
        default=5,
        help="Server only sends M client models to one client at each round",
    )
    # FedMTL
    parser.add_argument(
        "-itk",
        "--itk",
        type=int,
        default=4000,
        help="The iterations for solving quadratic subproblems",
    )
    # FedAMP
    parser.add_argument(
        "-alk",
        "--alphaK",
        type=float,
        default=1.0,
        help="lambda/sqrt(GLOABL-ITRATION) according to the paper",
    )
    parser.add_argument("-sg", "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument("-al", "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument("-pls", "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument("-ta", "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument("-fts", "--fine_tuning_steps", type=int, default=1)
    # APPLE
    parser.add_argument("-dlr", "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument("-L", "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument("-nd", "--noise_dim", type=int, default=512)
    parser.add_argument("-glr", "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=512)
    parser.add_argument("-se", "--server_epochs", type=int, default=1000)
    parser.add_argument("-lf", "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD
    parser.add_argument("-slr", "--server_learning_rate", type=float, default=1.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print(
            "Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma)
        )
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack evaluate: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack evaluate round gap: {}".format(args.dlg_gap))
    print("=" * 50)

    run(args)
