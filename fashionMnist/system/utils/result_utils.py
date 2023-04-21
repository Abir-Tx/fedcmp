import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("fedcmpLogger")


def generate_graph(algorithm="", dataset="", goal="", times=10, rounds=3, clients=1):
    # Validate input parameters
    if not algorithm:
        raise ValueError("Algorithm name cannot be empty")
    if not dataset:
        raise ValueError("Dataset name cannot be empty")
    if not goal:
        raise ValueError("Goal name cannot be empty")
    if times <= 0:
        raise ValueError("Number of times must be positive")

    #  Get the results
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
    test_loss = get_train_loss_for_one_algo(algorithm, dataset, goal, times)
    test_acc = np.array(test_acc)
    test_loss = np.array(test_loss)

    test_acc = [
        test_acc.ravel().tolist() for acc in test_acc
    ]  # convert numpy arrays to lists
    test_loss = [
        test_loss.ravel().tolist() for loss in test_loss
    ]  # convert numpy arrays to lists

    #  Plot the results
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(times):
        ax1.plot(
            range(1, len(test_acc[i]) + 1), test_acc[i], "g-", label="Run " + str(i)
        )
        ax2.plot(range(1, len(test_loss[i]) + 1), test_loss[i], "b-")

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Test accuracy", color="g")
    ax2.set_ylabel("Train loss", color="b")
    ax1.legend()
    plt.title(
        "Test Accuracy and Train Loss vs Communication Rounds for "
        + algorithm
        + " on "
        + dataset
        + " dataset"
    )

    # Save and show plot
    filename = (
        dataset
        + "_"
        + algorithm
        + "_rounds_"
        + str(rounds)
        + "_clients_"
        + str(clients)
    )
    plt.savefig("../results/" + filename + ".png")
    plt.show()
    print(
        "The graph has been saved to the: ",
        "../results/" + filename + ".png",
    )


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    logger.info("std for best accurancy: %s", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))
    logger.info("mean for best accurancy: %s", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, "r") as hf:
        rs_test_acc = np.array(hf.get("rs_test_acc"))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


def get_train_loss_for_one_algo(algorithm="", dataset="", goal="", times=10):
    train_loss = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        train_loss.append(
            np.array(
                read_train_loss_then_delete(file_name, delete=False, is_train=True)
            )
        )

    return train_loss


def read_train_loss_then_delete(file_name, delete=False, is_train=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, "r") as hf:
        if is_train:
            rs_train_loss = np.array(hf.get("rs_train_loss"))
            data = rs_train_loss
        else:
            rs_test_acc = np.array(hf.get("rs_test_acc"))
            data = rs_test_acc

    if delete:
        os.remove(file_path)
    print("Length: ", len(data))

    return data
