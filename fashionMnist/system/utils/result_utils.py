import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_graph(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
    test_loss = get_train_loss_for_one_algo(algorithm, dataset, goal, times)

    for i in range(times):
        plt.plot(test_loss[i], test_acc[i], label="run " + str(i))

    plt.xlabel("Train Loss")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Train Loss for " + algorithm + " on " + dataset + " dataset")
    plt.legend()
    plt.savefig("../results/" + dataset + "_" + algorithm + "_" + goal + ".png")
    plt.show()
    plt.clf()



def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))

def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


def get_train_loss_for_one_algo(algorithm="", dataset="", goal="", times=10):
    train_loss = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        train_loss.append(np.array(read_train_loss(file_name, delete=False, is_train=True)))

    return train_loss


def read_train_loss(file_name, delete=False, is_train=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        if is_train:
            rs_train_loss = np.array(hf.get('rs_train_loss'))
            data = rs_train_loss
        else:
            rs_test_acc = np.array(hf.get('rs_test_acc'))
            data = rs_test_acc

    if delete:
        os.remove(file_path)
    print("Length: ", len(data))

    return data
