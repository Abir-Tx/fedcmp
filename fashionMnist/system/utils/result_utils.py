import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


#  def generate_graph(algorithm="", dataset="", goal="", times=10):
#      test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
#      test_loss = get_train_loss_for_one_algo(algorithm, dataset, goal, times)
#
#
#      print ("test_acc: ", test_acc)
#      print ("test_loss: ", test_loss)


    #  for i in range(times):
        #  plt.plot(test_loss[i], test_acc[i], label="run " + str(i))

    #  plt.xlabel("Train Loss")
    #  plt.ylabel("Test Accuracy")
    #  plt.title("Test Accuracy vs Train Loss for " + algorithm + " on " + dataset + " dataset")
    #  plt.legend()
    #  plt.clf()
    #
    #  fig, ax1 = plt.subplots()
    #  ax2 = ax1.twinx()
    #  x = range(1, 20 + 1)
    #  y1 = test_acc
    #  #  y2 = np.array(test_loss).ravel()  # flatten the y array to match the length of x
    #  ax1.plot(x, y1, "g-")
    #  ax2.plot(x, y2, "b-")
    #  ax1.set_xlabel("Communication round")
    #  ax1.set_ylabel("Test accuracy", color="g")
    #  ax2.set_ylabel("Train loss", color="b")
    #  plt.show()

#      fig, ax1 = plt.subplots()
    #  ax2 = ax1.twinx()
    #  ax1.plot( test_acc, "g-")
    #  ax2.plot( test_loss, "b-")
    #  ax1.set_xlabel("Communication round")
    #  ax1.set_ylabel("Test accuracy", color="g")
    #  ax2.set_ylabel("Train loss", color="b")
    #  plt.show()
    #  plt.savefig("../results/" + dataset + "_" + algorithm + "_" + goal + "_" + str(times) + ".png")

#  def generate_graph(algorithm="", dataset="", goal="", times=10):
#      test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
#      test_loss = get_train_loss_for_one_algo(algorithm, dataset, goal, times)
#
#
#      #  test_acc = np.array(test_acc).tolist()
#      #  test_loss = np.array(test_loss).tolist()
#      print ("test_acc: ", test_acc)
#      print ("test_loss: ", test_loss)
#      # Plot the data
#      fig, ax1 = plt.subplots()
#      ax2 = ax1.twinx()
#      ax1.plot(range(1, len(test_acc) + 1), test_acc, "g-")
#      ax2.plot(range(1, len(test_acc) + 1), test_loss, "b-")
#      ax1.set_xlabel("Communication round")
#      ax1.set_ylabel("Test accuracy", color="g")
#      ax2.set_ylabel("Train loss", color="b")
#      plt.title("Test Accuracy and Train Loss vs Communication Round for " + algorithm + " on " + dataset + " dataset")
#      plt.show()

def generate_graph(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)
    test_loss = get_train_loss_for_one_algo(algorithm, dataset, goal, times)
    test_acc = np.array(test_acc)
    test_loss = np.array(test_loss)

    test_acc = [test_acc.ravel().tolist() for acc in test_acc]  # convert numpy arrays to lists
    test_loss = [test_loss.ravel().tolist() for loss in test_loss]  # convert numpy arrays to lists

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(times):
        ax1.plot(range(1, len(test_acc[i]) + 1), test_acc[i], "g-", label="Run " + str(i))
        ax2.plot(range(1, len(test_loss[i]) + 1), test_loss[i], "b-")

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Test accuracy", color="g")
    ax2.set_ylabel("Train loss", color="b")
    plt.title("Test Accuracy and Train Loss vs Communication Rounds for " + algorithm + " on " + dataset + " dataset")
    ax1.legend()
    plt.show()





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
