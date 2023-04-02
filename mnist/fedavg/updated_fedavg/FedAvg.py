# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time
import sys

sys.stdout = open("FedAvg.txt", "w")

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Select the first GPU
torch.cuda.set_device(0)

# Increase the memory allocation to 8 GB
torch.cuda.empty_cache()
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.8, 0)
#  torch.cuda.set_per_process_memory_growth(True)


# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# Define the client class
class Client:
    def __init__(
        self, dataset, batch_size=32, learning_rate=0.1, local_epochs=20, num_workers=0
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_workers = num_workers


    def train(self):
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss().to(self.device)

        self.model.train()
        for epoch in range(self.local_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.model.state_dict()


# Define the server class
class Server:
    def __init__(self, clients, learning_rate=0.1, num_workers=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clients = clients
        self.learning_rate = learning_rate
        self.num_workers = num_workers

    def aggregate(self, weights):
        new_weights = {}
        for key in weights[0].keys():
            new_weights[key] = torch.zeros_like(weights[0][key]).to(self.device)
            for weight in weights:
                new_weights[key] += weight[key].to(self.device)
            new_weights[key] /= len(weights)

        return new_weights

    def train(self):
        weights = [client.train() for client in self.clients]
        aggregated_weights = self.aggregate(weights)

        for client in self.clients:
            client.model.load_state_dict(aggregated_weights)

        return aggregated_weights


def main(
    num_rounds=10, num_clients=10, batch_size=10, learning_rate=0.1, local_epochs=5
    ):
    #  CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=num_rounds, help='Number of federated learning rounds to run')
    parser.add_argument('--c', type=int, default=num_clients, help='Number of clients in the federated learning setting')
    parser.add_argument('--b', type=int, default=batch_size, help='Batch size for local training')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate for local training')
    parser.add_argument('--le', type=int, default=local_epochs, help='Number of epochs to run for local training')
    args = parser.parse_args()

    # Retrieve the command-line arguments
    num_rounds = args.r
    num_clients = args.c
    batch_size = args.b
    learning_rate = args.lr
    local_epochs = args.le
    # End of CLA

    # print the current params
    print ("Current parameters:")
    print("Number of rounds: ", num_rounds)
    print("Number of clients: ", num_clients)
    print("Batch size: ", batch_size)
    print("Learning rate: ", learning_rate)
    print("Local epochs: ", local_epochs)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start the time count
    start_time = time.time()


    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Split the data into non-iid subsets
    num_samples = len(train_dataset)
    num_samples_per_client = int(num_samples / num_clients)
    samples = torch.randperm(num_samples)
    client_datasets = []
    for i in range(num_clients):
        client_samples = samples[
            i * num_samples_per_client : (i + 1) * num_samples_per_client
        ]
        client_dataset = torch.utils.data.Subset(train_dataset, client_samples)
        client_datasets.append(client_dataset)

    # Create the clients and the server
    clients = [
        Client(
            client_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            num_workers=15,
        )
        for client_dataset in client_datasets
    ]
    server = Server(clients, learning_rate=learning_rate, num_workers=15)
    test_accs = []
    train_losses = []  # list to store the train loss vs communication round

    for round in range(num_rounds):
        print(f"Round {round+1}")
        weights = server.train()

        # Evaluate the model on the test dataset
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss().to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = server.clients[0].model(data)
                loss = criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.shape[0]
        accuracy = correct / total
        print(f"Test accuracy: {accuracy}")
        test_accs.append(accuracy)

        # Compute the train loss
        train_loss = 0.0
        for client in clients:
            train_loader = DataLoader(
                client.dataset, batch_size=batch_size, shuffle=True, num_workers=10
            )
            criterion = nn.CrossEntropyLoss().to(device)
            client.model.eval()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = client.model(data)
                loss = criterion(output, target)
                train_loss += loss.item() * data.shape[0]
            client.model.train()
        train_loss /= num_samples
        print(f"Train loss: {train_loss}")
        train_losses.append(train_loss)

    # Plot the test accuracy and train loss vs communication round
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(1, num_rounds + 1), test_accs, "g-")
    ax2.plot(range(1, num_rounds + 1), train_losses, "b-")
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Test accuracy", color="g")
    ax2.set_ylabel("Train loss", color="b")
    #  filename = str(num_rounds) + "_r" + str(num_clients) + "_c" + str(batch_size) + "_b" + str(learning_rate) + "_lr" + str(local_epochs) + "_le.png"
    filename = f"{num_rounds}r_{num_clients}c_{batch_size}b_{learning_rate}lr_{local_epochs}le.png"
    plt.savefig("images/" + filename)

    # End the time count
    end_time = time.time()
    print ("----------------------------------------")
    print(f"Total time: {end_time - start_time}")
    print ("----------------------------------------")

    # Restore standard output
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
