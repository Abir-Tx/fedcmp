import torch
import copy
import time
import numpy as np
import math
from flcore.clients.clientamp import clientAMP
from flcore.servers.serverbase import Server
from threading import Thread


class FedAMP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAMP)

        self.alphaK = args.alphaK
        self.sigma = args.sigma

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()


    def send_models(self):
        assert (len(self.selected_clients) > 0)

        if len(self.uploaded_models) > 0:
            for c in self.selected_clients:
                mu = copy.deepcopy(self.global_model)
                for param in mu.parameters():
                    param.data.zero_()

                coef = torch.zeros(self.num_join_clients)
                for j, mw in enumerate(self.uploaded_models):
                    if c.id != self.uploaded_ids[j]:
                        weights_i = torch.cat([p.data.view(-1) for p in c.model.parameters()], dim=0)
                        weights_j = torch.cat([p.data.view(-1) for p in mw.parameters()], dim=0)
                        sub = (weights_i - weights_j).view(-1)
                        sub = torch.dot(sub, sub)
                        coef[j] = self.alphaK * self.e(sub)
                    else:
                        coef[j] = 0
                coef_self = 1 - torch.sum(coef)
                # print(i, coef)

                for j, mw in enumerate(self.uploaded_models):
                    for param, param_j in zip(mu.parameters(), mw.parameters()):
                        param.data += coef[j] * param_j

                start_time = time.time()

                if c.send_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                c.set_parameters(mu, coef_self)

                c.send_time_cost['num_rounds'] += 1
                c.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def e(self, x):
        return math.exp(-x/self.sigma)/self.sigma
