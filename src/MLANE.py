# -*- coding:utf-8 -*-

from gensim.models import Word2Vec
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import networkx as nx
from src.utils import read_node_label, Classifier
from sklearn.linear_model import LogisticRegression
import numpy
import time
import os.path


class MLANE:

    def __init__(self, graph, graph_label_path, walk_length, num_walks, embed_size, window_size, train_percent):

        self.G = graph
        self.graph_label_path = graph_label_path
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embed_size = embed_size
        self.window_size = window_size
        self.train_percent = train_percent
        self.policy = MLPPolicy(input_dim=nx.number_of_nodes(self.G)+1, hidden_dims=[10,5])
        self.shortest_dist = dict(nx.all_pairs_shortest_path_length(self.G, walk_length+1))

        self.construct_index_node()
        self.train_test_spilt()

        self._embeddings = {}



    def construct_index_node(self):
        i = 0
        self.index_2_nodes = {}
        self.nodes_2_index = {}
        for node in nx.nodes(self.G):
            self.index_2_nodes[i] = node
            self.nodes_2_index[node] = i
            i += 1
        self.nodes_index_one_hot = self.one_hot(
            list(self.index_2_nodes.keys()),
            nx.number_of_nodes(self.G)
        )

    def train_test_spilt(self):
        X, Y = read_node_label(self.graph_label_path)
        numpy.random.seed(0)
        training_size = int(self.train_percent * len(X))
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        self.X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        self.Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]

        self.X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        self.Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]


    def one_hot(self, x, nodes_count):
        return torch.eye(nodes_count)[x, :]

    def evaluate_embeddings(self, embeddings):
        print("Training classifier using {:.2f}% nodes...".format(
            self.train_percent * 100))
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='liblinear'))
        results = clf.train_evaluate(self.X_train, self.Y_train, self.X_test, self.Y_test)
        return results

    def evaluate_embeddings_during_policy_training(self, X, Y, embeddings):
        print("Training classifier using {:.2f}% nodes...".format(
            self.train_percent * 100))
        clf = Classifier(embeddings=embeddings, clf=LogisticRegression(solver='liblinear'))
        results = clf.split_train_evaluate(X, Y, self.train_percent)
        return results

    def policy_train(self, i_episode=10, epoch=50, workers = 4, **kwargs):
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = self.embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = self.window_size
        kwargs["iter"] = 5

        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0025)
        temp_epoch = 1
        while temp_epoch<=epoch:
            t1 = time.time()
            print("epoch %d started" % (temp_epoch))
            log_probs = []
            train_rewards = []
            for i in range(i_episode):
                print(self.num_walks)
                sentences, log_prob = self.adaptive_simulate_walks(
                    num_walks=self.num_walks, walk_length=self.walk_length)
                kwargs["sentences"] = sentences
                print("Learning embedding vectors...")
                model = Word2Vec(**kwargs)
                print("Learning embedding vectors done!")
                self.w2v_model = model
                results = self.evaluate_embeddings_during_policy_training(self.X_train, self.Y_train, self.get_embeddings())
                train_reward = results['macro']
                train_rewards.append(train_reward)
                log_probs.append(log_prob)

            self.update_policy(log_probs, train_rewards)
            t2 = time.time()
            print("epoch %d using time %f" % (temp_epoch,t2-t1))
            temp_epoch += 1

    def update_policy(self, log_probs, rewards):
        rewards = Variable(torch.tensor(rewards))
        policy_loss = []
        rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss)
        print('policy loss:{}'.format(policy_loss.data.cpu()))
        policy_loss.backward()
        self.optimizer.step()


    def train(self, workers=4, iter=5, **kwargs):
        self.sentences = self.simulate_walks(
            num_walks=self.num_walks, walk_length=self.walk_length)
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = self.embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = self.window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.G.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def save_model(self, name):
        if not os.path.exists('../saved_model/'):
            os.mkdir('../saved_model/')
        torch.save(self.policy, '../saved_model/%s.model' % name)

    def load_model(self, name):
        self.policy = torch.load('../saved_model/%s.model' % name)
        self.policy.eval()

    def save_embeddings(self, name):
        if not os.path.exists('../emb/'):
            os.mkdir('../emb/')
        self.w2v_model.save('../emb/%s.emb' % name)

    def load_embeddings(self, name):
        return Word2Vec.load("../emb/%s.emb" % name).wv


    def adaptive_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        walk = [start_node]
        walk_log_probs = []
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            random.shuffle(cur_nbrs)
            cur_dist = float(self.shortest_dist[start_node][cur])

            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[0])
                else:
                    walk_probs = self.policy(
                        torch.cat(
                            (torch.tensor(self.nodes_index_one_hot[self.nodes_2_index[cur]]),
                             torch.tensor([cur_dist]))
                            , 0)
                    )
                    m = Categorical(walk_probs)
                    action = m.sample()
                    have_state = False
                    for nbr in cur_nbrs:
                        if (self.shortest_dist[cur][nbr] - cur_dist) == (int(action) - 1):
                            have_state = True
                            walk.append(nbr)
                            walk_log_probs.append(torch.unsqueeze(m.log_prob(action), 0))
                            break

                    if not have_state:
                        walk.append(cur_nbrs[0])
            else:
                break
        if len(walk_log_probs) == 0:
            return walk, False
        else:
            walk_log_probs = torch.cat(walk_log_probs, 0)
            walk_log_prob = walk_log_probs.mean()
            return walk, walk_log_prob

    def adaptive_simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        log_probs = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                node_walk, node_log_probs = self.adaptive_walk(walk_length=walk_length, start_node=node)
                walks.append(node_walk)
                if node_log_probs != False:
                    log_probs.append(torch.unsqueeze(node_log_probs, 0))
            print(len(log_probs))
        log_probs = torch.cat(log_probs, 0)
        log_prob = log_probs.mean()
        return walks,log_prob

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                node_walk, _ = self.adaptive_walk(walk_length=walk_length, start_node=node)
                walks.append(node_walk)
        return walks


class MLPPolicy(nn.Module):
    '''
    Works when input dimension is low.
    '''
    def __init__(self, input_dim, hidden_dims, activations=None, output_dim=3):
        super(MLPPolicy, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.linears = []
        for i in range(1, len(dims)):
            linear = nn.Linear(dims[i - 1], dims[i])
            setattr(self, 'linear_{}'.format(i), linear)
            self.linears.append(linear)
        if activations is not None:
            self.activations = activations
        else:
            self.activations = [nn.Sigmoid() for _ in range(len(dims)-2)] + [nn.Softmax(dim=0)]

    def forward(self, x):
        x_hat = x
        for linear, activation in zip(self.linears, self.activations):
            x_hat = activation(linear(x_hat))
        return x_hat

