"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
# import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
from itertools import product
import torch.nn.functional as F
import pickle


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config.recdim
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class SAGCN(BasicModel):
    def __init__(self, 
                 config, 
                 dataset:BasicDataset):
        super(SAGCN, self).__init__()
        self.args = config
        self.dataset : BasicDataset = dataset
        self.has_explicit = self.dataset.has_explicit
        self.has_implicit = self.args.has_implicit
        assert self.has_explicit or self.has_implicit, "no data to train (as least one of 'explicit' and 'implicit' is needed)"
        self.__init_weight()
        pass

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.args.recdim
        self.n_layers = self.args.layer
        self.keep_prob = self.args.dropout
        self.A_split = self.args.A_split

        if self.has_explicit:
            self.explicit_factors = self.dataset.explicit_factors
        
        pretrained_embedding = None
        if 'embedding_file' in self.args and self.args['embedding_file']:
            with open(self.args['embedding_file'], 'rb') as f:
                pretrained_embedding = pickle.load(f)
        
        self.embedding_user = torch.nn.ModuleDict()
        self.embedding_item = torch.nn.ModuleDict()
        self.emsize = self.latent_dim

        if self.has_implicit:
            self.embedding_user['implicit'] = torch.nn.Embedding(
                num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.embedding_item['implicit'] = torch.nn.Embedding(
                num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            if pretrained_embedding is not None:
                self.embedding_user['implicit'].weight.data.copy_(torch.from_numpy(pretrained_embedding['user']))
                self.embedding_item['implicit'].weight.data.copy_(torch.from_numpy(pretrained_embedding['item']))


        self.add_weight = None

        if self.has_explicit:
            for factor in self.explicit_factors:
                self.embedding_user[factor] = torch.nn.Embedding(
                    num_embeddings=self.num_users, embedding_dim=self.emsize)
                self.embedding_item[factor] = torch.nn.Embedding(
                    num_embeddings=self.num_items, embedding_dim=self.emsize)
                if pretrained_embedding is not None:
                    self.embedding_user[factor].weight.data.copy_(torch.from_numpy(pretrained_embedding['user']))
                    self.embedding_item[factor].weight.data.copy_(torch.from_numpy(pretrained_embedding['item']))
                
        if self.args.mode == 'weighted_add':
            self.add_weight = nn.Parameter(torch.ones(len(self.embedding_user)))

        if self.args.pretrain == 0 and pretrained_embedding is None:
            for embedding_user in self.embedding_user.values():
                nn.init.normal_(embedding_user.weight, std=0.1)
            for embedding_item in self.embedding_item.values():
                nn.init.normal_(embedding_item.weight, std=0.1)
            # nn.init.normal_(embedding_user.weight, std=0.1)
            # nn.init.normal_(embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution initilizer')
        elif self.args.pretrain == 0:
            pass
        else:
            raise NotImplementedError
            print('use pretarined data')

        self.f = nn.Sigmoid()

        if self.has_explicit:
            self.Implicit_Graph, self.Explicit_Graph = self.dataset.getSparseGraph()
        else:
            self.Implicit_Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.args.dropout})")

    def get_embedding(self, embedding, idx):
        if idx is None:
            weight = embedding.weight
        else:
            weight = embedding(idx)
        return weight

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        implicit_graph = None
        if self.has_implicit:
            if self.A_split:
                implicit_graph = []
                for g in self.Implicit_Graph:
                    implicit_graph.append(self.__dropout_x(g, keep_prob))
            else:
                implicit_graph = self.__dropout_x(self.Implicit_Graph, keep_prob)

        explicit_graph = None
        if self.has_explicit:
            explicit_graph = {}
            if self.A_split:
                for factor, gs in zip(self.explicit_factors, self.Explicit_Graph):
                    explicit_graph = []
                    for g in gs:
                        explicit_graph.append(self.__dropout_x(g, keep_prob))
                    explicit_graph[factor] = explicit_graph
            else:
                explicit_graph = {}
                for g in self.Explicit_Graph:
                    explicit_graph.append(self.__dropout_x(g, keep_prob))
                
        return implicit_graph, explicit_graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        if self.has_implicit:
            users_emb = self.get_embedding(self.embedding_user['implicit'], None)
            items_emb = self.get_embedding(self.embedding_item['implicit'], None)
            all_emb = torch.cat([users_emb, items_emb])
            implicit_embs = [all_emb]

        if self.has_explicit:
            explicit_all_emb = {}
            explicit_embs = {}
            for factor in self.explicit_factors:
                _users_emb = self.get_embedding(self.embedding_user[factor], None)
                _items_emb = self.get_embedding(self.embedding_item[factor], None)
                _all_emb = torch.cat([_users_emb, _items_emb])
                explicit_all_emb[factor] = _all_emb
                explicit_embs[factor] = [_all_emb]


        if self.args.dropout:
            if self.training:
                print("droping")
                im_g_droped, ex_g_droped = self.__dropout(self.keep_prob)
            else:
                if self.has_explicit:
                    ex_g_droped = self.Explicit_Graph
                if self.has_implicit:
                    im_g_droped = self.Implicit_Graph        
        else:
            if self.has_explicit:
                ex_g_droped = self.Explicit_Graph
            if self.has_implicit:
                im_g_droped = self.Implicit_Graph    
        
        for _ in range(self.n_layers):
            if self.A_split:
                # implicit
                if self.has_implicit:
                    temp_emb = []
                    for f in range(len(im_g_droped)):
                        temp_emb.append(torch.sparse.mm(im_g_droped[f], all_emb))
                    side_emb = torch.cat(temp_emb, dim=0)
                    all_emb = side_emb
                # explicit
                if self.has_explicit:
                    explicit_side_emb = {}
                    for factor in self.explicit_factors:
                        _temp_emb = []
                        for f in range(len(ex_g_droped)):
                            _temp_emb.append(torch.sparse.mm(ex_g_droped[f], explicit_all_emb[factor]))
                        _side_emb = torch.cat(_temp_emb, dim=0)
                        explicit_side_emb[factor] = _side_emb
                    explicit_all_emb = explicit_side_emb
            else:
                if self.has_implicit:
                    all_emb = torch.sparse.mm(im_g_droped, all_emb)
                if self.has_explicit:
                    for factor in self.explicit_factors:
                        explicit_all_emb[factor] = torch.sparse.mm(ex_g_droped[factor], explicit_all_emb[factor])
            if self.has_implicit:
                implicit_embs.append(all_emb)
            if self.has_explicit:
                for factor in self.explicit_factors:
                    explicit_embs[factor].append(explicit_all_emb[factor])

        if self.has_implicit:
            implicit_embs = torch.stack(implicit_embs, dim=1)
            implicit_light_out = torch.mean(implicit_embs, dim=1)
            implicit_users, implicit_items = torch.split(implicit_light_out, [self.num_users, self.num_items])

        if self.has_explicit:
            explicit_light_out = {}
            explicit_users = {}
            explicit_items = {}
            for factor in self.explicit_factors:
                explicit_embs[factor] = torch.stack(explicit_embs[factor], dim=1)
                explicit_light_out[factor] = torch.mean(explicit_embs[factor], dim=1)
                explicit_users[factor], explicit_items[factor] = torch.split(explicit_light_out[factor], [self.num_users, self.num_items])
            

        users, items = [], []
        if self.has_implicit:
            users.append(implicit_users)
            items.append(implicit_items)
        if self.has_explicit:
            for factor in self.explicit_factors:
                users.append(explicit_users[factor])
                items.append(explicit_items[factor])
        if len(users) == 1:
            user = users[0]
            item = items[0]
        else:
            # aggregate
            if self.args.mode == 'mean':
                user = torch.mean(torch.stack(users), dim=0)
                item = torch.mean(torch.stack(items), dim=0)
            elif self.args.mode == 'add':
                user = torch.sum(torch.stack(users), dim=0)
                item = torch.sum(torch.stack(items), dim=0)
            elif self.args.mode == 'concat':
                user = torch.cat(users, dim=1)
                item = torch.cat(items, dim=1)
            elif self.args.mode == 'weighted_add':
                user = torch.sum(torch.stack(users) * self.add_weight.unsqueeze(-1).unsqueeze(-1), dim=0)
                item = torch.sum(torch.stack(items) * self.add_weight.unsqueeze(-1).unsqueeze(-1), dim=0)
            else:
                raise NotImplementedError("no such mode: {}".format(self.args.mode))

        return user, item
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = []
        if self.has_implicit:
            users_emb_ego.append(self.get_embedding(self.embedding_user['implicit'], users))
        if self.has_explicit:
            for factors in self.explicit_factors:
                users_emb_ego.append(self.get_embedding(self.embedding_user[factors], users))
        if len(users_emb_ego) != 1:
            users_emb_ego = torch.stack(users_emb_ego)
        else:
            users_emb_ego = users_emb_ego[0]

        pos_emb_ego = []
        if self.has_implicit:
            pos_emb_ego.append(self.get_embedding(self.embedding_item['implicit'], pos_items))
        if self.has_explicit:
            for factors in self.explicit_factors:
                pos_emb_ego.append(self.get_embedding(self.embedding_item[factors], pos_items))
        if len(pos_emb_ego) != 1:
            pos_emb_ego = torch.stack(pos_emb_ego)
        else:
            pos_emb_ego = pos_emb_ego[0]

        neg_emb_ego = []
        if self.has_implicit:
            neg_emb_ego.append(self.get_embedding(self.embedding_item['implicit'], neg_items))
        if self.has_explicit:
            for factors in self.explicit_factors:
                neg_emb_ego.append(self.get_embedding(self.embedding_item[factors], neg_items))
        if len(neg_emb_ego) != 1:
            neg_emb_ego = torch.stack(neg_emb_ego)
        else:
            neg_emb_ego = neg_emb_ego[0]

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
