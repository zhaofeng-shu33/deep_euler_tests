# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, num_in_features, num_out_features, neurons_per_layer):
        super(SimpleMLP, self).__init__()
        self.act    = nn.ELU()
        self.l_in   = nn.Linear(
            in_features = num_in_features,
            out_features= neurons_per_layer
            )
        self.l1   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l2   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l3   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l4   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l5   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l6   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= neurons_per_layer
            )
        self.l_out   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= num_out_features
            )
        #weight init
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        torch.nn.init.xavier_normal_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        torch.nn.init.xavier_normal_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)
        torch.nn.init.xavier_normal_(self.l3.weight)
        torch.nn.init.zeros_(self.l3.bias)
        torch.nn.init.xavier_normal_(self.l4.weight)
        torch.nn.init.zeros_(self.l4.bias)
        torch.nn.init.xavier_normal_(self.l5.weight)
        torch.nn.init.zeros_(self.l5.bias)
        torch.nn.init.xavier_normal_(self.l6.weight)
        torch.nn.init.zeros_(self.l6.bias)
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)
        
       

    def forward(self, x):
        x   = self.act(self.l_in(x))
        x   = self.act(self.l1(x))
        x   = self.act(self.l2(x))
        x   = self.act(self.l3(x))
        x   = self.act(self.l4(x))
        x   = self.act(self.l5(x))
        x   = self.act(self.l6(x))
        x   = self.l_out(x)
        return x

class SimpleMLPGen(nn.Module):
    def __init__(self, num_in_features, num_out_features, neurons_per_layer, num_meta_in_features):
        super(SimpleMLPGen, self).__init__()
        # compute the needed parameters
        self.num_in_features = num_in_features
        self.neurons_per_layer = neurons_per_layer
        self.num_out_features = num_out_features
        self.meta_out_features = num_in_features * neurons_per_layer + neurons_per_layer +\
             neurons_per_layer * num_out_features + num_out_features
        self.act    = nn.ELU()
        self.l_in   = nn.Linear(
            in_features = num_meta_in_features,
            out_features= neurons_per_layer
            )
        self.l_out   = nn.Linear(
            in_features = neurons_per_layer,
            out_features= self.meta_out_features
            )
        #weight init
        torch.nn.init.xavier_normal_(self.l_in.weight)
        torch.nn.init.zeros_(self.l_in.bias)
        torch.nn.init.xavier_normal_(self.l_out.weight)
        torch.nn.init.zeros_(self.l_out.bias)
        
       

    def forward(self, x):
        real_features = x[:, :self.num_in_features]
        meta_features = x[:, self.num_in_features:]
        x   = self.act(self.l_in(meta_features))
        x   = self.l_out(x)
        # x is a long vector, now split it into 4 parts
        _base = self.num_in_features * self.neurons_per_layer
        l_in_weight = x[:, :_base].reshape((-1, self.num_in_features, self.neurons_per_layer))
        # now l_in_weight is a matrix
        l_in_bias = x[:, _base:_base + self.neurons_per_layer]
        _base += self.neurons_per_layer
        _base_add = self.neurons_per_layer * self.num_out_features
        l_out_weight = x[:, _base:_base + _base_add].reshape((-1, self.neurons_per_layer, self.num_out_features))
        _base += _base_add
        l_out_bias = x[:, _base:]
        y = self.act(torch.tensordot(real_features, l_in_weight, dims=([1], [1])).diagonal(dim1=0, dim2=1).t() + l_in_bias)
        y = torch.tensordot(y, l_out_weight, dims=([1], [1])).diagonal(dim1=0, dim2=1).t() + l_out_bias
        return y

if __name__ == '__main__':
    model = SimpleMLPGen(4,2,3,5)
    x = torch.rand(2,9)  
    y = model.forward(x)
