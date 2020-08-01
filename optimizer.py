import torch
import torch.nn.modules.loss
import torch.nn.functional as F

def loss_function(adj_preds, adj_labels, n_nodes):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    
    return cost