from __future__ import division
from __future__ import print_function
import os, sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch

np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
from model import LinTrans, LogReg
from optimizer import loss_function
from utils import *
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=1, help="Number of gnn layers")
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.0011, help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
parser.add_argument('--dataset', type=str, default='wiki', help='type of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda is True:
    print('Using GPU')
    torch.cuda.manual_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def clustering(Cluster, feature, true_labels):
    f_adj = np.matmul(feature, np.transpose(feature))
    predict_labels = Cluster.fit_predict(f_adj)
    
    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)

    return db, acc, nmi, adj

def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape([-1,])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1-lower_treshold) * len(cosine))
    
    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
    
    return np.array(pos_inds), np.array(neg_inds)

def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth


def gae_for(args):
    print("Using {} dataset".format(args.dataset))
    if args.dataset == 'cora':
        n_clusters = 7
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    elif args.dataset == 'citeseer':
        n_clusters = 6
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    elif args.dataset == 'pubmed':
        n_clusters = 3
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    elif args.dataset == 'wiki':
        n_clusters = 17
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity = 'precomputed', random_state=0)
    
    adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    n_nodes, feat_dim = features.shape
    dims = [feat_dim] + args.dims
    
    layers = args.linlayers
    # Store original adjacency matrix (without diagonal entries) for later
    
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_orig = adj

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    n = adj.shape[0]

    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    
    print('Laplacian Smoothing...')
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    adj_1st = (adj + sp.eye(n)).toarray()

    adj_label = torch.FloatTensor(adj_1st)
    
    model = LinTrans(layers, dims)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_label = adj_label.reshape([-1,])

    if args.cuda:
        model.cuda()
        inx = sm_fea_s.cuda()
        adj_label = adj_label.cuda()

    pos_num = len(adj.indices)
    neg_num = n_nodes*n_nodes-pos_num

    up_eta = (args.upth_ed - args.upth_st) / (args.epochs/args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs/args.upd)

    pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num, neg_num)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    bs = min(args.bs, len(pos_inds))
    length = len(pos_inds)
    
    pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
    best_lp = 0.
    print('Start Training...')
    for epoch in tqdm(range(args.epochs)):
        
        st, ed = 0, bs
        batch_num = 0
        model.train()
        length = len(pos_inds)
        
        while ( ed <= length ):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed-st)).cuda()
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()
            optimizer.zero_grad()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            x = torch.index_select(inx, 0, xind)
            y = torch.index_select(inx, 0, yind)
            zx = model(x)
            zy = model(y)
            batch_label = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).cuda()
            batch_pred = model.dcs(zx, zy)
            loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed-st)
            
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()
            
            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs

            
        if (epoch + 1) % args.upd == 0:
            model.eval()
            mu = model(inx)
            hidden_emb = mu.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
            bs = min(args.bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
            val_auc, val_ap = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            if val_auc + val_ap >= best_lp:
                best_lp = val_auc + val_ap
                best_emb = hidden_emb
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))
            
            
        
    tqdm.write("Optimization Finished!")
    auc_score, ap_score = get_roc_score(best_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test AUC score: ' + str(auc_score))
    tqdm.write('Test AP score: ' + str(ap_score))
    

if __name__ == '__main__':
    gae_for(args)