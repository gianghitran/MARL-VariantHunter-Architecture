# CELL 1
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
import os
import torch.nn.functional as F
import json
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')
from torch_geometric.loader import NeighborLoader
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
%matplotlib inline

# CELL 2
import gdown
url = "https://drive.google.com/file/d/1dmezgT9zQ-8ydHrXvJmxwJN-LjPtumbU/view"
# gdown.download(url, quiet=False, use_cookies=False, fuzzy=True)
gdown.download(url, quiet=False, use_cookies=False)


# CELL 3
import zipfile

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

zip_path = 'unicorn.zip'
extract_to = 'unicorn'
unzip_file(zip_path, extract_to)

# CELL 4
Train_Gnn = False
Train_Word2vec = False

# CELL 5
from pprint import pprint
import gzip
from sklearn.manifold import TSNE
import json
import copy
import os

# CELL 6
import os.path as osp
import csv
def show(str):
	print (str + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

def parse_data():
    for i in range(3):
        os.system('tar -zxvf camflow-attack-' + str(i) + '.gz.tar')
    for i in range(13):
        os.system('tar -zxvf camflow-benign-' + str(i) + '.gz.tar')

    os.system('rm error.log')
    os.system('rm parse-error-camflow-*')
    show('Start processing.')
    for i in range(25):
        show('Attack graph ' + str(i+125))
        f = open('camflow-attack.txt.'+str(i), 'r')
        fw = open('unicorn/'+str(i+125)+'.txt', 'w')
        for line in f:
                tempp = line.strip('\n').split('\t')
                temp = []
                temp.append(tempp[0])
                temp.append(tempp[2].split(':')[0])
                temp.append(tempp[1])
                temp.append(tempp[2].split(':')[1])
                temp.append(tempp[2].split(':')[2])
                temp.append(tempp[2].split(':')[3])
                fw.write(temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\t'+temp[3]+'\t'+temp[4]+'\t'+temp[5]+'\n')
        f.close()
        fw.close()
        os.system('rm camflow-attack.txt.' + str(i))

    for i in range(125):
        show('Benign graph ' + str(i))
        f = open('camflow-normal.txt.'+str(i), 'r')
        fw = open('unicorn/'+str(i)+'.txt', 'w')
        for line in f:
                tempp = line.strip('\n').split('\t')
                temp = []
                temp.append(tempp[0])
                temp.append(tempp[2].split(':')[0])
                temp.append(tempp[1])
                temp.append(tempp[2].split(':')[1])
                temp.append(tempp[2].split(':')[2])
                temp.append(tempp[2].split(':')[3])
                fw.write(temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\t'+temp[3]+'\t'+temp[4]+'\t'+temp[5]+'\n')
        f.close()
        fw.close()
        os.system('rm camflow-normal.txt.' + str(i))
    show('Done.')

# CELL 7
def prepare_graph(df):
    def process_node(node, action, node_dict, label_dict, dummies, node_type):
        node_dict.setdefault(node, []).append(action)
        label_dict[node] = dummies.get(getattr(row, node_type), -1)  

    nodes = {}
    labels = {}
    edges = []
    dummies = {
        "7998762093665332071": 0, "14709879154498484854": 1, "10991425273196493354": 2,
        "14871526952859113360": 3, "8771628573506871447": 4, "7877121489144997480": 5,
        "17841021884467483934": 6, "7895447931126725167": 7, "15125250455093594050": 8,
        "8664433583651064836": 9, "14377490526132269506": 10, "15554536683409451879": 11,
        "8204541918505434145": 12, "14356114695140920775": 13
    }

    for row in df.itertuples():
        process_node(row.actorID, row.action, nodes, labels, dummies, 'actor_type')
        process_node(row.objectID, row.action, nodes, labels, dummies, 'object')

        edges.append((row.actorID, row.objectID))

    features = [nodes[node] for node in nodes]
    feat_labels = [labels[node] for node in nodes]
    edge_index = [[], []]
    for src, dst in edges:
        src_index = list(nodes.keys()).index(src)
        dst_index = list(nodes.keys()).index(dst)
        edge_index[0].append(src_index)
        edge_index[1].append(dst_index)

    return features, feat_labels, edge_index, list(nodes.keys())


# CELL 8
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = SAGEConv(in_channel, 32, normalize=True)
        self.conv2 = SAGEConv(32, 20, normalize=True)
        self.linear = nn.Linear(in_features=20,out_features=out_channel)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.linear(x)
        return F.softmax(x, dim=1)

# CELL 9
from gensim.models.callbacks import CallbackAny2Vec
import gensim
from gensim.models import Word2Vec
from multiprocessing import Pool
from itertools import compress
from tqdm import tqdm
import time

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        model.save('trained_weights/unicorn/unicorn.model')
        self.epoch += 1

# CELL 10
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

# CELL 11
logger = EpochLogger()
saver = EpochSaver()

# CELL 12
if Train_Word2vec:
    comb_data = []
    for i in range(20):
        f = open(f"unicorn/{i}.txt")
        data = f.read().split('\n')
        data = [line.split('\t') for line in data]
        comb_data = comb_data + data

    df = pd.DataFrame (comb_data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
    df.sort_values(by='timestamp', ascending=True,inplace=True)
    df = df.dropna()
    phrases,labels,edges,mapp = prepare_graph(df)
    
    word2vec = Word2Vec(sentences=phrases, vector_size=30, window=5, min_count=1, workers=8,epochs=300,callbacks=[saver,logger])

# CELL 13
from sklearn.utils import class_weight
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

model = GCN(30,14).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# CELL 14
from collections import Counter
import math

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]

def infer(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in  w2vmodel.wv]
    
    if not word_embeddings:
        return np.zeros(20)

    output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)

encoder = PositionalEncoder(30)
w2vmodel = Word2Vec.load("trained_weights/unicorn/unicorn.model")

# CELL 15
from torch_geometric import utils

################################## Training Main Model #####################################
if Train_Gnn:
    for i in range(95):
        f = open(f"unicorn/{i}.txt")
        data = f.read().split('\n')

        data = [line.split('\t') for line in data]
        df = pd.DataFrame (data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
        df.sort_values(by='timestamp', ascending=True,inplace=True)
        df = df.dropna()
        phrases,labels,edges,mapp = prepare_graph(df)

        criterion = CrossEntropyLoss()

        nodes = [infer(x) for x in phrases]
        nodes = np.array(nodes)  

        graph = Data(x=torch.tensor(nodes,dtype=torch.float).to(device),y=torch.tensor(labels,dtype=torch.long).to(device), edge_index=torch.tensor(edges,dtype=torch.long).to(device))
        graph.n_id = torch.arange(graph.num_nodes)
        mask = torch.tensor([True]*graph.num_nodes, dtype=torch.bool)

        for m_n in range(20):
            loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=5000,input_nodes=mask)
            total_loss = 0
            for subg in loader:
                model.train()
                optimizer.zero_grad() 
                out = model(subg.x, subg.edge_index) 
                loss = criterion(out, subg.y) 
                loss.backward() 
                optimizer.step()      
                total_loss += loss.item() * subg.batch_size

            loader = NeighborLoader(graph, num_neighbors=[-1,-1], batch_size=5000,input_nodes=mask)
            for subg in loader:
              model.eval()
              out = model(subg.x, subg.edge_index)
              sorted, indices = out.sort(dim=1,descending=True)
              conf = (sorted[:,0] - sorted[:,1]) / sorted[:,0]
              conf = (conf - conf.min()) / conf.max()
              pred = indices[:,0]
              cond = (pred == subg.y)
              mask[subg.n_id[cond]] = False

            print(f'Model# {m_n}. {mask.sum().item()} nodes still misclassified \n')
            torch.save(model.state_dict(), f'trained_weights/unicorn/unicorn{m_n}.pth')

# CELL 17
for i in range(95,98):
    print(f"Graph #: {i}")
    f = open(f"unicorn/{i}.txt")
    data = f.read().split('\n')

    data = [line.split('\t') for line in data]
    df = pd.DataFrame (data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
    df.sort_values(by='timestamp', ascending=True,inplace=True)
    df = df.dropna()

    phrases,labels,edges,mapp = prepare_graph(df)

    nodes = [infer(x) for x in phrases]
    nodes = np.array(nodes)  

    graph = Data(x=torch.tensor(nodes,dtype=torch.float).to(device),y=torch.tensor(labels,dtype=torch.long).to(device), edge_index=torch.tensor(edges,dtype=torch.long).to(device))
    graph.n_id = torch.arange(graph.num_nodes)
    flag = torch.tensor([True]*graph.num_nodes, dtype=torch.bool)

    for m_n in range(20):
        model.load_state_dict(torch.load(f'trained_weights/unicorn/unicorn{m_n}.pth'))
        model.eval()
        out = model(graph.x, graph.edge_index)

        sorted, indices = out.sort(dim=1,descending=True)
        conf = (sorted[:,0] - sorted[:,1]) / sorted[:,0]
        conf = (conf - conf.min()) / conf.max()

        pred = indices[:,0]
        cond = (pred == graph.y)
        flag[graph.n_id[cond]] = torch.logical_and(flag[graph.n_id[cond]], torch.tensor([False]*len(flag[graph.n_id[cond]]), dtype=torch.bool))
            
    print(flag.sum().item(), (flag.sum().item() / len(flag))*100)

# CELL 19
thresh = 330

# CELL 20
correct_benign = 0

for i in range(100,125):
    print(f"Graph #: {i}")
    f = open(f"unicorn/{i}.txt")
    data = f.read().split('\n')

    data = [line.split('\t') for line in data]
    df = pd.DataFrame (data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
    df.sort_values(by='timestamp', ascending=True,inplace=True)
    df = df.dropna()

    phrases,labels,edges,mapp = prepare_graph(df)

    nodes = [infer(x) for x in phrases]
    nodes = np.array(nodes)  

    graph = Data(x=torch.tensor(nodes,dtype=torch.float).to(device),y=torch.tensor(labels,dtype=torch.long).to(device), edge_index=torch.tensor(edges,dtype=torch.long).to(device))
    graph.n_id = torch.arange(graph.num_nodes)
    flag = torch.tensor([True]*graph.num_nodes, dtype=torch.bool)

    for m_n in range(20):
        model.load_state_dict(torch.load(f'trained_weights/unicorn/unicorn{m_n}.pth'))
        model.eval()
        out = model(graph.x, graph.edge_index)

        sorted, indices = out.sort(dim=1,descending=True)
        conf = (sorted[:,0] - sorted[:,1]) / sorted[:,0]
        conf = (conf - conf.min()) / conf.max()

        pred = indices[:,0]
        cond = (pred == graph.y)
        flag[graph.n_id[cond]] = torch.logical_and(flag[graph.n_id[cond]], torch.tensor([False]*len(flag[graph.n_id[cond]]), dtype=torch.bool))

    if flag.sum().item() <= thresh:
        correct_benign = correct_benign + 1
            
    print(flag.sum().item(), (flag.sum().item() / len(flag))*100)

# CELL 21
correct_attack = 0

for i in range(125,150):
    print(f"Graph #: {i}")
    f = open(f"unicorn/{i}.txt")
    data = f.read().split('\n')

    data = [line.split('\t') for line in data]
    df = pd.DataFrame (data, columns = ['actorID', 'actor_type','objectID','object','action','timestamp'])
    df.sort_values(by='timestamp', ascending=True,inplace=True)
    df = df.dropna()
    
    phrases,labels,edges,mapp = prepare_graph(df)

    nodes = [infer(x) for x in phrases]
    nodes = np.array(nodes)  
    
    graph = Data(x=torch.tensor(nodes,dtype=torch.float).to(device),y=torch.tensor(labels,dtype=torch.long).to(device), edge_index=torch.tensor(edges,dtype=torch.long).to(device))
    graph.n_id = torch.arange(graph.num_nodes)
    flag = torch.tensor([True]*graph.num_nodes, dtype=torch.bool)

    for m_n in range(20):
        model.load_state_dict(torch.load(f'trained_weights/unicorn/unicorn{m_n}.pth'))
        model.eval()
        out = model(graph.x, graph.edge_index)

        sorted, indices = out.sort(dim=1,descending=True)
        conf = (sorted[:,0] - sorted[:,1]) / sorted[:,0]
        conf = (conf - conf.min()) / conf.max()

        pred = indices[:,0]
        cond = (pred == graph.y)
        flag[graph.n_id[cond]] = torch.logical_and(flag[graph.n_id[cond]], torch.tensor([False]*len(flag[graph.n_id[cond]]), dtype=torch.bool))

    if  flag.sum().item() > thresh:
        correct_attack = correct_attack + 1
   
    print(flag.sum().item(), (flag.sum().item() / len(flag))*100)

# CELL 22
TP = correct_attack
FP = 25 - correct_benign
TN = correct_benign
FN = 25 - correct_attack

FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"Number of True Positives (TP): {TP}")
print(f"Number of False Positives (FP): {FP}")
print(f"Number of False Negatives (FN): {FN}")
print(f"Number of True Negatives (TN): {TN}\n")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TPR  
print(f"Precision: {precision}")
print(f"Recall: {recall}")

fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Fscore: {fscore}\n")

