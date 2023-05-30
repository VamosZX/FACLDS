import numpy as np
import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(2022)
np.random.seed(2022)


def  dirichlet_split_noniid(train_labels, alpha, n_clients):

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
   
    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
   
    client_idcs = [[] for _ in range(n_clients)]
   
    for c, fracs in zip(class_idcs, label_distribution): 
        
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs


def pre_dirichlet(clients_num,class_num, alpha , train_data,test_data):

    train_labels = np.array(train_data['target'])

    client_idcs = dirichlet_split_noniid(train_labels, alpha, clients_num)

    return client_idcs


