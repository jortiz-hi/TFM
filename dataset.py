import pickle
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import (DataLoader,)
# se cargan los input de la red LSTM
# se guarda en dicts las tuplas T de cada uno de los vídeos (40 vid x ~140 fr x 13 kp x 2 M,A)
# se convierten los frames x kp x M,A en secuencias completas de tensores float (~4000)

direct = 'pose-net/posenet-pytorch/data_tupla/'
dat = {}
for n in os.scandir(direct):
    inp = n.path
    with open(inp, 'rb') as data_serialized:
        input = pickle.load(data_serialized)
        tensor = torch.from_numpy(np.array(input)).float().view(-1)
    dat.update({inp.split('/')[-1]: tensor})


# crear secuencia. input: diccionario de videos [estructura = {'video': tensor}], y tamaño de ventana
# output: lista de ventanas deslizantes a traves de los tensores de todos lo vídeos, etiquetando cada ventana con
# otro tensor que indique si el movimiento es correcto o no.


def crear_secuencia_a4(videos, sw):
    out_seq = []
    for j in videos.keys():
        n_t = len(dat[j])
        act = int(j.split('_')[3].strip('a'))
        train_label = [1.] if act == 4 else [0.]
        for k in range(n_t-sw):
            train_seq = videos[j][k:k+sw]
            out_seq.append((train_seq, torch.tensor(train_label, dtype=torch.float32)))
    return out_seq


# ejemplo de datos, target

i = 0
for idx, g in enumerate(out_data):
    for h in out_data[idx]:
        i += 1
        print('data: ', h[0])
        print('target: ', h[1], i)


'''
# elijo un número fijo de frames para luego cargar todos los datos con el mismo batch_size (con dataloader)
#batch_size = 141
#train_loader = DataLoader(dataset=out_data, batch_size=batch_size)
#for idx, (data) in enumerate(train_loader):
#    print(idx)'''