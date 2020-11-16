import pickle
import pandas as pd
import numpy as np

# se cargan los datos serializados en una variable diccionario donde se establecen los keypoints

inp = './pose-net/posenet-pytorch/cl02_cam1_s1_a4_R04_png.pickle'
with open(inp, 'rb') as data_serialized:
    dic_kp = pickle.load(data_serialized)

# se filtran los keypoints relevantes (se quita ojos y orejas -4-)
kp_frames = []
for i in range(len(dic_kp.values())):
    kp_values = list(dic_kp.values())[i][0].tolist()
    del kp_values[1:5]
    kp_frames.append(kp_values)

df_kp = pd.DataFrame(kp_frames, index=list(dic_kp.keys()))

# se crea la lista de tuplas de mag y ang ([M, A]) para cada kp de cada frame, [2 (x, y) x 13 (keypoints) x f (frames)]

T = []
for f in range(len(df_kp.index)-1):
    t = []
    for g in range(len(df_kp.values[0])):
        M = np.sqrt(np.square(df_kp.values[f+1][g][0] - df_kp.values[f][g][0]) +
                    np.square(df_kp.values[f+1][g][1] - df_kp.values[f][g][1]))

        A = np.arctan((df_kp.values[f+1][g][1] - df_kp.values[f][g][1]) /
                      (df_kp.values[f+1][g][0] - df_kp.values[f][g][0]))
        t.append([M, A])

    T.append(t)


# se serializan los datos de salida para la red LSTM

print(np.array(T).shape)
name = input.split('.')[0] + '_tupla' +'.pickle'
print(name)
filename = open("tupla.pickle", "wb")

pickle.dump(T, filename)
