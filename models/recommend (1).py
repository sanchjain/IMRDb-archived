import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

dataset = pd.read_csv("data/ratings.csv")
# movies = pd.read_csv("D:/assets/data/ml-latest-small/ml-latest-small/movies.csv")
movies = pd.read_csv("data/filtered_movies.csv")

pivot_table = dataset.pivot_table(index = ["userId"], columns = ["movieId"], values = "rating", fill_value=0)
table = pivot_table.to_numpy()

data = pd.merge(dataset, movies)
data = data.drop(["timestamp", "movieId"], axis=1)

# sample input data
sel_movies = data[data["userId"]==2]
input = table[50]  # for user 2
# print(sel_movies)
# input = np.random.randint(low=0, high=5, size=9724)
# REMOVE
print(input)
print(len(input))

cols = pivot_table.columns

arr = np.array(dataset, dtype='int')
nb_users = int(max(arr[:, 0]))
nb_movies = len(dataset.movieId.unique())

# class SAE(nn.Module):
#     def __init__(self, ):
#         super(SAE, self).__init__()
#         self.fc1 = nn.Linear(nb_movies, 256)
#         self.fc2 = nn.Linear(256, 16)
#         self.fc3 = nn.Linear(16, 256)
#         self.fc4 = nn.Linear(256, nb_movies)
#         self.activation = nn.Tanh()
#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.activation(self.fc3(x))
#         x = self.fc4(x)
#         return x

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 8)
        self.fc3 = nn.Linear(8, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

model = SAE()
model.load_state_dict(torch.load('sae_200.pt', map_location=torch.device('cpu')))

input = torch.FloatTensor(input)
output = model(input)
output = output.detach().numpy()
output[input != 0] = 0  # make output for movies rated 0
print(output)
print(len(output))

l = []
for i in range(10):
    j = np.argmax(output)
    output[j] = 0
    l.append(j)
print(l)
# indices

ids = []
for i in l:
    ids.append(cols[i])
print(ids)
# movie ids

names = []
for i in ids:
    value = movies.loc[movies.movieId == i].index
    value = movies.iat[value[0], 2]
    names.append(value)
print(names)