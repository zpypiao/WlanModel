import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from config import PHY
import seaborn as sns
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x


df = pd.read_csv('./data/aaap2.csv')

need_code_cols = ['test_id', 'test_dur', 'loc_id', 'protocol', 'pkt_len', 'bss_id',
         'ap_name', 'ap_mac', 'ap_id', 'pd', 'ed', 'nav', 'eirp', 'sta_mac',
         'sta_id']

for col in need_code_cols:
    df[col] = df[col].astype('category').cat.codes

std = df.std()

cols = df.columns[std != 0]
df = df[cols[:11]]
corr = df.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, annot=True)
plt.show()

print(cols)
assert False
df = df[cols]

mean = df.mean()
std = df.std()

y_mean = mean.iloc[-2]
y_std = std.iloc[-2]

dff = df.copy()

df = (df - mean) / std

data = df.values

X = data[:, :-8]
y = data[:, -8:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
the_y = y_test.copy()
the_y = the_y*std[-8:].values.reshape(-1,8) + mean[-8:].values.reshape(-1,8)
y_train = y_train[:, -2]
y_test = y_test[:, -2]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)

model = Net(X_train.shape[1])

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    # if loss.item() < 1e-5:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if epoch % 1000 == 0:
        print(f'epoch:{epoch}, loss:{loss.item()}')
model.eval()

output = model(X_test)

output = output.detach().numpy().reshape(-1)

y_test = y_test * y_std + y_mean
output = output * y_std + y_mean

print(f'MSE:{np.mean((y_test - output)**2)}')
print(f'MAE:{np.mean(np.abs(y_test - output))}')


test_df = {}

test_df['seq_time_'] = output
for i, each in enumerate(df.columns[-8:]):
    test_df[each] = the_y[:,i].reshape(-1)

print(test_df)

test_df = pd.DataFrame(test_df)

test_df.to_csv('test_df.csv', index=False)


# X = torch.tensor(X, dtype=torch.float32)
# output = model(X).detach().numpy().reshape(-1)
# output = output * y_std + y_mean

# dff['seq_time_'] = output

# dff['phy'] = dff[['nss', 'mcs']].apply(lambda x: PHY[x['nss']-1][x['mcs']], axis=1)

# dff['pre'] = dff['phy'] * (1-dff['per']) * dff['seq_time_']/60



# dff['error'] = np.abs(dff['pre'] - dff['throughput'])/dff['throughput']

# dff.to_csv('ap2__.csv', index=False)