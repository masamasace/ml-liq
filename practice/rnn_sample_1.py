import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 時系列データの生成（例としてsin波を使用）
def generate_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.randint(0, 100)
        end = start + seq_length
        x = np.sin(np.linspace(start, end, seq_length))
        X.append(x)
        y.append(np.sin(end))
    return np.array(X), np.array(y)

# ハイパーパラメータ
seq_length = 50
num_samples = 1000
input_size = 1
hidden_size = 50
output_size = 1
num_layers = 1
learning_rate = 0.01
num_epochs = 100

# データの生成
X, y = generate_data(seq_length, num_samples)
print(X.shape, y.shape)

X = X.reshape(-1, seq_length, input_size)
y = y.reshape(-1, output_size)

print(X.shape, y.shape)

plt.plot(X[0, :, :])
plt.plot(X[1, :, :])

plt.show()



# トレーニングデータとテストデータの分割
train_size = int(num_samples * 0.8)
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32)
X_test = torch.tensor(X[train_size:], dtype=torch.float32)
y_test = torch.tensor(y[train_size:], dtype=torch.float32)

# RNNモデルの定義
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# モデルの初期化、損失関数、最適化手法の定義
model = RNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# トレーニングループ
for epoch in range(num_epochs):
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# テストデータでの予測
model.eval()
with torch.no_grad():
    predicted = model(X_test).detach().numpy()

# 結果のプロット
plt.plot(y_test, label='Actual')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()
