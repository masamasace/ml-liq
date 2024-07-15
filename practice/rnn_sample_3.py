import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 時系列データの生成（例としてランダムデータを使用）
def generate_data(seq_len, input_len, output_len, num_samples):
    X = []
    y = []
    for i in range(num_samples):
        t = np.linspace(i, i + seq_len, seq_len)
        x = np.array([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t), np.sin(3 * t)]).T
        target_t = t + seq_len
        target = np.array([np.sin(target_t[-1]), np.cos(target_t[-1])])
        X.append(x)
        y.append(target)
    return np.array(X), np.array(y)

# ハイパーパラメータ
seq_len = 3
input_len = 5
output_len = 2
num_samples = 1000
hidden_size = 50
num_layers = 1
learning_rate = 0.01
num_epochs = 1000

# データの生成
X, y = generate_data(seq_len, input_len, output_len, num_samples)
print(X.shape, y.shape)


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
model = RNN(input_len, hidden_size, output_len, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# トレーニングループ
for epoch in range(num_epochs):
    model.train()
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

print(predicted)

# 結果のプロット
plt.plot(y_test[:, 0], label='Actual 0')
plt.plot(predicted[:, 0], label='Predicted 0')
plt.plot(y_test[:, 1], label='Actual 1')
plt.plot(predicted[:, 1], label='Predicted 1')
plt.legend()
plt.show()
