import pandas as pd
from pathlib import Path
from scipy.integrate import cumtrapz
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
import numpy as np
import matplotlib.pyplot as plt
import datetime

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

def setup_figure(num_row=1, num_col=1, width=5, height=4, left=0.125, right=0.9, hspace=0.2, wspace=0.2):

    fig, axes = plt.subplots(num_row, num_col, figsize=(width, height), squeeze=False)   
    fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
    return (fig, axes)

class LiqDataFormatter:
    def __init__(self, data_path, start_row_index=25) -> None:
        
        self.data_path = Path(data_path).resolve()
        self.result_path =self.data_path.parent / "result"
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        self.data_liq = pd.read_csv(self.data_path, sep="\t")
        self.data_liq = self.data_liq.iloc[start_row_index:, :]
        self.data_liq.reset_index(drop=True, inplace=True)
        self.data_liq = self.data_liq.iloc[:(len(self.data_liq)//100)*100+1, :]
        self.data_liq.reset_index(drop=True, inplace=True)
        
        # eaの絶対値が初めて0.075以上になったインデックスを取得し、それ以降のデータを削除
        temp_index = self.data_liq[self.data_liq["e(a)_(%)_"].abs() >= 7.5].index[0]
        self.data_liq = self.data_liq.iloc[:temp_index, :]
        
        # change unit
        self.data_liq["p_pa"] = self.data_liq["p'___(kPa)"] * 1000
        self.data_liq["q_pa"] = self.data_liq["q____(kPa)"] * 1000
        self.data_liq["ea_nodim"] = self.data_liq["e(a)_(%)_"] / 100    
    
        # Calcurate new columns
        self.data_liq["tau"] = self.data_liq["q_pa"] / 2
        self.data_liq["gamma"] = self.data_liq["ea_nodim"] * 3 / 4        
        self.data_liq["CDE"] = cumtrapz(self.data_liq["tau"], self.data_liq["gamma"], initial=0)
        self.data_liq["q_inc"] = self.data_liq["q_pa"].diff()
        self.data_liq["ea_inc"] = self.data_liq["ea_nodim"].diff()
        self.data_liq["p_inc"] = self.data_liq["p_pa"].diff()
        
        # Shift upward in the columns with Nan
        self.data_liq["q_inc"] = self.data_liq["q_inc"].shift(-1)
        self.data_liq["ea_inc"] = self.data_liq["ea_inc"].shift(-1)
        self.data_liq["p_inc"] = self.data_liq["p_inc"].shift(-1)
                
        # Drop unused columns
        temp_col_name = ["p_pa", "q_pa", "q_inc", "ea_nodim", "CDE", "p_inc", "ea_inc"]
        self.data_liq = self.data_liq[temp_col_name]
        
        self.data_liq = self.data_liq.dropna()
        
        # Normalize data for training
        temp_stress_variable = ["p_pa", "q_pa", "q_inc", "p_inc"]
        temp_strain_variable = ["ea_nodim", "ea_inc"]
        temp_energy_variable = ["CDE"]
        
        temp_stress_scale = self.data_liq[temp_stress_variable].abs().max(axis=None)
        temp_strain_scale = self.data_liq[temp_strain_variable].abs().max(axis=None)
        
        self.data_liq_scale = self.data_liq.copy()
        self.data_liq_scale = self.data_liq_scale.drop(index=self.data_liq_scale.index[1:])
        self.data_liq_scale[temp_stress_variable] = temp_stress_scale
        self.data_liq_scale[temp_strain_variable] = temp_strain_scale
        self.data_liq_scale[temp_energy_variable] = temp_stress_scale * temp_strain_scale
        
        self.data_liq_norm = self.data_liq.copy()
        
        self.data_liq_norm[temp_stress_variable] = self.data_liq_norm[temp_stress_variable] / temp_stress_scale
        self.data_liq_norm[temp_strain_variable] = self.data_liq_norm[temp_strain_variable] / temp_strain_scale
        self.data_liq_norm[temp_energy_variable] = self.data_liq_norm[temp_energy_variable] / (temp_stress_scale * temp_strain_scale)
        
        self.data_liq.to_csv(self.result_path / "formatted_data.csv", index=False)
        self.data_liq_norm.to_csv(self.result_path / "formatted_data_norm.csv", index=False)
        self.data_liq_scale.to_csv(self.result_path / "formatted_data_scale.csv", index=False)
    
    def export_figures(self, ylim=[-25, 25]):
        
        self.ylim = ylim
        
        fig, axes = setup_figure(num_row=2, num_col=1, height=8)
        
        axes[0, 0].plot(self.data_liq["p_pa"], self.data_liq["q_pa"], color="k", linewidth=1)
        axes[0 ,0].set_xlabel("p' (kPa)")
        axes[1, 0].plot(self.data_liq["ea_nodim"], self.data_liq["q_pa"], color="k", linewidth=1)
        axes[1 ,0].set_xlabel("e(a) (%)")
        
        for i in range(2):
            axes[i ,0].set_ylabel("q (kPa)")
            axes[i, 0].set_ylim(self.ylim)
            axes[i, 0].spines["top"].set_linewidth(0.5)
            axes[i, 0].spines["bottom"].set_linewidth(0.5)
            axes[i, 0].spines["right"].set_linewidth(0.5)
            axes[i, 0].spines["left"].set_linewidth(0.5)
            axes[i, 0].xaxis.set_tick_params(width=0.5)
            axes[i, 0].yaxis.set_tick_params(width=0.5)
        
        fig_name = self.result_path / "Stress_Strain.png"
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported Trained Data!")
        
        plt.clf()
        plt.close()
        gc.collect()        
        
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, hidden, cell):
        output, (hidden, cell) = self.lstm(src, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        
        input = trg[:,0,:]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)
            outputs[:,t,:] = output.squeeze(1)
            input = output.squeeze(1)
        
        return outputs

class StressStrainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StressStrainLSTM, self).__init__()
        self.hidden_size = hidden_size

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        
        # Initialize cell state
        c0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 
        return out

class StressStrainRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StressStrainRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class StressStrainCNN(nn.Module):
    def __init__(self):
        super(StressStrainCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 1, 256) # Adjust the size based on your sequence length after conv and pool layers
        self.fc2 = nn.Linear(256, 2) # Output 2 values: Increment in effective mean principal stress and axial strain

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TrainLiqData:
    def __init__(self, formatted_data, model_type = "RNN", device_type = "mps") -> None:
        
        self.formatted_data_raw = formatted_data.data_liq
        self.formatted_data_norm = formatted_data.data_liq_norm
        self.formatted_data_scale = formatted_data.data_liq_scale
        self.result_path = formatted_data.result_path
        self.model_type = model_type
        
        self.input_dim = 5
        self.output_dim = 2
        self.hidden_dim = 128
        self.num_layers = 2
        self.learning_rate = 0.005
        self.num_epochs = 1500

        if device_type == "mps":
            self.device = torch.device(device_type if torch.backends.mps.is_available() else "cpu")
        elif device_type == "cuda":
            self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        
        if model_type == "RNN":
            
            self.input_dim = 5
            self.output_dim = 2
            self.hidden_dim = 128
            self.num_layers = 2
            self.learning_rate = 0.005
            self.num_epochs = 100
            self.seq_length = 3
            
            self.model = StressStrainRNN(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            if self.seq_length == 1:
                self.X_train = torch.tensor(self.formatted_data_norm.iloc[:, :5].values.reshape(-1, 1, 5), device=self.device, dtype=torch.float32)
                y_train = torch.tensor(self.formatted_data_norm.iloc[:, 5:].values, device=self.device, dtype=torch.float32)
            else:
                temp_X_train = np.array([self.formatted_data_norm.iloc[:, :5].values[i:i+self.seq_length, :] for i in range(len(self.formatted_data_norm.iloc[:, :5].values) - self.seq_length)])
                self.X_train = torch.tensor(temp_X_train, device=self.device, dtype=torch.float32)
                y_train = torch.tensor(self.formatted_data_norm.iloc[:, 5:].values[self.seq_length:], device=self.device, dtype=torch.float32)


        elif model_type == "Seq2Seq":            
            encoder = Encoder(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
            decoder = Decoder(self.output_dim, self.hidden_dim, self.num_layers).to(self.device)
            self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            self.X_train = torch.tensor(self.formatted_data_norm.iloc[:, :5].values.reshape(-1, 10, 5), device=self.device, dtype=torch.float32)
            y_train = torch.tensor(self.formatted_data_norm.iloc[:, 5:].values.reshape(-1, 10, 2), device=self.device, dtype=torch.float32)
        
        elif model_type == "LSTM":
            self.model = StressStrainLSTM(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            seq_len = 10
            temp_X_train = np.array([self.formatted_data_norm.iloc[:, :5].values[i:i+seq_len, :] for i in range(len(self.formatted_data_norm.iloc[:, :5].values) - seq_len)])
            self.X_train = torch.tensor(temp_X_train, device=self.device, dtype=torch.float32)
            
            temp_y_train = np.array([[self.formatted_data_norm.iloc[:, 5:].values[i+seq_len, :]] for i in range(len(self.formatted_data_norm.iloc[:, 5:].values) - seq_len)])
            print(temp_X_train.shape, temp_y_train.shape)
            y_train = torch.tensor(temp_y_train, device=self.device, dtype=torch.float32)
        
        elif model_type == "CNN":
            self.model = StressStrainCNN().to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            self.X_train = torch.tensor(self.formatted_data_norm.iloc[:, :5].values.reshape(-1, 5, 1), device=self.device, dtype=torch.float32)
            y_train = torch.tensor(self.formatted_data_norm.iloc[:, 5:].values.reshape(-1, 2), device=self.device, dtype=torch.float32)
            
            print(y_train.shape)
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            
            optimizer.zero_grad()
            if self.model_type in ["RNN", "LSTM", "CNN"]:
                output = self.model(self.X_train)
            elif self.model_type == "Seq2Seq":
                output = self.model(self.X_train, y_train)
                
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
                        
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
        
        model_path = self.result_path / (datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + self.model_type + "_model.pth")
        torch.save(self.model, model_path)
    
    def predict(self, q_step = 1, q_amp = 22, end_cycle = 100, is_exporting = True, ylim=[-25, 25], unit="kPa"):
        
        self.model.eval()
        
        if unit == "kPa":
            q_step = q_step * 1000
            q_amp = q_amp * 1000
            ylim = [i * 1000 for i in ylim]
        
        self.q_step = q_step
        self.q_amp = q_amp
        self.end_cycle = end_cycle
        self.ylim = ylim
        self.is_exporting = is_exporting
        
        temp_cycle = 0
        temp_load_forward = True
        
        if self.model_type == "RNN" and self.seq_length > 1:
            temp_q = self.formatted_data_raw["q_pa"].iloc[:self.seq_length].values
            temp_p = self.formatted_data_raw["p_pa"].iloc[:self.seq_length].values
            temp_ea = self.formatted_data_raw["ea_nodim"].iloc[:self.seq_length].values
            temp_CDE = self.formatted_data_raw["CDE"].iloc[:self.seq_length].values
            if temp_load_forward:
                temp_q_inc = np.full(self.seq_length, self.q_step)
            else:
                temp_q_inc = np.full(self.seq_length, -self.q_step)
            
        else:
            temp_q = [self.formatted_data_raw["q_pa"].iloc[0]]
            temp_p = [self.formatted_data_raw["p_pa"].iloc[0]]
            temp_ea = [self.formatted_data_raw["ea_nodim"].iloc[0]]
            temp_CDE = [self.formatted_data_raw["CDE"].iloc[0]]
            if temp_load_forward:
                temp_q_inc = [self.q_step]
            else:
                temp_q_inc = [-self.q_step]
        
        temp_result = np.zeros((0, 5))
        
        while temp_cycle < self.end_cycle:
            temp_input = np.array([temp_p, temp_q, temp_q_inc, temp_ea, temp_CDE]).reshape(-1, 1, 5)

            temp_input = temp_input / self.formatted_data_scale.values[:, :5]
            
            if self.model_type in ["LSTM", "Seq2Seq"]:
                temp_data = torch.tensor(temp_input.reshape(-1, 1, 5), device=self.device, dtype=torch.float32)
            
            elif self.model_type == "RNN":
                temp_data = torch.tensor(temp_input, device=self.device, dtype=torch.float32)
            
            elif self.model_type == "CNN":
                temp_data = torch.tensor(temp_input.reshape(-1, 5, 1), device=self.device, dtype=torch.float32)
            
            with torch.no_grad():
                if self.model_type == "Seq2Seq":
                    output = self.model(temp_data, temp_data)
                else:
                    output = self.model(temp_data)
            
            output = output.squeeze(0).squeeze(0)
            output = output.cpu().detach().numpy()
            
            output = output[-1] * self.formatted_data_scale.values.flatten()[5:]
            
            
            if self.seq_length > 1:
                temp_q_inc[:-1] = temp_q_inc[1:]
                temp_q[:-1] = temp_q[1:]
                temp_p[:-1] = temp_p[1:]
                temp_ea[:-1] = temp_ea[1:]
                temp_CDE[:-1] = temp_CDE[1:]
            if temp_load_forward:
                temp_q_inc[-1] = self.q_step
            else:
                temp_q_inc[-1] = -self.q_step
            temp_q[-1] += float(temp_q_inc[-1])
            temp_p[-1] += output[0]
            temp_ea[-1] += output[1]
            
            temp_CDE[-1]+= (temp_q[-1] - temp_q_inc[-1] / 2)  * output[1] * 3 / 4
                        
            temp_result = np.vstack((temp_result, np.array([temp_q[-1], temp_p[-1], temp_ea[-1], temp_CDE[-1], temp_cycle])))
            
            if temp_load_forward and temp_q[-1] >= self.q_amp:
                temp_load_forward = False
                temp_cycle += 0.5
                print(temp_q[-1], temp_p[-1], temp_ea[-1], temp_CDE[-1], temp_cycle)
                
            elif not temp_load_forward and temp_q[-1] <= -self.q_amp:
                temp_load_forward = True
                temp_cycle += 0.5
                print(temp_q[-1], temp_p[-1], temp_ea[-1], temp_CDE[-1], temp_cycle)
                
        
        
            
        if self.is_exporting:
            
            fig, axes = setup_figure(num_row=5, num_col=1, height=12)
            
            temp_x_result = np.linspace(0, 1, len(temp_result))
            temp_x_original = np.linspace(0, 1, len(self.formatted_data_raw))
            
            axes[0, 0].plot(temp_result[:, 1], temp_result[:, 0], color="k", linewidth=1, alpha=0.5)
            axes[0 ,0].plot(self.formatted_data_raw["p_pa"], self.formatted_data_raw["q_pa"], color="r", linewidth=1, alpha=0.5)
            axes[0 ,0].set_xlabel("p'")
            axes[1, 0].plot(temp_result[:, 2], temp_result[:, 0], color="k", linewidth=1)
            axes[1 ,0].plot(self.formatted_data_raw["ea_nodim"], self.formatted_data_raw["q_pa"], color="r", linewidth=1)
            axes[1 ,0].set_xlabel("e(a)")
            axes[2, 0].plot(temp_x_result, temp_result[:, 1], color="k", linewidth=1)
            axes[2, 0].plot(temp_x_original, self.formatted_data_raw["p_pa"], color="r", linewidth=1)
            axes[2 ,0].set_ylabel("p")
            axes[3, 0].plot(temp_x_result, temp_result[:, 2], color="k", linewidth=1)
            axes[3, 0].plot(temp_x_original, self.formatted_data_raw["ea_nodim"], color="r", linewidth=1)
            axes[3 ,0].set_ylabel("e(a)")
            axes[4, 0].plot(temp_x_result, temp_result[:, 3], color="k", linewidth=1)
            axes[4, 0].plot(temp_x_original, self.formatted_data_raw["CDE"], color="r", linewidth=1)
            axes[4 ,0].set_ylabel("CDE")
            
            
            for i in range(5):
                axes[i, 0].spines["top"].set_linewidth(0.5)
                axes[i, 0].spines["bottom"].set_linewidth(0.5)
                axes[i, 0].spines["right"].set_linewidth(0.5)
                axes[i, 0].spines["left"].set_linewidth(0.5)
                axes[i, 0].xaxis.set_tick_params(width=0.5)
                axes[i, 0].yaxis.set_tick_params(width=0.5)
                axes[i, 0].grid(True, which="both", linestyle="--", linewidth=0.25)
            
            plt.show()
            
            fig_name = self.result_path / (datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_predicted_stress_strain_" + self.model_type + ".png")
            
            fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported Trained Data!")
            
            plt.clf()
            plt.close()
            gc.collect()  
        

        
