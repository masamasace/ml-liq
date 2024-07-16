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
import random

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

def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class LiqDataFormatter:
    def __init__(self, data_path, start_row_index=[25], seed=0,
                 file_name_col_name=["file_name"],
                 input_col_name=["p_pa", "q_pa", "q_inc", "ea_nodim", "CSW"],
                 output_col_name=["p_inc", "ea_inc"],
                 stress_col_name=["p_pa", "q_pa", "q_inc", "p_inc"],
                 strain_col_name=["ea_nodim", "ea_inc"],
                 energy_col_name=["CSW"],
                 ea_thres = 7.5,
                 ) -> None:
        """
        CSW: Cumulative Shear Work
        """
        # fix seed
        torch_fix_seed(seed)
        
        # check data_path and start_row_index
        self.data_path = data_path
        if type(self.data_path) == list and type(start_row_index) == list:
            self.data_path = [Path(i).resolve() for i in self.data_path]
            self.data_path_stem = [i.stem for i in self.data_path]
            
        else:
            raise ValueError("data_path and start_row_index must be list")
        
        # check length of data_path and start_row_index
        if len(self.data_path) != len(start_row_index):
            raise ValueError("data_path and start_row_index must be same length")
        
        # check set of input_col_name and output_col_name is equal to the sum of stress_col_name, strain_col_name, and energy_col_name
        if set(input_col_name + output_col_name) != set(stress_col_name + strain_col_name + energy_col_name):
            raise ValueError("input_col_name and output_col_name must be same as the sum of stress_col_name, strain_col_name, and energy_col_name")
        
        # print number of data_path
        print(f"Number of data: {len(self.data_path)}")
        
        # create result folder
        self.result_path = Path(__file__).resolve().parent / "result"
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        # make instance variables
        self.file_name_col_name = file_name_col_name
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        self.used_col_name = self.file_name_col_name + self.input_col_name + self.output_col_name
        self.stress_col_name = stress_col_name
        self.strain_col_name = strain_col_name
        self.energy_col_name = energy_col_name
        self.ea_thres = ea_thres
        
        # load_data 
        self.liq_data = self._load_and_process_data(self.data_path, start_row_index)
        
        # get normalizing scale
        self.liq_data_scale = self._get_scale(self.liq_data)
        
        # normalize data
        self.liq_data_norm = self._normalize_data(self.liq_data, self.liq_data_scale)
        
    
    def _load_and_process_data(self, data_path, start_row_index):
        
        liq_data = pd.read_csv(data_path[0], sep="\t")
        print("Loaded Data: ", data_path[0].stem, "Total Length: ", len(liq_data))
        
        liq_data = self._process_data(liq_data, start_row_index[0], data_path[0].stem)
        
        # remove the last row
        liq_data = liq_data.drop(liq_data.index[-1])
        
        if len(data_path) > 1:
            
            for i in range(1, len(data_path)):
                temp_liq_data = pd.read_csv(data_path[i], sep="\t")
                
                temp_liq_data = self._process_data(temp_liq_data, start_row_index[i], data_path[i].stem)
                print("Loaded Data: ", data_path[i].stem, "Total Length: ", len(temp_liq_data))
                
                liq_data = pd.concat([liq_data, temp_liq_data], axis=0)
        
        # reset index
        liq_data = liq_data.reset_index(drop=True)
        
        # drop Nan column
        liq_data = liq_data.dropna(axis=1)
        
        # drop unused columns
        liq_data = liq_data[self.used_col_name]
        
        return liq_data


    def _process_data(self, liq_data_each, start_row_index_each, data_path_stem):
        
        liq_data_each = liq_data_each.iloc[start_row_index_each:, :]
        liq_data_each = liq_data_each.drop(liq_data_each.columns[-1], axis=1)
        liq_data_each["file_name"] = data_path_stem
        
        # process data
        if liq_data_each["e(a)_(%)_"].abs().max() > self.ea_thres:
            temp_index = liq_data_each[liq_data_each["e(a)_(%)_"].abs() >= self.ea_thres].index[0]
            liq_data_each = liq_data_each.iloc[:temp_index, :]
        
        # change unit
        liq_data_each["p_pa"] = liq_data_each["p'___(kPa)"] * 1000
        liq_data_each["q_pa"] = liq_data_each["q____(kPa)"] * 1000
        liq_data_each["ea_nodim"] = liq_data_each["e(a)_(%)_"] / 100
        
        # calcurate new columns
        liq_data_each["tau"] = liq_data_each["q_pa"] / 2
        liq_data_each["gamma"] = liq_data_each["ea_nodim"] * 3 / 4
        liq_data_each["CSW"] = cumtrapz(liq_data_each["tau"], liq_data_each["gamma"], initial=0)
        liq_data_each["q_inc"] = liq_data_each["q_pa"].diff()
        liq_data_each["ea_inc"] = liq_data_each["ea_nodim"].diff()
        liq_data_each["p_inc"] = liq_data_each["p_pa"].diff()
        
        # shift upward in the columns with Nan
        liq_data_each["q_inc"] = liq_data_each["q_inc"].shift(-1)
        liq_data_each["ea_inc"] = liq_data_each["ea_inc"].shift(-1)
        liq_data_each["p_inc"] = liq_data_each["p_inc"].shift(-1)
        
        # remove the last row
        liq_data_each = liq_data_each.drop(index=liq_data_each.index[-1])
        
        return liq_data_each
        

    def _get_scale(self, liq_data):
        
        # make dataframe for scale
        temp_liq_data_scale = liq_data.copy()
        temp_liq_data_scale = temp_liq_data_scale.drop(index=temp_liq_data_scale.index[1:])
        
        temp_stress_scale = liq_data[self.stress_col_name].abs().max(axis=None)
        temp_strain_scale = liq_data[self.strain_col_name].abs().max(axis=None)
        
        temp_liq_data_scale[self.stress_col_name] = temp_stress_scale
        temp_liq_data_scale[self.strain_col_name] = temp_strain_scale
        temp_liq_data_scale[self.energy_col_name] = temp_stress_scale * temp_strain_scale
        
        liq_data_scale = temp_liq_data_scale
        
        print(liq_data_scale)
        
        return liq_data_scale        
        
        
    def _normalize_data(self, liq_data, liq_data_scale):
        
        liq_data_norm = liq_data.copy()
        
        liq_data_norm[self.stress_col_name] = liq_data_norm[self.stress_col_name] / liq_data_scale[self.stress_col_name].values
        liq_data_norm[self.strain_col_name] = liq_data_norm[self.strain_col_name] / liq_data_scale[self.strain_col_name].values
        liq_data_norm[self.energy_col_name] = liq_data_norm[self.energy_col_name] / liq_data_scale[self.energy_col_name].values
        
        return liq_data_norm


    def export_figures(self, ylim=[], unit="kPa"):
        
        # check colname 
        if set(pd.unique(self.liq_data_norm["file_name"])) != set(self.data_path_stem):
            raise ValueError("file_name is not same as data_path_stem")
        
        # check ylim 
        if len(ylim) == 0:
            temp_flag_auto_ylim = True
        elif len(ylim) != 2:
            raise ValueError("ylim must be list with 2 elements")
        else:
            if unit == "kPa":
                ylim = [i * 1000 for i in ylim]
            
            temp_flag_auto_ylim = False
        
        for i in range(len(self.data_path_stem)):
            
            # extract data
            temp_liq_data = self.liq_data[self.liq_data["file_name"] == self.data_path_stem[i]]
        
            fig, axes = setup_figure(num_row=1, num_col=2, width=10, hspace=0.4)
            
            axes[0, 0].plot(temp_liq_data["p_pa"], temp_liq_data["q_pa"], color="k", linewidth=1)
            axes[0 ,0].set_xlabel("p'(Pa)")
            axes[0, 1].plot(temp_liq_data["ea_nodim"], temp_liq_data["q_pa"], color="k", linewidth=1)
            axes[0 ,1].set_xlabel("e(a)")
            
            for j in range(2):
                axes[0 ,j].set_ylabel("q (Pa)")
                if not temp_flag_auto_ylim:
                    axes[0, j].set_ylim(ylim)
                axes[0, j].spines["top"].set_linewidth(0.5)
                axes[0, j].spines["bottom"].set_linewidth(0.5)
                axes[0, j].spines["right"].set_linewidth(0.5)
                axes[0, j].spines["left"].set_linewidth(0.5)
                axes[0, j].xaxis.set_tick_params(width=0.5)
                axes[0, j].yaxis.set_tick_params(width=0.5)
            
            
            fig_name = self.result_path / (self.data_path_stem[i] + "_stress_strain.png")
            fig.savefig(fig_name, format="jpg", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported Figure!")
            
            plt.clf()
            plt.close()
            gc.collect()        


# define RNN model
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


# define train class
class TrainLiqData:
    def __init__(self, formatted_data, model_type = "RNN", device_type = "mps",
                 input_col_name = ["p_pa", "q_pa", "q_inc", "ea_nodim", "CSW"],
                 output_col_name = ["p_inc", "ea_inc"],
                 train_data_stem = ["T50-0-1", "T50-0-3", "T50-0-4"]
                 ) -> None:
        
        # make instance variables
        self.formatted_data_raw = formatted_data.liq_data
        self.formatted_data_norm = formatted_data.liq_data_norm
        self.formatted_data_scale = formatted_data.liq_data_scale
        self.result_path = formatted_data.result_path
        
        self.model_type = model_type
        
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        self.train_data_stem = train_data_stem
        
        # check if input_col, output_col, and train_data_stem exist in formatted_data
        if set(self.input_col_name + self.output_col_name) > set(self.formatted_data_raw.columns):
            raise ValueError("input_col and output_col must be in formatted_data")
        if set(self.train_data_stem) > set(pd.unique(self.formatted_data_raw["file_name"])):
            raise ValueError("train_data_stem must be in formatted_data")
        
        # make other instance variables
        self.input_dim = len(self.input_col_name)
        self.output_dim = len(self.output_col_name)
        self.hidden_dim = 128
        self.num_layers = 2
        self.learning_rate = 0.005
        self.num_epochs = 100
        self.seq_length = 1

        # check device type
        if device_type == "mps":
            self.device = torch.device(device_type if torch.backends.mps.is_available() else "cpu")
        elif device_type == "cuda":
            self.device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        
        # check model type
        if model_type == "RNN":
            
            self.model = StressStrainRNN(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # make training data
            if self.seq_length == 1:
                self.X_train = torch.tensor(self.formatted_data_norm[self.input_col_name].values.reshape(-1, 1, 5), device=self.device, dtype=torch.float32)
                y_train = torch.tensor(self.formatted_data_norm[self.output_col_name].values, device=self.device, dtype=torch.float32)
            else:
                
                temp_X_train = np.zeros((0, self.seq_length, self.input_dim))
                temp_y_train = np.zeros((0, self.output_dim))
                
                for i in range(len(self.train_data_stem)):
                    temp_each_data = self.formatted_data_norm[self.formatted_data_norm["file_name"] == self.train_data_stem[i]]
                    
                    temp_X_train = np.vstack((temp_X_train, np.array([temp_each_data[self.input_col_name].values[j:j+self.seq_length, :] for j in range(len(temp_each_data.iloc[:, :5].values) - self.seq_length)])))
                    temp_y_train = np.vstack((temp_y_train, temp_each_data[self.output_col_name].values[self.seq_length:]))
                                             
                self.X_train = torch.tensor(temp_X_train, device=self.device, dtype=torch.float32)
                y_train = torch.tensor(temp_y_train, device=self.device, dtype=torch.float32)
            
            print("X_train shape", self.X_train.shape, "y_train shape", y_train.shape)
        
        self.model.train()
        
        # train model
        for epoch in range(self.num_epochs):
            
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
                        
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.7f}')
        
        # save model
        self.completed_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = self.result_path / (self.completed_time + "_" + self.model_type + "_model.pth")
        torch.save(self.model, model_path)
    
    def predict(self, q_step = 1, q_amp = 26, end_cycle = 50, unit="kPa", is_exporting = True, ylim=[-25, 25], target_data_stem="T50-0-2"):
        
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
        
        if self.model_type == "RNN":
            
            temp_input_raw = np.zeros((1, self.seq_length, self.input_dim))
            temp_output_raw = np.zeros((1, self.output_dim))
            
            temp_input_raw[0, :, :] = self.formatted_data_raw[self.input_col_name].iloc[:self.seq_length].values
            temp_output_raw[0, :] = self.formatted_data_raw[self.output_col_name].iloc[self.seq_length].values
        
        temp_result = np.zeros((0, 5))
        
        while temp_cycle < self.end_cycle:
            
            temp_input_norm = temp_input_raw / self.formatted_data_scale[self.input_col_name].values[:, :5]
            temp_data = torch.tensor(temp_input_norm, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                output_norm = self.model(temp_data)
            
            output_norm = output_norm.squeeze(0).squeeze(0)
            output_norm = output_norm.cpu().detach().numpy()
            
            output_raw = output_norm * self.formatted_data_scale[self.output_col_name].values.flatten()
            
            if self.seq_length > 1:
                temp_input_raw[0, :-1, :] = temp_input_raw[0, 1:, :]
            
            temp_q_inc_index = self.input_col_name.index("q_inc")
            temp_q_index = self.input_col_name.index("q_pa")
            temp_p_index = self.input_col_name.index("p_pa")
            temp_ea_index = self.input_col_name.index("ea_nodim")
            temp_CSW_index = self.input_col_name.index("CSW")
            
            if temp_load_forward:
                temp_input_raw[0, -1, temp_q_inc_index] = self.q_step
            else:
                temp_input_raw[0, -1, temp_q_inc_index] = -self.q_step
            
            temp_input_raw[0, -1, temp_q_index] += temp_input_raw[0, -1, temp_q_inc_index]
            temp_input_raw[0, -1, temp_p_index] += output_raw[0]
            temp_input_raw[0, -1, temp_ea_index] += output_raw[1]
            temp_input_raw[0, -1, temp_CSW_index] += (temp_input_raw[0, -1, temp_q_index] - temp_input_raw[0, -1, temp_q_inc_index] / 2) * output_raw[1] * 3 / 4
            
            temp_result = np.vstack((temp_result, np.array([temp_input_raw[0, -1, temp_p_index], temp_input_raw[0, -1, temp_q_index], temp_input_raw[0, -1, temp_ea_index], temp_input_raw[0, -1, temp_CSW_index], temp_cycle])))
            
            if temp_load_forward and temp_input_raw[0, -1, temp_q_index] >= self.q_amp:
                temp_load_forward = False
                temp_cycle += 0.5
                # print(temp_p[-1], temp_q[-1], temp_ea[-1], temp_CSW[-1], temp_cycle)
                
            elif not temp_load_forward and temp_input_raw[0, -1, temp_q_index] <= -self.q_amp:
                temp_load_forward = True
                temp_cycle += 0.5
                # print(temp_p[-1], temp_q[-1], temp_ea[-1], temp_CSW[-1], temp_cycle)
                
            
        if self.is_exporting:
            
            # check if target_data_stem exist in formatted_data
            if target_data_stem not in pd.unique(self.formatted_data_raw["file_name"]):
                raise ValueError("target_data_stem must be in formatted_data")
            else:
                temp_target_data = self.formatted_data_raw[self.formatted_data_raw["file_name"] == target_data_stem]
            
            fig, axes = setup_figure(num_row=1, num_col=5, width=24)
            
            temp_x_result = np.linspace(0, 1, len(temp_result))
            temp_x_original = np.linspace(0, 1, len(temp_target_data) - self.seq_length)
            
            # なぜか分からないが、self.formatted_data_rawの最初のself.seq_length行分がtemp_resultのself.seq_length行の内容に入れ替わっている。
            # print(temp_result)
            # print(self.formatted_data_raw)
            
            axes[0, 0].plot(temp_result[:, 0], temp_result[:, 1], color="k", linewidth=1, alpha=0.5)
            axes[0, 0].plot(temp_target_data["p_pa"].iloc[self.seq_length:], temp_target_data["q_pa"].iloc[self.seq_length:], color="r", linewidth=1, alpha=0.5)
            axes[0, 0].set_xlabel("p'")
            axes[0, 0].set_ylabel("q")
            axes[0, 1].plot(temp_result[:, 2], temp_result[:, 1], color="k", linewidth=1)
            axes[0, 1].plot(temp_target_data["ea_nodim"].iloc[self.seq_length:], temp_target_data["q_pa"].iloc[self.seq_length:], color="r", linewidth=1)
            axes[0, 1].set_xlabel("e(a)")
            axes[0, 1].set_ylabel("q")
            axes[0, 2].plot(temp_x_result, temp_result[:, 0], color="k", linewidth=1)
            axes[0, 2].plot(temp_x_original, temp_target_data["p_pa"].iloc[self.seq_length:], color="r", linewidth=1)
            axes[0, 2].set_ylabel("p")
            axes[0, 3].plot(temp_x_result, temp_result[:, 2], color="k", linewidth=1)
            axes[0, 3].plot(temp_x_original, temp_target_data["ea_nodim"].iloc[self.seq_length:], color="r", linewidth=1)
            axes[0, 3].set_ylabel("e(a)")
            axes[0, 4].plot(temp_x_result, temp_result[:, 3], color="k", linewidth=1)
            axes[0, 4].plot(temp_x_original, temp_target_data["CSW"].iloc[self.seq_length:], color="r", linewidth=1)
            axes[0, 4].set_ylabel("CSW")
            
            for i in range(5):
                axes[0, i].spines["top"].set_linewidth(0.5)
                axes[0, i].spines["bottom"].set_linewidth(0.5)
                axes[0, i].spines["right"].set_linewidth(0.5)
                axes[0, i].spines["left"].set_linewidth(0.5)
                axes[0, i].xaxis.set_tick_params(width=0.5)
                axes[0, i].yaxis.set_tick_params(width=0.5)
                # rotate y labels
                axes[0, i].tick_params(axis="y", rotation=90)
                axes[0, i].grid(True, which="both", linestyle="--", linewidth=0.25)
            0, 
            fig_name = self.result_path / (self.completed_time + "_predicted_stress_strain_" + self.model_type + ".jpg")
            
            fig.savefig(fig_name, format="jpg", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported Trained Data!")
            
            plt.clf()
            plt.close()
            gc.collect()  
        

        
