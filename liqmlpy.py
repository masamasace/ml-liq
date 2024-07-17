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
from torch.utils.data import DataLoader, TensorDataset, random_split

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

def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class LiqDataFormatter:
    def __init__(self, data_path, start_row_index=[25],
                 file_name_col_name=["file_name"],
                 stress_col_name=["p_pa", "q_pa", "q_inc", "p_inc"],
                 strain_col_name=["ea_nodim", "ea_inc"],
                 energy_col_name=["CSW"],
                 ea_thres = 7.5,
                 ) -> None:
        """
        CSW: Cumulative Shear Work
        """
        
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
        
        # print number of data_path
        print(f"Number of data: {len(self.data_path)}")
        
        # create result folder
        self.result_path = Path(__file__).resolve().parent / "result"
        self.result_path.mkdir(parents=True, exist_ok=True)
        
        # create temp folder
        self.temp_path = Path(__file__).resolve().parent / "tmp"
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # make instance variables
        self.file_name_col_name = file_name_col_name
        self.stress_col_name = stress_col_name
        self.strain_col_name = strain_col_name
        self.energy_col_name = energy_col_name
        self.param_col_name = self.stress_col_name + self.strain_col_name + self.energy_col_name
        self.used_col_name = self.file_name_col_name + self.stress_col_name + self.strain_col_name + self.energy_col_name
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


    def export_stress_strain_figures(self, ylim=[], unit="kPa"):
        
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
    
    def export_histogram(self):
        
        temp_num_param = len(self.param_col_name)
        
        fig, axes = setup_figure(num_row=1, num_col=temp_num_param, width=5 * temp_num_param, height=4)
        
        for i in range(temp_num_param):
            
            # liq_data_each["file_name"]で各データを区別。積み上げヒストグラム(barstacked)として表示

            
            temp_hist_data = []
            
            for j in range(len(pd.unique(self.liq_data["file_name"]))):
                
                temp_liq_data = self.liq_data[self.liq_data["file_name"] == pd.unique(self.liq_data["file_name"])[j]]
                temp_hist_data.append(temp_liq_data[self.param_col_name[i]].values)
            
            axes[0, i].hist(temp_hist_data, histtype="barstacked", bins=20, label=pd.unique(self.liq_data["file_name"]))
            
            axes[0, i].set_xlabel(self.param_col_name[i])
            
            axes[0, i].spines["top"].set_linewidth(0.5)
            axes[0, i].spines["bottom"].set_linewidth(0.5)
            axes[0, i].spines["right"].set_linewidth(0.5)
            axes[0, i].spines["left"].set_linewidth(0.5)
            axes[0, i].xaxis.set_tick_params(width=0.5)
            axes[0, i].yaxis.set_tick_params(width=0.5)
            axes[0, i].grid(True, which="both", linestyle="--", linewidth=0.25)

            if i == 0:
                axes[0, i].legend()
            
        fig_name = self.result_path / "histogram.png"
        fig.savefig(fig_name, format="jpg", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported Histogram!")
        
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

# define StressStrainCNN
class StressStrainCNN(nn.Module):
    def __init__(self, seq_length):
        super(StressStrainCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=256, kernel_size=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (seq_length + 2), 64) 
        self.fc2 = nn.Linear(64, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# define train class
class TrainLiqData:
    def __init__(self, formatted_data, model_type = "RNN", device_type = "mps",
                 input_col_name = ["p_pa", "q_pa", "q_inc", "ea_nodim", "CSW"],
                 output_col_name = ["p_inc", "ea_inc"],
                 train_data_stem = ["T50-0-1", "T50-0-3", "T50-0-4"],
                 train_split_ratio = 0.8,
                 batch_div=8, num_epoch=100,
                 hidden_dim=16, num_layers=2, learning_rate=0.001,
                 seq_length=1, seed=0) -> None:
        
        # make instance variables
        self.formatted_data_raw = formatted_data.liq_data
        self.formatted_data_norm = formatted_data.liq_data_norm
        self.formatted_data_scale = formatted_data.liq_data_scale
        self.result_path = formatted_data.result_path
        torch_fix_seed(seed)
        
        self.model_type = model_type
        
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name
        self.train_data_stem = train_data_stem
        
        self.train_split_ratio = train_split_ratio
        self.batch_div = batch_div
        self.num_epoch = num_epoch
        
        self.export_figures_counter = 0
        
        # check if input_col, output_col, and train_data_stem exist in formatted_data
        if set(self.input_col_name + self.output_col_name) > set(self.formatted_data_raw.columns):
            raise ValueError("input_col and output_col must be in formatted_data")
        if set(self.train_data_stem) > set(pd.unique(self.formatted_data_raw["file_name"])):
            raise ValueError("train_data_stem must be in formatted_data")
        
        # make other instance variables
        self.input_dim = len(self.input_col_name)
        self.output_dim = len(self.output_col_name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.seq_length = seq_length

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
            temp_X_train = np.zeros((0, self.seq_length, self.input_dim))
            temp_y_train = np.zeros((0, self.output_dim))
            
            for i in range(len(self.train_data_stem)):
                temp_each_data = self.formatted_data_norm[self.formatted_data_norm["file_name"] == self.train_data_stem[i]]
                
                temp_X_train = np.vstack((temp_X_train, np.array([temp_each_data[self.input_col_name].values[j:j+self.seq_length, :] for j in range(len(temp_each_data.iloc[:, :5].values) - self.seq_length)])))
                temp_y_train = np.vstack((temp_y_train, temp_each_data[self.output_col_name].values[self.seq_length:]))
                                            
            self.X_train = torch.tensor(temp_X_train, device=self.device, dtype=torch.float32)
            y_train = torch.tensor(temp_y_train, device=self.device, dtype=torch.float32)
            
        
        elif model_type == "CNN":
            
            self.model = StressStrainCNN(self.seq_length).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # make training data
            temp_X = np.zeros((0, self.input_dim, self.seq_length))
            temp_y = np.zeros((0, self.output_dim))
            
            for i in range(len(self.train_data_stem)):
                temp_each_data = self.formatted_data_norm[self.formatted_data_norm["file_name"] == self.train_data_stem[i]]
                
                if self.seq_length == 1:
                    temp_X = np.vstack((temp_X, temp_each_data[self.input_col_name].values.reshape(-1, self.input_dim, 1)))
                    temp_y = np.vstack((temp_y, temp_each_data[self.output_col_name].values))
                
                else:
                    temp_X = np.vstack((temp_X, np.array([temp_each_data[self.input_col_name].values[j:j+self.seq_length, :] for j in range(len(temp_each_data.iloc[:, :5].values) - self.seq_length)])))
                    temp_y = np.vstack((temp_y, temp_each_data[self.output_col_name].values[self.seq_length:]))
            
            train_size = int(len(temp_X) * self.train_split_ratio)
            val_size = len(temp_X) - train_size
            
            temp_train_X, temp_val_X = random_split(temp_X, [train_size, val_size])
            temp_train_y, temp_val_y = random_split(temp_y, [train_size, val_size])
            
            X_train = torch.tensor(temp_train_X, device=self.device, dtype=torch.float32)
            X_val = torch.tensor(temp_val_X, device=self.device, dtype=torch.float32)
            y_train = torch.tensor(temp_train_y, device=self.device, dtype=torch.float32)
            y_val = torch.tensor(temp_val_y, device=self.device, dtype=torch.float32)
            
        print("X_train shape", X_train.shape, "y_train shape", y_train.shape)
        print("X_val shape", X_val.shape, "y_val shape", y_val.shape)
        
        # create dataloader
        self.batch_size = len(X_train) // self.batch_div
        self.num_iterations = self.num_epoch * len(X_train)
        
        print("batch_size", self.batch_size, "num_iterations", self.num_iterations, "num_epochs", self.num_epoch)
        
        train_loader = create_dataloader(X_train, y_train, self.batch_size)
        
        # start training
        
        epoch_result = np.zeros((0, 3))
        
        for epoch in range(self.num_epoch):
            
            # train model
            self.model.train()
            
            for i, (batch_X, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            
            # evaluate model
            self.model.eval()
            val_output = self.model(X_val)
            val_loss = criterion(val_output, y_val)
            
            print(f'Epoch [{epoch+1}/{self.num_epoch}], Loss: {loss.item():.9f}, Val Loss: {val_loss.item():.9f}')
            epoch_result = np.vstack((epoch_result, np.array([epoch+1, loss.item(), val_loss.item()])))
                        
        # save model
        self.completed_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = self.result_path / (self.completed_time + "_" + self.model_type + "_model.pth")
        torch.save(self.model, model_path)
        
        # save epoch_result
        epoch_result_path = self.result_path / (self.completed_time + "_" + self.model_type + "_epoch_result.csv")
        pd.DataFrame(epoch_result, columns=["epoch", "train_loss", "val_loss"]).to_csv(epoch_result_path, index=False)
    
    
    # method to predict
    def predict(self, inc_control="q", inc_step=1, reverse_control="q", reverse_thres=26, 
                end_cycle=100, end_p=-10, end_ea=10, stress_unit="kPa", strain_unit="percent", is_exporting = True, ylim=[-25, 25], target_data_stem="T50-0-2"):
        
        self.model.eval()
        
        # check control parameters
        if inc_control not in ["q", "ea"] or reverse_control not in ["q", "ea"]:
            raise ValueError("inc_control and reverse_control must be 'q' or 'ea'")
        else:
            if inc_control == "q" and "q_inc" not in self.input_col_name:
                raise ValueError("q_inc must be in input_col_name")
            if inc_control == "ea" and "ea_inc" not in self.input_col_name:
                raise ValueError("ea_inc must be in input_col_name")
        
        # check strees_unit and strain_unit
        if stress_unit not in ["kPa", "Pa"]:
            raise ValueError("stress_unit must be 'kPa' or 'Pa'")
        if strain_unit not in ["percent", "nodim"]:
            raise ValueError("strain_unit must be 'percent' or 'nodim'")
        
        if stress_unit == "kPa":
            if inc_control == "q":
                inc_step = inc_step * 1000
            if reverse_control == "q":
                reverse_thres = reverse_thres * 1000
            ylim = [i * 1000 for i in ylim]
            end_p = end_p * 1000
        
        if strain_unit == "percent":
            if inc_control == "ea":
                inc_step = inc_step / 100
            if reverse_control == "ea":
                reverse_thres = reverse_thres / 100
            end_ea = end_ea / 100
        
        self.inc_control = inc_control
        self.inc_step = inc_step
        self.reverse_control = reverse_control
        self.reverse_thres = reverse_thres
        self.end_cycle = end_cycle
        self.ylim = ylim
        self.is_exporting = is_exporting
        self.target_data_stem = target_data_stem
        self.end_p = end_p
        self.end_ea = end_ea
        
        temp_cycle = 0
        temp_load_forward = True
        
        if self.model_type in ["RNN", "CNN"]:
            
            temp_input_raw = np.zeros((1, self.seq_length, self.input_dim))
            temp_output_raw = np.zeros((1, self.output_dim))
            
            temp_input_raw[0, :, :] = self.formatted_data_raw[self.input_col_name].iloc[:self.seq_length].values
            temp_output_raw[0, :] = self.formatted_data_raw[self.output_col_name].iloc[self.seq_length].values
        
        temp_result = np.zeros((0, 5))
        
        temp_p_index = self.input_col_name.index("p_pa")
        temp_ea_index = self.input_col_name.index("ea_nodim")
        
        while temp_cycle < self.end_cycle and temp_input_raw[0, -1, temp_p_index] > self.end_p and np.abs(temp_input_raw[0, -1, temp_ea_index]) < self.end_ea:
            
            temp_input_norm = temp_input_raw / self.formatted_data_scale[self.input_col_name].values[:, :5]
            temp_data = torch.tensor(temp_input_norm, device=self.device, dtype=torch.float32)
            
            if self.model_type == "CNN":
                temp_data = temp_data.permute(0, 2, 1)
            
            with torch.no_grad():
                output_norm = self.model(temp_data)
            
            output_norm = output_norm.squeeze(0).squeeze(0)
            output_norm = output_norm.cpu().detach().numpy()
            
            output_raw = output_norm * self.formatted_data_scale[self.output_col_name].values.flatten()
            
            if self.seq_length > 1:
                temp_input_raw[0, :-1, :] = temp_input_raw[0, 1:, :]
                
            temp_input_raw, temp_result, temp_load_forward, temp_cycle = self._update_stress_strain(temp_input_raw, temp_result, temp_load_forward, output_raw, temp_cycle)
                
        self.result = temp_result
        
        if self.is_exporting:
            
            self._export_figures()
    
    
    def _update_stress_strain(self, temp_input_raw, temp_result, temp_load_forward, output_raw, temp_cycle):
        
        if self.inc_control == "q" and "q_inc" in self.input_col_name:
            
            temp_q_inc_index = self.input_col_name.index("q_inc")
            temp_q_index = self.input_col_name.index("q_pa")
            temp_p_index = self.input_col_name.index("p_pa")
            temp_ea_index = self.input_col_name.index("ea_nodim")
            temp_CSW_index = self.input_col_name.index("CSW")
            
            if temp_load_forward:
                temp_input_raw[0, -1, temp_q_inc_index] = self.inc_step
            else:
                temp_input_raw[0, -1, temp_q_inc_index] = -self.inc_step
            
            temp_input_raw[0, -1, temp_q_index] += temp_input_raw[0, -1, temp_q_inc_index]
            temp_input_raw[0, -1, temp_p_index] += output_raw[0]
            temp_input_raw[0, -1, temp_ea_index] += output_raw[1]
            temp_input_raw[0, -1, temp_CSW_index] += (temp_input_raw[0, -1, temp_q_index] - temp_input_raw[0, -1, temp_q_inc_index] / 2) * output_raw[1] * 3 / 4

            if temp_load_forward and temp_input_raw[0, -1, temp_q_index] >= self.reverse_thres:
                temp_load_forward = False
                temp_cycle += 0.5
            elif not temp_load_forward and temp_input_raw[0, -1, temp_q_index] <= -self.reverse_thres:
                temp_load_forward = True
                temp_cycle += 0.5
            
        elif self.inc_control == "ea" and "ea_inc" in self.input_col_name:
            
            temp_ea_inc_index = self.input_col_name.index("ea_inc")
            temp_q_index = self.input_col_name.index("q_pa")
            temp_p_index = self.input_col_name.index("p_pa")
            temp_ea_index = self.input_col_name.index("ea_nodim")
            temp_CSW_index = self.input_col_name.index("CSW")
            
            if temp_load_forward:
                temp_input_raw[0, -1, temp_ea_inc_index] = self.inc_step
            else:
                temp_input_raw[0, -1, temp_ea_inc_index] = -self.inc_step
            
            temp_input_raw[0, -1, temp_ea_index] += temp_input_raw[0, -1, temp_ea_inc_index]
            temp_input_raw[0, -1, temp_p_index] += output_raw[0]
            temp_input_raw[0, -1, temp_q_index] += output_raw[1]
            temp_input_raw[0, -1, temp_CSW_index] += temp_input_raw[0, -1, temp_q_index] - output_raw[1] / 2 * temp_input_raw[0, -1, temp_ea_inc_index] * 3 / 4    
        else:
            raise ValueError("inc_control does not match with input_col_name") 
        
        temp_result = np.vstack((temp_result, np.array([temp_input_raw[0, -1, temp_p_index], temp_input_raw[0, -1, temp_q_index], temp_input_raw[0, -1, temp_ea_index], temp_input_raw[0, -1, temp_CSW_index], temp_cycle])))
        
        if self.reverse_control == "q":
            
            if temp_load_forward and temp_input_raw[0, -1, temp_q_index] >= self.reverse_thres:
                temp_load_forward = False
                temp_cycle += 0.5
                print(temp_cycle, temp_input_raw[0, -1, temp_p_index], temp_input_raw[0, -1, temp_q_index], temp_input_raw[0, -1, temp_ea_index])
                
            elif not temp_load_forward and temp_input_raw[0, -1, temp_q_index] <= -self.reverse_thres:
                temp_load_forward = True
                temp_cycle += 0.5
                print(temp_cycle, temp_input_raw[0, -1, temp_p_index], temp_input_raw[0, -1, temp_q_index], temp_input_raw[0, -1, temp_ea_index])
                
        elif self.reverse_control == "ea":
                
            if temp_load_forward and temp_input_raw[0, -1, temp_ea_index] >= self.reverse_thres:
                temp_load_forward = False
                temp_cycle += 0.5
                print(temp_cycle, temp_input_raw[0, -1, temp_p_index], temp_input_raw[0, -1, temp_q_index], temp_input_raw[0, -1, temp_ea_index])
            elif not temp_load_forward and temp_input_raw[0, -1, temp_ea_index] <= -self.reverse_thres:
                temp_load_forward = True
                temp_cycle += 0.5
                print(temp_cycle, temp_input_raw[0, -1, temp_p_index], temp_input_raw[0, -1, temp_q_index], temp_input_raw[0, -1, temp_ea_index])
                
                
        else:
            raise ValueError("'reverse_control' is not 'q' or 'ea'")       
        
        return temp_input_raw, temp_result, temp_load_forward, temp_cycle
            
    
    def _export_figures(self):
                
        # check if target_data_stem exist in formatted_data
        if self.target_data_stem not in pd.unique(self.formatted_data_raw["file_name"]):
            raise ValueError("target_data_stem must be in formatted_data")
        else:
            temp_target_data = self.formatted_data_raw[self.formatted_data_raw["file_name"] == self.target_data_stem]
        
        fig, axes = setup_figure(num_row=1, num_col=5, width=24)
        
        temp_x_result = np.linspace(0, 1, len(self.result))
        temp_x_original = np.linspace(0, 1, len(temp_target_data) - self.seq_length)
        
        # なぜか分からないが、self.formatted_data_rawの最初のself.seq_length行分がself.resultのself.seq_length行の内容に入れ替わっている。
        # print(self.result)
        # print(self.formatted_data_raw)
        
        axes[0, 0].plot(self.result[:, 0], self.result[:, 1], color="k", linewidth=1, alpha=0.5)
        axes[0, 0].plot(temp_target_data["p_pa"].iloc[self.seq_length:], temp_target_data["q_pa"].iloc[self.seq_length:], color="r", linewidth=1, alpha=0.5)
        axes[0, 0].set_xlabel("p'")
        axes[0, 0].set_ylabel("q")
        axes[0, 1].plot(self.result[:, 2], self.result[:, 1], color="k", linewidth=1)
        axes[0, 1].plot(temp_target_data["ea_nodim"].iloc[self.seq_length:], temp_target_data["q_pa"].iloc[self.seq_length:], color="r", linewidth=1)
        axes[0, 1].set_xlabel("e(a)")
        axes[0, 1].set_ylabel("q")
        axes[0, 2].plot(temp_x_result, self.result[:, 0], color="k", linewidth=1)
        axes[0, 2].plot(temp_x_original, temp_target_data["p_pa"].iloc[self.seq_length:], color="r", linewidth=1)
        axes[0, 2].set_ylabel("p")
        axes[0, 3].plot(temp_x_result, self.result[:, 2], color="k", linewidth=1)
        axes[0, 3].plot(temp_x_original, temp_target_data["ea_nodim"].iloc[self.seq_length:], color="r", linewidth=1)
        axes[0, 3].set_ylabel("e(a)")
        axes[0, 4].plot(temp_x_result, self.result[:, 3], color="k", linewidth=1)
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
        
        fig_name = self.result_path / (self.completed_time + "_predicted_stress_strain_" + self.model_type + "_" + "{:03d}".format(self.export_figures_counter) + ".jpg")
        
        fig.savefig(fig_name, format="jpg", dpi=600, pad_inches=0.05, bbox_inches="tight")
        self.export_figures_counter += 1
        print("Exported Trained Data!")
        
        plt.clf()
        plt.close()
        gc.collect()
    
