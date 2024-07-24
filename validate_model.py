from pathlib import Path
import torch
import numpy as np
import mlliqpy

model_path = r"O:\OneDrive - nagaokaut.ac.jp\01_Research\05_Papers\22_202409_JGS EEC\02_code\result\2024-07-18_12-37-31\2024-07-18_13-26-44_CNN_model.pth"
model_path = Path(model_path).resolve()

result_path = model_path.parent
print(result_path)

model = torch.load(model_path)
model.eval()

p_pa = 100 * 10 **3
q_pa = 0
q_inc = 1 * 10 **3
ea = 0
CSW = 0

input_data = np.array([p_pa, q_pa, q_inc, ea, CSW])
input_data_tensor = torch.tensor(input_data.reshape(-1, 5, 1), device="cuda", dtype=torch.float32)

with torch.no_grad():
    output = model(input_data_tensor)
    
output = output.squeeze(0).squeeze(0)
output = output.cpu().detach().numpy()

print(output)