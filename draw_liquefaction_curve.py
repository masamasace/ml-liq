from pathlib import Path
import torch
import numpy as np

model_path = r"result\20240331220754_CNN_model.pth"
model_path = Path(model_path).resolve()

result_path = model_path.parent

formatted_data_offset_path = result_path / "formatted_data_offset.csv"
formatted_data_scale_path = result_path / "formatted_data_scale.csv"

model = torch.load(model_path)
model.eval()

formatted_data_offset = np.loadtxt(formatted_data_offset_path, delimiter=",")[1:]
formatted_data_scale = np.loadtxt(formatted_data_scale_path, delimiter=",")[1:]

q_amp_list = np.arange(14, 40, 1)

q_step = 1
end_cycle = 100

for q_amp in q_amp_list:
    
    temp_q = 0
    temp_p = 100
    temp_ea = 0
    temp_CDE = 0
    
    temp_cycle = 0
    temp_load_forward = True
    
    if temp_load_forward:
        temp_q_inc = q_step
    else:
        temp_q_inc = -q_step
    
    temp_result = np.zeros((0, 5))
    
    while temp_p >= 5:
        
        temp_input = np.array([temp_p, temp_q, temp_q_inc, temp_ea, temp_CDE])
        
        temp_imput = (temp_input - formatted_data_offset[:5]) / formatted_data_scale[:5]
        
        temp_data = torch.tensor(temp_imput.reshape(-1, 5, 1), device="cuda", dtype=torch.float32)
        
        with torch.no_grad():
            output = model(temp_data)
        
        output = output.squeeze(0).squeeze(0)
        output = output.cpu().detach().numpy()
        
        output = output * formatted_data_scale[5:] + formatted_data_offset[5:]
        
        if temp_load_forward:
            temp_q_inc = q_step
        else:
            temp_q_inc = -q_step
        
        temp_q += temp_q_inc
        temp_p += output[0]
        temp_ea += output[1]
        temp_CDE += (temp_q - temp_q_inc / 2) * 1000 * output[1] / 100
                    
        temp_result = np.vstack((temp_result, np.array([temp_q, temp_p, temp_ea, temp_CDE, temp_cycle])))
        
        if temp_load_forward and temp_q >= q_amp:
            temp_load_forward = False
            temp_cycle += 0.5
            
        elif not temp_load_forward and temp_q <= -q_amp:
            temp_load_forward = True
            temp_cycle += 0.5
        
        # print(temp_p)
        
    print(q_amp, temp_cycle, temp_result[:, 1].min(), sep=",")
