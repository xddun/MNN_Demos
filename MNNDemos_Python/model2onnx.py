import torch
import torch.nn as nn
import torch.onnx
import onnx
from mobilenet import mobilenet_v2

pt_model_path = './mobilenet_v2-b0353104.pth'
onnx_model_path = './mobilenet_v2-b0353104.onnx'
model = mobilenet_v2(pretrained=False)
model.load_state_dict(torch.load(pt_model_path, map_location=torch.device('cpu')))
input_tensor = torch.randn(1, 3, 224, 224)
input_names = ['input']
output_names = ['output']
torch.onnx.export(model, input_tensor, onnx_model_path, verbose=True, input_names=input_names, output_names=output_names)
