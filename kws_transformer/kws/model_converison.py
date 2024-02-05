from model import ViT_Lightning
import torch

model = ViT_Lightning.load_from_checkpoint("outputs/2023-12-19/13-26-05/weights/epoch=48-step=8134.ckpt")

save_onnx = "outputs/final_weights/model.onnx"
input_sample = torch.randn((1, 1, 40, 100))
model.to_onnx(save_onnx, input_sample, export_params=True)