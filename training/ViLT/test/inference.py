from transformers import ViltProcessor, ViltModel, ViltForQuestionAnswering, ViltConfig
import requests
from PIL import Image
import torch
import os
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import json
import re
from typing import Optional
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import requests
from PIL import Image

url = "https://img.freepik.com/premium-photo/blue-premium-business-sedan-car-sports-configuration-white-background-3d-rendering_101266-26564.jpg?size=626&ext=jpg&ga=GA1.1.1292351815.1712188800&semt=sph"
image = Image.open(requests.get(url, stream=True).raw)
text = "What color is the car?"

config = json.load(open('config.json'))

configuration = ViltConfig(**config)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir="/scratch/gpfs/jmonas")
model =ViltForQuestionAnswering.from_pretrained("/home/jmonas/acceltran/training/ViLT/test/Model/vilt-saved-model-0", config=configuration, use_safetensors=True)

model2 =ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")


total_params = sum(p.numel() for p in model2.parameters())
print(f"Number of parameters2: {total_params}")
# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])


