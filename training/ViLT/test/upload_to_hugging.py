from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import requests
import json
import requests
from PIL import Image


config = json.load(open('config_small.json'))
configuration = ViltConfig(**config)
model =ViltForQuestionAnswering.from_pretrained("/scratch/gpfs/jmonas/ViLT/Model/vilt-saved-model-0", config=configuration, use_safetensors=True)

model.push_to_hub("ViLT-5M-vqa")