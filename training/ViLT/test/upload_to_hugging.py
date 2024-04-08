from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import requests
import json
import requests
from PIL import Image


config = json.load(open('config_small2.json'))
configuration = ViltConfig(**config)
model =ViltForQuestionAnswering.from_pretrained("/scratch/gpfs/jmonas/ViLT/Models/l2_h256_i512/vilt-saved-model-ft-97-0", config=configuration, use_safetensors=True)

model.push_to_hub("ViLT-11M-vqa")