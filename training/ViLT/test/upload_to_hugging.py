from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import requests
import json
import requests
from PIL import Image


config = json.load(open('config_medium_plus.json'))
configuration = ViltConfig(**config)
model =ViltForQuestionAnswering.from_pretrained("/scratch/gpfs/jmonas/ViLT/Models/l6_h512_i1024/vilt-saved-model-ft-93-0", config=configuration, use_safetensors=True)

model.push_to_hub("ViLT-33M-vqa")