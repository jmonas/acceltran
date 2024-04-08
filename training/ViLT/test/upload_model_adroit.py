from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import json
import requests


config = json.load(open('config_medium_plus.json'))
configuration = ViltConfig(**config)
model =ViltForQuestionAnswering.from_pretrained("jmonas/ViLT-33M-vqa", config=configuration, use_safetensors=True, cache_dir = "/scratch/network/jmonas")