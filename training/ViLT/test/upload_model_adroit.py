from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import json
import requests


config = json.load(open('config_small2.json'))
configuration = ViltConfig(**config)
model =ViltForQuestionAnswering.from_pretrained("jmonas/ViLT-11M-vqa", config=configuration, use_safetensors=True, cache_dir = "/scratch/network/jmonas")