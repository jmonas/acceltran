from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import requests
import json
import requests
from PIL import Image

url = "https://www.realmenrealstyle.com/wp-content/uploads/2023/11/The-Porkpie-hat-stubble.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "?"

# size0 = "l2_h128_i512"
size1 = "l2_h256_i512"
# size2 = "l4_h256_i512" 
# size3 = "l6_h512_i1024"

# config0 = json.load(open('config_small.json'))
config1 = json.load(open('config_small2.json'))
# config2 = json.load(open('config_medium.json'))
# config3 = json.load(open('config_medium_plus.json'))

# configuration0 = ViltConfig(**config0)
configuration1 = ViltConfig(**config1)
# configuration2 = ViltConfig(**config2)
# configuration3 = ViltConfig(**config3)


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir="/scratch/gpfs/jmonas")
# model0 =ViltForQuestionAnswering.from_pretrained("/home/jmonas/acceltran/training/ViLT/test/Model/vilt-saved-model-0", config=configuration0, use_safetensors=True)
model1 =ViltForQuestionAnswering.from_pretrained("/scratch/gpfs/jmonas/ViLT/Models/{size1}/vilt-saved-model-ft-97-0", config=configuration1, use_safetensors=True)
# model2 =ViltForQuestionAnswering.from_pretrained("/scratch/gpfs/jmonas/ViLT/Models/{size2}/vilt-saved-model-ft-97-5", config=configuration2, use_safetensors=True)
# model3 =ViltForQuestionAnswering.from_pretrained("/scratch/gpfs/jmonas/ViLT/Models/{size3}/vilt-saved-model-ft-93-0", config=configuration3, use_safetensors=True)


# total_params = sum(p.numel() for p in model0.parameters())
# print(f"Number of parameters {size0}: {total_params}")
total_params = sum(p.numel() for p in model1.parameters())
print(f"Number of parameters {size1}: {total_params}")
# total_params = sum(p.numel() for p in model2.parameters())
# print(f"Number of parameters {size2}: {total_params}")
# total_params = sum(p.numel() for p in model3.parameters())
# print(f"Number of parameters {size3}: {total_params}")

model2 =ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

total_params = sum(p.numel() for p in model2.parameters())
print(f"Number of parameters base: {total_params}")
# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
# outputs = model(**encoding)
# logits = outputs.logits
# idx = logits.argmax(-1).item()
# print("Predicted answer:", model.config.id2label[idx])


