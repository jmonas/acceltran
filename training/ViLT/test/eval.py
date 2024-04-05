import json
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import requests
import json
import requests
from PIL import Image
import re
import torch
from typing import Optional
from tqdm.auto import tqdm
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms

config = json.load(open('config_small.json'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configuration = ViltConfig(**config)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir="/scratch/gpfs/jmonas")
model =ViltForQuestionAnswering.from_pretrained("/home/jmonas/acceltran/training/ViLT/test/Model/vilt-saved-model-0", config=configuration, use_safetensors=True)



# Opening JSON file
f = open('/scratch/gpfs/jmonas/VQA/v2_OpenEnded_mscoco_test-dev2015_questions.json')

# Return JSON object as dictionary
questions = json.load(f)['questions']

filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

# source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
def id_from_filename(filename: str) -> Optional[int]:
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

root = '/scratch/gpfs/jmonas/VQA/test2015'
file_names = [f for f in tqdm(listdir(root)) if isfile(join(root, f))]

filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
id_to_filename = {v:k for k,v in filename_to_id.items()}

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, processor):
        self.questions = questions
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
        ])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # get image + text
        question = self.questions[idx]
        image_path = id_to_filename[question['image_id']]
        with Image.open(image_path) as img:
            # Convert any image to RGB (3 channels)
            image = self.transform(img)

        text = question['question']

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        encoding['question_id'] = question["question_id"]
        return encoding

test_dataloader = VQADataset(questions=questions,
                           processor=processor)  

model.eval()  # Set the model to evaluation mode

predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        # Adapt these lines based on how your DataLoader and model are set up
        inputs = {'pixel_values': batch['pixel_values'].to(device), 'input_ids': batch['input_ids'].to(device)}
        outputs = model(**inputs)
        preds = outputs.logits.argmax(-1).tolist()  # Convert logits to predicted indices
        
        for item, pred in zip(batch, preds):
            # Convert `pred` to the corresponding answer string. This may involve a mapping similar to `id2label`.
            answer = config["id2label"][str(pred)]  # This is a placeholder; adapt it to your model's specifics
            
            predictions.append({'question_id': item["question_id"], 'answer': answer})

# Save predictions to a JSON file
with open('vqa_predictions.json', 'w') as f:
    json.dump(predictions, f)

resFile = 'predictions/small/vqa_predictions_0.json'