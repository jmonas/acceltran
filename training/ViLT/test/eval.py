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
from torch.utils.data import DataLoader

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
        
        encoding["question_id"] = question["question_id"]
        return encoding




def collate_fn(batch):
  input_ids = [item['input_ids'][0] for item in batch]
  pixel_values = [item['pixel_values'][0] for item in batch]
  token_type_ids = [item['token_type_ids'][0] for item in batch]
  question_ids = [item['question_id'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['question_ids'] = torch.tensor(question_ids)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']

  return batch

batch_size = 32
print(len(questions))
test_dataset = VQADataset(questions=questions,
                           processor=processor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

model.eval()  # Set the model to evaluation mode
model.to(device)

predictions = []

with torch.no_grad():
    for idx, batch in enumerate(test_dataloader):
        print(idx, flush=True)
        # Adapt these lines based on how your DataLoader and model are set up
        inputs = {'pixel_values': batch['pixel_values'].to(device), 'input_ids': batch['input_ids'].to(device)}
        outputs = model(**inputs)
        preds = outputs.logits.argmax(-1).tolist()  # Convert logits to predicted indices
        
        for question, pred in zip(batch["question_ids"], preds):
            # Convert `pred` to the corresponding answer string. This may involve a mapping similar to `id2label`.
            answer = config["id2label"][str(pred)]  # This is a placeholder; adapt it to your model's specifics
            predictions.append({'question_id': question.item(), 'answer': answer})

# Save predictions to a JSON file
with open('vqa_predictions_.json', 'w') as f:
    json.dump(predictions, f)

resFile = 'predictions/small/vqa_predictions_0.json'