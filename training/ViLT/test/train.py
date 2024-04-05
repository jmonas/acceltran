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


config = json.load(open('config_medium_plus.json'))
size = f"l{config["num_hidden_layers"]}_h{config["hidden_size"]}_i{config["intermediate_size"]}"

def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0

# Opening JSON file
f = open('/scratch/gpfs/jmonas/VQA/v2_OpenEnded_mscoco_train2014_questions.json')

# Return JSON object as dictionary
questions = json.load(f)['questions']

filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

# source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
def id_from_filename(filename: str) -> Optional[int]:
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

root = '/scratch/gpfs/jmonas/VQA/train2014'
file_names = [f for f in tqdm(listdir(root)) if isfile(join(root, f))]

filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
id_to_filename = {v:k for k,v in filename_to_id.items()}


# Read annotations
f = open('/scratch/gpfs/jmonas/VQA/v2_mscoco_train2014_annotations.json')

# Return JSON object as dictionary
annotations = json.load(f)['annotations']

for annotation in tqdm(annotations):
    answers = annotation['answers']
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1
    labels = []
    scores = []
    for answer in answer_count: 
        if answer not in list(config["label2id"].keys()):
            continue
        labels.append(config["label2id"][answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    annotation['labels'] = labels
    annotation['scores'] = scores



configuration = ViltConfig(**config)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir="/scratch/gpfs/jmonas")
model =ViltForQuestionAnswering(configuration)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

torch.cuda.empty_cache()

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, processor):
        self.questions = questions
        self.annotations = annotations
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        image_path = id_to_filename[annotation['image_id']]
        with Image.open(image_path) as img:
            # Convert any image to RGB (3 channels)
            image = self.transform(img)

        text = questions['question']

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config["id2label"]))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets

        return encoding

# training_dataset = load_dataset("json", data_files="/scratch/gpfs/jmonas/IconDomainVQAData/train.jsonl", split="train[:90%]")
# valid_dataset = load_dataset("json", data_files="/scratch/gpfs/jmonas/IconDomainVQAData/train.jsonl", split="train[90%:]")
train_count = round(len(questions) * .99)
VALIDATION_SIZE = len(questions) - train_count  # Number of examples to use for validation

train_dataset = VQADataset(questions=questions[:train_count],
                           annotations=annotations[:train_count],
                           processor=processor)
valid_dataset = VQADataset(questions=questions[train_count:],
                           annotations=annotations[train_count:],
                          processor=processor)
print("Training sets: {} - Validating set: {}".format(train_count, VALIDATION_SIZE))


def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

num_epochs = 100
patience = 2
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        # attention_mask=attention_masked,
                        labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"{idx}, Loss: {loss}", flush=True)

        if (idx+1) % 5000 ==0:
            scheduler.step()

        if (idx+1) % 500==0:
            model.eval()
            eval_loss = 0
            for j, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
                input_ids = batch.pop('input_ids').to(device)
                pixel_values = batch.pop('pixel_values').to(device)
                attention_masked = batch.pop('attention_mask').to(device)
                labels = batch.pop('labels').to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                # attention_mask=attention_masked,
                                labels=labels)
                
                loss = outputs.loss
                eval_loss += loss.item()

            tracking_information.append((epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
            print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1, epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]), flush=True)
            # scheduler.step()
            if eval_loss < min_eval_loss:
                model.save_pretrained(f"Models/{size}/vilt-saved-model-{epoch}-{idx//500}", from_pt=True) 
                print(f"Saved model to Models/{size}/vilt-saved-model-{epoch}-{idx//500}")
                min_eval_loss = eval_loss
                early_stopping_hook = 0
            else:
                early_stopping_hook += 1
                if early_stopping_hook > patience:
                    break
            model.train()
    
            pickle.dump(tracking_information, open(f"Models/{size}/tracking_information-{idx//500}.pkl", "wb"))
print("The finetuning process has done!")