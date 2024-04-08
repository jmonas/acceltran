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
import os
from os import listdir
from os.path import isfile, join
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import datetime
import sys 
sys.path.insert(0, '../../VQA/PythonEvaluationTools')
from vqaEvalDemo import get_accuracy

def eval (config_file, questions_file, images_dir, batch_size = 32, VALIDATE=False, annFile = None, percentage = 1):
    config = json.load(open(config_file))
    size = f"l{config['num_hidden_layers']}_h{config['hidden_size']}_i{config['intermediate_size']}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configuration = ViltConfig(**config)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=cache_dir)
    model =ViltForQuestionAnswering.from_pretrained(model_location, config=configuration, use_safetensors=True, cache_dir=cache_dir)


    # Opening JSON file
    f = open(questions_file)

    # Return JSON object as dictionary
    questions = json.load(f)['questions']
    questions  = questions[:round(len(questions)*percentage)]

    filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

    # source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
    def id_from_filename(filename: str) -> Optional[int]:
        match = filename_re.fullmatch(filename)
        if match is None:
            return None
        return int(match.group(1))
    
    file_names = [f for f in tqdm(listdir(images_dir)) if isfile(join(images_dir, f))]

    filename_to_id = {images_dir + "/" + file: id_from_filename(file) for file in file_names}
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

    test_dataset = VQADataset(questions=questions,
                            processor=processor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    model.eval()  # Set the model to evaluation mode
    model.to(device)

    predictions = []
    print("start test")
    print(f"TOTAL BATCHES: {len(test_dataloader)}")
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

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    predictions_file = f'val_vqa_predictions_{size}_{current_date}.json' if VALIDATE else f'test_vqa_predictions_{size}_{current_date}.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
    
    if annFile:
        pwd = os.getcwd()
        predictions_with_path = os.path.join(pwd, predictions_file)
        get_accuracy(annFile, predictions_with_path, questions_file, percentage)
    
    


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Set DELLA and TEST parameters.')
    parser.add_argument('--adroit', action='store_true', dest='adroit', help='Enable adroit parameter')
    parser.add_argument('--validate', action='store_true',  dest='validate', help='Enable validate parameter')

    args = parser.parse_args()

    ADROIT = args.adroit
    VALIDATE = args.validate

    config_file = 'config_medium_plus.json'
    size = "l6_h512_i1024"
    cache_dir = "/scratch/network/jmonas" if ADROIT else "/scratch/gpfs/jmonas"  
    questions_type = "v2_OpenEnded_mscoco_val2014_questions.json" if VALIDATE else  "v2_OpenEnded_mscoco_test2015_questions.json" 
    images_type =   "val2014" if VALIDATE else "test2015"
    model_location = f"jmonas/ViLT-11M-vqa" if ADROIT else f"{cache_dir}/ViLT/Models/{size}/vilt-saved-model-ft-93-0"
    questions_file= f'{cache_dir}/VQA/{questions_type}'
    images_dir = f'{cache_dir}/VQA/{images_type}'

    if VALIDATE:
        # get validation proxy accuracy
        annFile =f'{cache_dir}/VQA/v2_mscoco_val2014_annotations.json'
        eval(config_file, questions_file, images_dir, 32, True, annFile, .01)        
    else:
        eval(config_file, questions_file, images_dir, 32,)
