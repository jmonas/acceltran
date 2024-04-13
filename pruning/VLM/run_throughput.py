import os
import sys
import numpy as np
import math 
import json 
import sys 
sys.path.insert(0, '../../training/ViLT')
from eval import evaluate
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
import torch
sys.path.append('../../transformers_new/src/transformers')
sys.path.append("/home/jmonas/acceltran/transformers/src/transformers/models/vilt/")
print(sys.path)

from modeling_dtvilt import DTViltModel, DTViltForQuestionAnswering
from datetime import datetime
import time

USE_NON_PRUNED = False

def main (model_info, max_pruning_threshold, min_k, method = "dynatran"):
	config = model_info['config']
	model_name = model_info['model_name']
	model_class = model_info['model_class']
	model_location = model_info['model_location']
	processor = model_info['processor']
	cache_dir = model_info['cache_dir']
	size = model_info["size"]


	output_dir = f'./results/throughput/ViLT/{size}/'
	os.makedirs(output_dir, exist_ok=True)

	# Set p 0.28840788773547676, and k for 0.286% activation sparsity
	cases = [(0.01, None), (0, 16)]

	results = []
	if os.path.exists(os.path.join(output_dir, f'results_{datetime.now()}.json')):
		results = json.load(open(os.path.join(output_dir, f'results_{datetime.now()}.json')))
		
	for p, k in cases:
		config.pruning_threshold = p
		config.k = k
		temp_dir = os.path.join(output_dir, f'threshold_p{str(p)[2:]}_k{k}')
		config.sparsity_file = os.path.join(temp_dir, 'sparsity.json')
		config.save_pretrained(temp_dir)


		if os.path.exists(config.sparsity_file): os.remove(config.sparsity_file)

		model = model_class.from_pretrained(model_location, config=config, use_safetensors=True, cache_dir=cache_dir)


		print(f'Running inference with pruning threshold: {p}, and \'k\': {k}')
		result = {'pruning_threshold': p, 'k': k}

		# Make new output directory
		if p in [result['pruning_threshold'] for result in results] and k in [result['k'] for result in results]:
			print(f'Results already stored')
			continue

		# run evalutation
		ann_file = "/scratch/gpfs/jmonas/VQA/v2_mscoco_val2014_annotations.json"
		images_dir = "/scratch/gpfs/jmonas/VQA/val2014"
		questions_file  =  "/scratch/gpfs/jmonas/VQA/v2_OpenEnded_mscoco_val2014_questions.json"

		# Run evaluation on the SST-2 task or the SQuAD task
		start_time = time.time()
		metrics = evaluate(model, processor, size, questions_file, images_dir, 64, True, ann_file, .0011942861)
		end_time = time.time()
		print(metrics)

		result['throughput'] = 256 / (end_time - start_time)
		print(f'Throughput: {256 / (end_time - start_time)} seq/sec')

		results.append(result)
		json.dump(results, open(os.path.join(output_dir, f'results_{datetime.now()}.json'), 'w+'))

	return


if __name__ == '__main__':
		config_file = '/home/jmonas/acceltran/training/ViLT/config_tiny.json'
		config = json.load(open(config_file))
		size = f"l{config['num_hidden_layers']}_h{config['hidden_size']}_i{config['intermediate_size']}"
		cache_dir= "/scratch/gpfs/jmonas" 
		# model_location = f"{cache_dir}/ViLT/Models/{size}/vilt-saved-model-ft-93-0"
		model_location = "jmonas/ViLT-5M-vqa"
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		config = ViltConfig(**config)
		processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=cache_dir)
		# model = DTViltForQuestionAnswering.from_pretrained(model_location, config=configuration, use_safetensors=True, cache_dir=cache_dir)
		model_info = {
			"config": config,
			"model_name": DTViltForQuestionAnswering.__name__,
			"model_class" : DTViltForQuestionAnswering,
			"model_location": model_location,
			"processor": processor,
			"cache_dir": cache_dir,
			"size":size
		}
		main(model_info, 0, 1, "top-k")

# main(model_info, .15, None)
# main(model_info, 0, 1, "top-k")





# def throughput ():



# 	dp = .05
# 	top_k = 16

# 	# Set p and k for 30% activation sparsity
# 	cases = [(dp, None), (0, top_k)]
# 	output_dir = f'./results/throughput/{model_name}_{size}_VQA'
# 	os.makedirs(output_dir, exist_ok=True)

# 	results = []

# 	for p, k in cases:
# 		print(f'Running inference with pruning threshold: {p}, and \'k\': {k}')
# 		result = {'pruning_threshold': p, 'k': k}

# 		# Make new output directory
# 		temp_dir = os.path.join(output_dir, f'threshold_p{str(p)[2:]}_k{int(k)}_{datetime.now()}')

# 		# Load and save new config
# 		config.pruning_threshold = p
# 		config.k = k
# 		config.sparsity_file = None
# 		config.save_pretrained(temp_dir)

# 		# Load model
# 		model = model_class.from_pretrained(model_location, config=config, use_safetensors=True, cache_dir=cache_dir)

# 		# Run evaluation on the SST-2 task or the SQuAD task
# 		start_time = time.time()
# 		metrics = run_glue(training_args)
# 		end_time = time.time()
# 		print(metrics)

# 		result['throughput'] = 256 / (end_time - start_time)
# 		print(f'Throughput: {256 / (end_time - start_time)} seq/sec')

# 		results.append(result)
# 		json.dump(results, open(os.path.join(output_dir, 'results.json'), 'w+'))

# 	return