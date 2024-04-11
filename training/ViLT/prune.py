import numpy as np
import torch 


model = #insert model

def apply_pruning(model, amount):
    parameters_to_prune = [(module, 'weight') for module in model.modules() if isinstance(module, torch.nn.Linear)]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


def iterative_pruning(target_sparsity = .96, pruning_iterations = 30):
    current_sparsity = 0.0
    pruning_rate = 1 - np.power(1 - target_sparsity, 1/pruning_iterations)

    for i in range(pruning_iterations):
        apply_pruning(model, pruning_rate)
        current_sparsity += pruning_rate - current_sparsity * pruning_rate
        finetune_model()
        evaluate_model()
