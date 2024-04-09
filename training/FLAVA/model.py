import torch
import torch.nn as nn
from transformers import FlavaModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Define the classifier head
class VQAClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Use dropout if needed
        self.fc2 = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define the complete model for VQA with FLAVA and the classifier head
class FlavaForVQA(nn.Module):
    def __init__(self, flava_model, num_labels):
        super().__init__()
        self.flava_model = flava_model
        self.vqa_classifier = VQAClassifierHead(flava_model.config.multimodal_config.hidden_size, flava_model.config.multimodal_config.hidden_size, num_labels)
        
    def forward(self, batch):
        labels = None
        if 'labels' in batch:
            labels = batch.pop('labels')

        outputs = self.flava_model(**batch)
        multimodal_embeddings = outputs.multimodal_embeddings
        hCLS = multimodal_embeddings[:, 0, :]
        print(f"Shape of multimodal_embeddings: {hCLS.shape}")

        logits = self.vqa_classifier(hCLS)
        
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean") * labels.shape[1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

