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
        
    def forward(self, input_ids, pixel_values, attention_mask, labels = None):
        # Get the multimodal encoder outputs
        outputs = self.flava_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        # Use the output corresponding to the [CLS_M] token
        hCLS_M = outputs.last_hidden_state[:, 0]  # assuming that the [CLS_M] token is the first token in the sequence
        
        # Pass the [CLS_M] representation through the classifier head
        logits = self.vqa_classifier(hCLS_M)
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) * labels.shape[1]
            # see https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

