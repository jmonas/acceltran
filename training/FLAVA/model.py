import torch
import sys
import torch.nn as nn
from transformers import FlavaModel, FlavaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
sys.path.append("/home/jmonas/acceltran/transformers/src/transformers/models/flava/")
print(sys.path)

from modeling_dtflava import DTFlavaModel

# Define the complete model for VQA with FLAVA and the classifier head
class FlavaForVQA(FlavaPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.flava = FlavaModel(config)

        # Classifier head
        hidden_size = config.multimodal_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_labels),
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, batch):
        labels = None
        if 'labels' in batch:
            labels = batch.pop('labels')
        outputs = self.flava(**batch)
        # pooler_output = outputs.multimodal_output.pooler_output
        # logits = self.classifier(pooler_output)

        multimodal_embeddings = outputs.multimodal_embeddings
        hCLS = multimodal_embeddings[:, 0, :]
        logits = self.classifier(hCLS)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean") * labels.shape[1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.multimodal_output.hidden_states,
            # attentions=outputs.multimodal_output.attentions,
        )



class DTFlavaForVQA(FlavaPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.flava = DTFlavaModel(config)

        # Classifier head
        hidden_size = config.multimodal_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_labels),
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, batch):
        labels = None
        if 'labels' in batch:
            labels = batch.pop('labels')
        outputs = self.flava(**batch)
        pooler_output = outputs.multimodal_output.pooler_output
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean") * labels.shape[1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.multimodal_output.hidden_states,
            attentions=outputs.multimodal_output.attentions,
        )

