from transformers import ViltProcessor, ViltModel, ViltForQuestionAnswering, ViltConfig
import requests
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
