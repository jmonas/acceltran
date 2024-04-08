from transformers import FlavaProcessor
import requests
from PIL import Image

FlavaProcessor.from_pretrained("facebook/flava-full", cache_dir="/scratch/gpfs/jmonas")