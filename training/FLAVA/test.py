from transformers import FlavaProcessor, FlavaImageProcessor

processor = FlavaImageProcessor.from_pretrained("facebook/flava-full")
print(processor.to_dict())