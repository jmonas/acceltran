# from transformers import FlavaMultimodalConfig, FlavaImageConfig, FlavaTextConfig, FlavaConfig, FlavaProcessor

# # def config_maker(unilayers, hidden_size, number_heads, intermediate_size):
# multi_config = FlavaMultimodalConfig(
#     num_hidden_layers=2//2, 
#     )

# image_config = FlavaImageConfig(
#     num_hidden_layers=2,     
#     )
# text_config  = FlavaTextConfig(
#     num_hidden_layers=2, 
#     )
# configuration = FlavaConfig(
#     multimodal_config=multi_config.to_dict(),
#     image_config=image_config.to_dict(),
#     text_config=text_config.to_dict(),
#     )

from PIL import Image
import requests

from transformers import FlavaProcessor, FlavaForPreTraining

model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
  text=["a photo of a cat", "a photo of a dog"], 
  images=[image, image], 
  return_tensors="pt", 
  padding="max_length", 
  max_length=77,
  return_codebook_pixels=True,
  return_image_mask=True,
  # Other things such as mlm_labels, itm_labels can be passed here. See docs
)
inputs.bool_masked_pos.zero_()

print(inputs)