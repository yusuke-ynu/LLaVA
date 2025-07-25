import requests
from PIL import Image

# train with xformwers
# Need to call this before improt transformers
# from llava.train import train_xformers
# train_xformers.replace_llama_attn_with_xformers_attn()

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-13b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {
      "role": "user",
      "content": [
        #   {"type": "text", "text": "What are these?"},
          {"type": "text", "text": "この画像には何が映っていますか？\n"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))


# CT slice

print('raw CT image')
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0061_raw_slice218.png"
raw_image = Image.open(image_file)
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "この画像の中に小腸はありますか？\n"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
#
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0066_raw_slice250.png"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
#
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0071_raw_slice280.png"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))


# point
'''
print('point CT image')
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0061_point_slice110.png"
raw_image = Image.open(image_file)
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "この画像にある白丸はどの臓器の中にありますか？\n"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# 
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0066_point_slice100.png"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
#
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0071_point_slice90.png"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
#
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0061_point_slice110_normalized.png"
raw_image = Image.open(image_file)
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
# 
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0066_point_slice100_normalized.png"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
#
image_file = "/home/ygoto/FoundationModel_02/distance_transform/result/img0071_point_slice90_normalized.png"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
'''