from transformers import pipeline, AutoProcessor
from PIL import Image    
import requests

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id)
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
url = "/home/ygoto/SpatialVQA/make_QA/IQA/img0001_slice100.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(url)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
processor = AutoProcessor.from_pretrained(model_id)
'''
conversation1 = [
    {
      "role": "user",
      "content": [
        #   {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud."},
          {"type": "text", "text": "What does the label 15 represent?"},
          {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation1, add_generation_prompt=True)

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
# >>> {"generated_text": "\nUSER: What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nASSISTANT: Lava"}
'''

conversation2 = [
    {
      "role": "user",
      "content": [
        #   {"type": "text", "text": "What is the color of sun? (1) blue (2) pink (3) green (4) yellow."},
          # {"type": "text", "text": "What is the color of sun?"},
          {"type": "text", "text": "太陽の色はなんですか?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation2, add_generation_prompt=True)

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)

conversation3 = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "この画像に肝臓はありますか?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation3, add_generation_prompt=True)

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)