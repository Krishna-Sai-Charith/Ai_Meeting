# ===============================
# ORIGINAL GOOGLE COLAB VERSION
# ===============================

# !pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# !pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai

# import os
# import requests
# from IPython.display import Markdown, display, update_display
# from openai import OpenAI
# from google.colab import drive
# from huggingface_hub import login
# from google.colab import userdata
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
# import torch

# AUDIO_MODEL = "whisper-1"
# LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# audio_filename = "/content/denver_extract.mp3"

# hf_token = userdata.get('HF_TOKEN')
# login(hf_token, add_to_git_credential=True)

# openai_api_key = userdata.get('OPENAI_API_KEY')
# openai = OpenAI(api_key=openai_api_key)

# audio_file = open(audio_filename, "rb")
# transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
# print(transcription)

# system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
# user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

# messages = [
#     {"role": "system", "content": system_message},
#     {"role": "user", "content": user_prompt}
# ]
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4"
# )

# tokenizer = AutoTokenizer.from_pretrained(LLAMA)
# tokenizer.pad_token = tokenizer.eos_token
# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
# streamer = TextStreamer(tokenizer)
# model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
# outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)

# response = tokenizer.decode(outputs[0])
# display(Markdown(response))


# ===============================
# VS CODE / LOCAL PYTHON VERSION
# ===============================

# 1️⃣ Install dependencies locally (run these in terminal or add to requirements.txt):
# pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# pip install requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai huggingface_hub

import os
import requests
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch

# Constants
AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Set your local audio file path
audio_filename = "denver_extract.mp3"  # Place file in same folder as script

# Load environment variables or paste tokens directly
# (Better: store them in a .env file and use python-dotenv to load)
hf_token = os.getenv("HF_TOKEN", "<YOUR_HF_TOKEN>")
openai_api_key = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_KEY>")

# HuggingFace login
login(hf_token, add_to_git_credential=True)

# OpenAI client
openai = OpenAI(api_key=openai_api_key)

# Transcribe audio
with open(audio_filename, "rb") as audio_file:
    transcription = openai.audio.transcriptions.create(
        model=AUDIO_MODEL,
        file=audio_file,
        response_format="text"
    )
print(transcription)

# Prompt setup
system_message = (
    "You are an assistant that produces minutes of meetings from transcripts, "
    "with summary, key discussion points, takeaways and action items with owners, in markdown."
)
user_prompt = (
    f"Below is an extract transcript of a Denver council meeting. "
    f"Please write minutes in markdown, including a summary with attendees, location and date; "
    f"discussion points; takeaways; and action items with owners.\n{transcription}"
)

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

# Model config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config
)

# Generate response
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
response = tokenizer.decode(outputs[0])

# Print markdown to console
print("\n=== Generated Meeting Minutes ===\n")
print(response)
