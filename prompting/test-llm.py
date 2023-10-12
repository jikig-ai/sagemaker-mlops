from sagemaker.huggingface import HuggingFacePredictor

# create predictor
#predictor = HuggingFacePredictor("llama2-chat-endpoint-1hbtij6p70e80a1f5b5ef9a") 
# predictor = HuggingFacePredictor("huggingface-pytorch-tgi-inference-2023-10-04-13-28-13-041") 
predictor = HuggingFacePredictor("huggingface-pytorch-tgi-inference-2023-10-12-07-37-47-034")

# run inference
#predictor.predict({"inputs": "Can you tell me something about AWS CDK?"})

# hyperparameters for llm
prompt = f"""<s>[INST] <<SYS>>
You are an AWS Expert
<</SYS>>

Should I rather use AWS CDK or Terraform? [/INST]
"""

# prompt = f"""<s>[INST] <<SYS>>

# <</SYS>>
# Please be on point and provide a succint answer only the following question:
# Which specific people gave the UN the land in NY to build their HQ? [/INST]
# """

# prompt = f"""<s>[INST] <<SYS>>

# <</SYS>>
# When was Tomoaki Komorida born? [/INST]
# """

payload = {
  "inputs": prompt,
  "parameters": {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.7,
    "top_k": 50,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.03,
    "stop": ["</s>"]
  }
}

# send request to endpoint
response = predictor.predict(payload)

print(response[0]["generated_text"])
