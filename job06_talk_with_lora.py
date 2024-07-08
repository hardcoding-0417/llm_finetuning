import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 모델과 토크나이저 로드
model_id = "Qwen/Qwen2-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# LoRA 가중치 로드
lora_weights_path = "./Qwen2-7B-Instruct_lora"
model = PeftModel.from_pretrained(model, lora_weights_path)

# 추론 수행
prompt = "Term: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
