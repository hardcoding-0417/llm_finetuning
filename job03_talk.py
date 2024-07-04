import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat_with_model(model_path, max_length=512, num_return_sequences=1, top_k=50, top_p=0.95, temperature=0.7, repetition_penalty=1.2, no_repeat_ngram_size=2):
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    while True:
        prompt = input("Term: ")
        if prompt.lower() in ['exit', 'quit']:
            break

        # 입력 텍스트 토크나이징
        inputs = tokenizer(prompt, return_tensors='pt')

        # 모델 예측 (응답 생성)
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,     # 샘플링 활성화
            top_k=top_k,        # Top-K 샘플링
            top_p=top_p,        # Nucleus 샘플링
            temperature=temperature,  # 온도 조절
            repetition_penalty=repetition_penalty,  # 반복 패널티
            no_repeat_ngram_size=no_repeat_ngram_size  # N-그램 반복 방지
        )

        # 토큰을 텍스트로 디코딩하여 출력
        responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        for idx, response in enumerate(responses):
            print(response)
            print("-" * 50)

if __name__ == '__main__':
    model_path = "./fine_tuned_qwen2-0.5B"  # 파인튜닝된 모델 경로
    max_length = 512
    top_k = 50
    top_p = 0.5
    temperature = 0.1
    repetition_penalty = 1.2
    no_repeat_ngram_size = 3
    num_return_sequences = 1

    chat_with_model(model_path, max_length=512, num_return_sequences=1, top_k=50, top_p=0.95, temperature=0.7, repetition_penalty=1.2, no_repeat_ngram_size=2)

