import json
import torch
from transformers import AutoTokenizer, logging, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# 로깅 레벨 설정 (경고 및 오류 메시지 표시)
logging.set_verbosity_warning()

# JSON 데이터셋 로드
def load_json_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f"Successfully loaded dataset from {file_path}")
    return data

# 데이터셋을 토크나이징 및 인코딩
def encode_dataset(dataset, tokenizer, max_length):
    # term과 definition을 결합한 텍스트를 인코딩
    inputs = tokenizer(dataset, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    print("Successfully encoded dataset")
    return inputs

# 데이터셋을 PyTorch 데이터셋으로 변환
class TermDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_dataset):
        self.input_ids = encoded_dataset['input_ids']
        self.attention_mask = encoded_dataset['attention_mask']

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

    def __len__(self):
        return len(self.input_ids)

# 토크나이징 결과를 확인하는 함수
def check_tokenization(dataset, encoded_dataset, tokenizer):
    for i in range(15,16):
        print("-" * 50)
        print(f"Original: \n{dataset[i]}")
        print("-" * 50)
        print(f"Tokenized: \n{encoded_dataset['input_ids'][i]}")
        print("-" * 50)
        print(f"Decoded: \n{tokenizer.decode(encoded_dataset['input_ids'][i])}")
        print("-" * 50)
    input("학습을 계속 진행하려면 Enter 키를 누르세요.")

# 메인 함수: 데이터 로드, 토크나이징, 인코딩 및 모델 학습
def main(train_dataset_path, valid_dataset_path, model_name, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = load_json_dataset(train_dataset_path)
    valid_dataset = load_json_dataset(valid_dataset_path)

    encoded_train_dataset = encode_dataset(train_dataset, tokenizer, max_length)
    encoded_valid_dataset = encode_dataset(valid_dataset, tokenizer, max_length)

    # 토크나이징 결과 확인
    print("Train Dataset Tokenization Result:")
    check_tokenization(train_dataset, encoded_train_dataset, tokenizer)
    
    print("Validation Dataset Tokenization Result:")
    check_tokenization(valid_dataset, encoded_valid_dataset, tokenizer)

    train_term_dataset = TermDataset(encoded_train_dataset)
    valid_term_dataset = TermDataset(encoded_valid_dataset)

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

#####################################
    # LoRA 설정 추가
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.to("cuda")
#####################################
    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # 데이터 콜레이터 정의
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 학습 인자 정의
    training_args = TrainingArguments(
        output_dir='output',  # 출력 디렉토리
        num_train_epochs=3,  # 학습 에포크 수
        per_device_train_batch_size=4,  # 장치당 학습 배치 크기
        per_device_eval_batch_size=4,  # 장치당 평가 배치 크기
        gradient_accumulation_steps=2,
        warmup_steps=500,  # 웜업 스텝 수
        weight_decay=0.01,  # 가중치 감쇠
        logging_dir='logs',  # 로그 디렉토리
        logging_steps=100,  # 로깅 간격
        eval_steps=5000,  # 평가 간격
        save_steps=5000,  # 저장 간격
        eval_strategy="steps",  # 평가 전략
        save_total_limit=3,  # 최대 저장 모델 수
        load_best_model_at_end=True,  # 학습 종료 시 최고 성능 모델 로드
        fp16=True,  # 혼합 정밀도 학습 활성화
    )

    # Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_term_dataset,
        eval_dataset=valid_term_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 파인튜닝 실행
    trainer.train()

    # 파인튜닝된 모델 저장
    output_dir = "Qwen2-1.5B-Instruct_lora"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"파인튜닝된 모델이 {output_dir}에 저장되었습니다.")

# 실행
if __name__ == '__main__':
    train_dataset_path = 'train_data_processed.json'
    valid_dataset_path = 'valid_data_processed.json'
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    max_length = 512  # 최대 길이 지정
    main(train_dataset_path, valid_dataset_path, model_name, max_length)

