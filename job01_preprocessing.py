import json
import random
from collections.abc import MutableMapping

# JSON 데이터 로드
def load_json_dataset(json_string):
    data = json.loads(json_string)
    return data

# 데이터셋 가공 (텍스트 형태로 변환)
def process_dataset(dataset):
    processed_texts = [f"Term: {entry['term']}\nDefinition: {entry['definition']}\nPOS: {entry['pos']}\nFacet: {entry['facet']}\nTop Level Domain: {entry['top_level_domain']}\nLevel2: {entry['level2']}\nLevel3: {entry['level3']}" for entry in dataset]
    return processed_texts

# 데이터셋 분할
def split_dataset(dataset, train_ratio=0.8):
    random.shuffle(dataset)  # 데이터셋 섞기
    train_size = int(len(dataset) * train_ratio)
    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size:]
    return train_dataset, valid_dataset

# 전처리 실행
def run_preprocessing(json_string, train_ratio=0.8):
    dataset = load_json_dataset(json_string)
    processed_texts = process_dataset(dataset)
    
    # 처리된 텍스트 데이터 확인
    print(processed_texts[0])
            
    # 데이터셋을 학습 데이터셋과 검증 데이터셋으로 분할
    train_dataset, valid_dataset = split_dataset(processed_texts, train_ratio)

    # 전처리된 데이터셋을 JSON 파일로 저장
    with open('train_data_processed.json', 'w') as file:
        json.dump(train_dataset, file, ensure_ascii=False, indent=1)

    with open('valid_data_processed.json', 'w') as file:
        json.dump(valid_dataset, file, ensure_ascii=False, indent=1)

    print("Data preprocessing completed.")

# 전처리 실행
if __name__ == '__main__':
    with open('dataset.json', 'r') as f:
        dataset_json = f.read()
    train_ratio = 0.8
    run_preprocessing(dataset_json, train_ratio)
