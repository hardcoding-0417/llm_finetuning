# README

## 1. 용어집 다운로드 및 설정

1. [AI Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71614)에서 용어집을 다운로드합니다.
    - 경로: `160.문화, 게임 콘텐츠 분야 용어 말뭉치\01-1.정식개방데이터\Training\02.라벨링데이터\TL용어.json`
2. 다운로드한 용어집의 이름을 `dataset.json`으로 변경하여 코드와 같은 폴더에 넣어줍니다.

## 2. PyTorch 및 Transformers 설치

### NVIDIA Driver, CUDA, CuDNN 설치

1. 먼저, 설치하고자 하는 CUDA를 지원하는 NVIDIA DRIVER를 설치합니다.
    - [NVIDIA Driver 다운로드](https://www.nvidia.com/Download/index.aspx)

2. CUDA를 설치합니다.
    - [CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-toolkit-archive)
    - 예: CUDA 11.8 또는 CUDA 12.1

3. CuDNN을 설치합니다.
    - [CuDNN 다운로드](https://developer.nvidia.com/cudnn)
    - CUDA 버전에 맞는 CuDNN 버전을 설치합니다.

### PyTorch 설치

4. PyTorch 공홈에서 스크롤을 쭉 내리면  
내 CUDA에 맞는 설치 명령어를 찾을 수 있습니다. 해당 명령어를 터미널에서 실행해줍니다.
    - [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
    - 예시:
      ```bash
      pip install torch torchvision torchaudio
      # CUDA 11.8
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      # CUDA 12.1
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      ```

5. Transformers 라이브러리를 설치해줍니다.
    ```bash
    pip install transformers
    ```
  
## 코드

코드는 크게 두 부분, 전처리와 파인튜닝으로 나뉘어 있습니다.

### 데이터 전처리
- JSON 파일에서 데이터를 로드합니다.
- 중첩된 JSON 구조를 평탄화합니다.
- 데이터를 학습 데이터셋과 검증 데이터셋으로 분할합니다.
- 전처리된 데이터를 JSON 파일로 저장합니다.

### 모델 파인튜닝
- Hugging Face의 사전 학습모델과 해당 모델의 토크나이저를 로드합니다.
- 데이터셋을 토크나이저로 인코딩합니다. (토크나이징도 전처리에 해당하는 부분입니다.)
- 인코딩된 데이터를 PyTorch 데이터셋으로 변환합니다.
- Trainer 클래스로 모델을 파인튜닝합니다.
- 파인튜닝된 모델과 토크나이저를 저장합니다.

## 모델 정보

튜닝에 사용한 모델은 [qwen2](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)입니다. 24년 06월 기준으로 굉장히 핫한 모델로, 중국에서 만들었습니다.

다국어 모델 중에서 가장 높은 성능을 갖고 있고, 0.5b 파라미터짜리 모델도 나와있어 4090 한 대로도 파인튜닝이 가능합니다. 
  (VRAM이 충분하다면 LLaMA3 기반의 블라썸(서울 과기대)을 추천드립니다.)

모델이 학습할 때 사용된 토크나이저를 그대로 사용해야 파인튜닝도 효과적으로 이루어지니 주의하세요.

## 기타 추천 사항

이 레포에서는 Hugging Face로 파인튜닝을 진행했으나 아홀로틀(Axolotl)과 같이  
파인튜닝을 위해 만들어진 라이브러리를 익혀두는 편이 더 좋습니다.  
아홀로틀보다 더 쉬운 걸 찾고 있다면 코드 없이도 진행 가능한 라마-팩토리나 우바부가를 추천드리고,  
VRAM이 부족해도 파인튜닝을 진행해보고 싶다면 Unsloth을 추천드립니다.  
여유가 되면 extra 폴더에 있는 코드로 yolo도 파인튜닝해보세요.  
