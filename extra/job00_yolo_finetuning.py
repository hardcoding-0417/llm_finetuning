import torch
from ultralytics import YOLO

# yolo를 파인튜닝해보는 실습입니다.

def main():
    # YOLO 모델 로드 (사전학습 모델)
    model = YOLO('yolov8n.pt')
    
    #모델 학습
    model.train(
        data={'data.yaml'},
        epochs=100,
        batch_size=32,
        imgsz=640,
        freeze=10
    )

    # 모델 저장
    model.save('finetuned_yolo.pt')

if __name__ == '__main__':
    main()
