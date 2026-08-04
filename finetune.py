import os
from ultralytics import YOLO

def main():
    # 1. 載入我們「最一開始」用 Kaggle 訓練出來的 6 類原始大模型
    # 若該路徑不存在（例如新 Clone 的環境），則退回使用根目錄目前的 best.pt
    original_model_path = "runs/detect/garbage_classification_runs/yolo11x_garbage_model/weights/best.pt"
    if not os.path.exists(original_model_path):
        print(f"Warning: 找不到原始 6 類模型，將使用目前的 best.pt 進行微調。")
        original_model_path = "best.pt"
        
    model = YOLO(original_model_path) 
    
    print("🚀 開始進行視角微調 (Fine-tuning)...")
    
    # 2. 開始訓練
    # YOLO 會自動偵測到 custom_dataset.yaml 只有 4 類
    # 於是它會「保留大部分的視覺辨識能力」，只重置最後一層的分類器來適應新的 4 大類！
    results = model.train(
        data="custom_dataset.yaml",
        epochs=300,            # 增加輪數，讓它徹底收斂
        patience=50,           # 若 50 輪沒進步自動提早結束
        imgsz=640,
        batch=16,
        workers=8,
        freeze=10,             # ⭐️ 凍結骨幹網路：保留原本大模型對特徵的理解，避免小資料破壞權重
        lr0=0.001,             # ⭐️ 降低初始學習率，讓微調過程更穩固
        project="garbage_classification_runs",
        name="finetuned_robot_model_pro",
        device="0"
    )
    
    print("🎉 微調完成！新的專屬權重已儲存！")

if __name__ == '__main__':
    main()
