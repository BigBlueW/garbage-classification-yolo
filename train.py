import os
import shutil
import torch
from ultralytics import YOLO

def main():
    model_name = "yolo11x-obb.pt"
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 準備啟動 YOLOv11-OBB 端到端訓練流程...")
    print(f"📌 使用裝置: {device} | 基礎模型: {model_name}")

    # 載入官方 OBB 預訓練模型
    model = YOLO(model_name)

    # 開始端到端 OBB 訓練
    results = model.train(
        data="custom_dataset.yaml",
        epochs=300,
        patience=50,
        imgsz=640,
        batch=16,
        workers=8,
        cache=True,
        lr0=0.002,
        cos_lr=True,
        project="garbage_classification_runs",
        name="yolo11x_obb_model",
        device=device,
        save=True
    )

    best_weight = "garbage_classification_runs/yolo11x_obb_model/weights/best.pt"
    if os.path.exists(best_weight):
        shutil.copy(best_weight, "best.pt")
        print(f"🎉 訓練完成！最佳權重已儲存至: {best_weight}")
        print(f"📦 已自動更新根目錄 best.pt 為最新的 OBB 模型權重！")
    else:
        print("🎉 訓練結束！")

if __name__ == '__main__':
    main()
