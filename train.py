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

    # 開始端到端 OBB 訓練（4 類：plastic / metal / paper / general_waste）
    model.train(
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

    # 從 trainer 直接取得最佳權重路徑並複製到根目錄（避免路徑硬編碼錯誤）
    best_weight = getattr(getattr(model, "trainer", None), "best", None)
    if best_weight and os.path.exists(best_weight):
        shutil.copy(best_weight, "best.pt")
        print(f"🎉 訓練完成！最佳權重已複製至: best.pt")
    else:
        print("🎉 訓練結束！")

if __name__ == '__main__':
    main()