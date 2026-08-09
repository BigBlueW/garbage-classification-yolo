import os
from ultralytics import YOLO

# 模型路徑
MODEL_PATH = "best.pt"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型檔案: {MODEL_PATH}，請確認訓練是否已經完成！")
        return

    print(f"🚀 載入模型進行評估: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 選擇資料集配置檔
    data_yaml = "custom_dataset.yaml" if os.path.exists("custom_dataset.yaml") else "dataset.yaml"
    print(f"🔍 開始在測試集 ({data_yaml} -> val/test) 上進行驗證...")
    
    metrics = model.val(data=data_yaml, split="val")

    print("\n" + "="*55)
    print("📊 測試集評估結果 (Test Set Metrics)")
    print("="*55)

    # 判斷是 OBB 還是 HBB
    metric_obj = metrics.obb if hasattr(metrics, 'obb') and metrics.obb is not None else metrics.box
    task_type = "OBB (Oriented Bounding Box)" if metric_obj == getattr(metrics, 'obb', None) else "HBB (Horizontal Box)"
    
    print(f"任務類型: {task_type}")
    print("-" * 55)

    # 取得整體指標
    map50 = metric_obj.map50
    map95 = metric_obj.map
    precision = metric_obj.mp
    recall = metric_obj.mr

    print(f"綜合 Precision (精準率): {precision:.4f}  (預測有垃圾時的準確率)")
    print(f"綜合 Recall    (召回率): {recall:.4f}  (真實垃圾中被成功偵測的比例)")
    print(f"綜合 mAP@50            : {map50:.4f}  (主流 IoU 0.50 指標)")
    print(f"綜合 mAP@50-95         : {map95:.4f}  (嚴格綜合指標)")
    print("="*55)
    
    # 針對各類別的詳細指標
    print(f"{'類別名稱 (Class)':<18} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10}")
    print("-" * 55)
    
    names = model.names
    for i, class_idx in enumerate(metric_obj.ap_class_index):
        c_name = names[class_idx]
        c_p = metric_obj.p[i]
        c_r = metric_obj.r[i]
        c_map50 = metric_obj.ap50[i]
        print(f"{c_name:<18} | {c_p:<10.4f} | {c_r:<10.4f} | {c_map50:<10.4f}")
        
    print("="*55)
    print(f"評估報表與圖表已儲存至: {metrics.save_dir}")

if __name__ == '__main__':
    main()
