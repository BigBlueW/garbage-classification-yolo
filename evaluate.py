import os
from ultralytics import YOLO

# 請確保這個路徑指向您訓練出來的 best.pt
MODEL_PATH = "best.pt"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型檔案: {MODEL_PATH}")
        print("請確認訓練是否已經完成，或者路徑是否正確！")
        return

    print(f"🚀 載入模型進行評估: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 執行評估 (針對 dataset.yaml 裡定義的 test 資料集)
    print("🔍 開始在測試集 (Test Set) 上進行驗證，這可能需要一點時間...")
    # model.val 會自動計算各種指標
    metrics = model.val(data="dataset.yaml", split="test")

    print("\n" + "="*55)
    print("📊 測試集評估結果 (Test Set Metrics)")
    print("="*55)

    # 取得整體指標 (Mean Metrics)
    map50 = metrics.box.map50
    map95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print(f"綜合 Precision (精準率): {precision:.4f}  (模型說有垃圾時，有多準)")
    print(f"綜合 Recall    (召回率): {recall:.4f}  (真正的垃圾中，模型抓出多少)")
    print(f"綜合 mAP@50            : {map50:.4f}  (主流評估指標)")
    print(f"綜合 mAP@50-95         : {map95:.4f}  (嚴格評估指標)")
    print("="*55)
    
    # 針對各類別的詳細指標 (Per-class Metrics)
    print(f"{'類別名稱 (Class)':<16} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10}")
    print("-" * 55)
    
    names = model.names
    # 巡覽所有有出現的類別
    for i, class_idx in enumerate(metrics.ap_class_index):
        c_name = names[class_idx]
        c_p = metrics.box.p[i]
        c_r = metrics.box.r[i]
        c_map50 = metrics.box.ap50[i]
        
        print(f"{c_name:<18} | {c_p:<10.4f} | {c_r:<10.4f} | {c_map50:<10.4f}")
        
    print("="*55)
    print(f"Results saved to: {metrics.save_dir}")

if __name__ == '__main__':
    main()
