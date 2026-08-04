# AI Robotics - Garbage Classification (YOLOv11)

這是一個基於 Ultralytics YOLOv11x 訓練的垃圾分類影像辨識模組。

## 專案介紹

* **原始分類 (6類)**：`BIODEGRADABLE` (廚餘), `CARDBOARD` (紙板), `GLASS` (玻璃), `METAL` (金屬), `PAPER` (紙張), `PLASTIC` (塑膠)。
* **實戰應用合併 (4大類)**：為了符合機器人的夾爪分類邏輯，透過腳本將這 6 種類別動態合併為 4 大類：`general_waste` (一般垃圾/其他), `paper` (紙類), `metal` (金屬), `plastic` (塑膠)。

## 環境與安裝 (Installation)

1. **虛擬環境**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux
# 或在 Windows 用: .venv\Scripts\activate
```

2. **安裝依賴套件**
```bash
pip install -r requirements.txt
```

## 資料準備 (Dataset & Weights)

1. **預訓練權重 (Model Weights)**
   * 本專案已使用 **Git LFS (Large File Storage)** 來追蹤訓練好的模型權重檔 `best.pt` (114MB)。
   * 當您 `git clone` 本專案時，若有正確安裝 Git LFS，`best.pt` 就會自動下載到專案根目錄。

2. **資料集 (Dataset)**
   * 由於幾萬張圖片過於龐大，未包含在此 Repo 中。
   * 資料來源：[Kaggle - Garbage Detection (viswaprakash1990)](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection)
   * 若需重新訓練，請自行下載並解壓縮，確保資料夾結構為：`archive/GARBAGE CLASSIFICATION/train/...`

## 使用方式

### 1. 圖形化測試
圖形化介面，可以隨機抽取測試集中的照片進行預測，並展示 6合4 的動態標籤轉換效果。
```bash
python test_gui.py
```

### 2. 評估模型
若要查看模型在測試集上的詳細數據（包含 Precision, Recall, mAP@50 等統計表格），請執行：
```bash
python evaluate.py
```

### 3. 重新訓練
若您有更強的顯示卡（本專案原使用 RTX 5090 32GB 進行訓練），或者想加入自己的資料，請執行：
```bash
python train.py
```
> 💡 提示：`train.py` 預設的 batch size 為 32，若您的顯示卡 VRAM 較小（如 8GB），請進入 `train.py` 將 `batch` 改為 8 或 4，以避免 OOM 錯誤。

## 📄 授權 (License)
資料集標註採用 CC BY 4.0 授權。程式碼部分可自由用於學術與專案交流。
