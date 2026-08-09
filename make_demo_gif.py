import os
import glob
import re
import math
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"
TEST_IMAGES_DIR = "custom_dataset/test/images"
OUTPUT_GIF_PATH = "demo.gif"

CLASS_COLORS = {
    0: (46, 204, 113),   # plastic - 綠色
    1: (235, 152, 52),   # metal - 藍色
    2: (34, 126, 230),   # paper - 橘色
    3: (255, 0, 255)     # general_waste - 螢光洋紅/亮紫紅
}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def obb_iou(pts1, pts2):
    p1 = np.ascontiguousarray(pts1, dtype=np.float32)
    p2 = np.ascontiguousarray(pts2, dtype=np.float32)
    area1 = cv2.contourArea(p1)
    area2 = cv2.contourArea(p2)
    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def apply_agnostic_nms(boxes_list, iou_thresh=0.40):
    if len(boxes_list) <= 1:
        return boxes_list
    boxes_list = sorted(boxes_list, key=lambda x: x['conf'], reverse=True)
    kept = []
    while boxes_list:
        best = boxes_list.pop(0)
        kept.append(best)
        boxes_list = [b for b in boxes_list if obb_iou(best['pts'], b['pts']) < iou_thresh]
    return kept

def main():
    print(f"🚀 正在載入模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext.upper())))
        
    image_paths = sorted(image_paths, key=natural_sort_key)
    total_imgs = len(image_paths)
    print(f"📸 找到 {total_imgs} 張測試圖片，開始繪製標註影像...")
    
    gif_frames = []
    
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        
        # 1. 執行推論
        results = model(img_bgr, conf=0.30, iou=0.45, verbose=False)[0]
        
        boxes_list = []
        if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
            xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
            confs = results.obb.conf.cpu().numpy()
            clss = results.obb.cls.cpu().numpy().astype(int)
            
            for i in range(len(confs)):
                boxes_list.append({
                    'cls': clss[i],
                    'name': model.names[clss[i]],
                    'conf': float(confs[i]),
                    'pts': xyxyxyxy[i]
                })
                
        # 2. 跨類別 NMS
        boxes_list = apply_agnostic_nms(boxes_list, iou_thresh=0.40)
        
        # 3. 繪製邊界框與標籤
        annotated = img_bgr.copy()
        font_scale = max(0.85, min(1.3, w / 900.0))
        font_thick = max(2, int(round(font_scale * 2.0)))
        line_thick = max(2, int(round(w / 450.0)))
        
        for b in boxes_list:
            cls_id = b['cls']
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            pts = b['pts'].astype(np.int32).reshape((-1, 1, 2))
            
            # 旋轉框線條
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=line_thick, lineType=cv2.LINE_AA)
            
            # 頂點標籤
            top_pt = b['pts'][np.argmin(b['pts'][:, 1])]
            bx, by = int(top_pt[0]), int(top_pt[1])
            
            badge_text = f" {b['name']} {b['conf']:.2f} "
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            pad_y = int(th * 0.35)
            badge_y1 = max(0, by - th - pad_y * 2)
            badge_y2 = badge_y1 + th + pad_y * 2
            badge_x1 = max(0, bx)
            badge_x2 = min(w, badge_x1 + tw)
            
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), 1)
            
            cv2.putText(annotated, badge_text, (badge_x1, badge_y2 - pad_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # 4. 底部資訊條 (HUD Bar)
        hud_h = 36
        hud_bar = np.zeros((hud_h, w, 3), dtype=np.uint8)
        hud_bar[:] = (25, 25, 25)
        
        cv2.putText(hud_bar, f"YOLOv11-OBB Inference Demo [{idx+1}/{total_imgs}] {filename}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        counts_str = f"Objects: {len(boxes_list)}"
        cv2.putText(hud_bar, counts_str, (w - 150, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        
        final_frame = np.vstack([annotated, hud_bar])
        
        # 5. 調整尺寸至適合展示的解析度 (高度 720px)
        target_h = 720
        target_w = int(final_frame.shape[1] * (target_h / float(final_frame.shape[0])))
        frame_resized = cv2.resize(final_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # 轉換為 RGB PIL Image 並做調色盤量化
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).convert('P', palette=Image.ADAPTIVE, colors=256)
        gif_frames.append(pil_frame)
        print(f"  [{idx+1}/{total_imgs}] {filename} -> 完成 (偵測到 {len(boxes_list)} 個垃圾)")

    if gif_frames:
        print(f"\n🎬 正在產生最佳化 GIF 動圖（每張 1.0 秒，共 {len(gif_frames)} 幀）...")
        gif_frames[0].save(
            OUTPUT_GIF_PATH,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=1000,  # 1000ms = 1秒
            loop=0
        )
        file_size_mb = os.path.getsize(OUTPUT_GIF_PATH) / (1024 * 1024)
        print(f"🎉 demo.gif 製作完成！儲存至: {OUTPUT_GIF_PATH} (大小: {file_size_mb:.2f} MB)")
    else:
        print("❌ 沒有產生任何幀。")

if __name__ == '__main__':
    main()
