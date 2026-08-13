import os
import glob
import re
import math
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "best.pt"
TEST_IMAGES_DIR = "custom_dataset/demo"
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
        results = model(img_bgr, conf=0.5, iou=0.45, verbose=False)[0]
        
        boxes_list = []
        if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
            xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
            xywhr = results.obb.xywhr.cpu().numpy()
            confs = results.obb.conf.cpu().numpy()
            clss = results.obb.cls.cpu().numpy().astype(int)
            
            for i in range(len(confs)):
                boxes_list.append({
                    'cls': clss[i],
                    'name': model.names[clss[i]],
                    'conf': float(confs[i]),
                    'pts': xyxyxyxy[i],
                    'xywhr': xywhr[i]
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
            
            # cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), 1)
            
            cv2.putText(annotated, badge_text, (badge_x1, badge_y2 - pad_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

            # 抓取姿態標註 (Grasp Pose Overlay)
            cx, cy, bw, bh, rad = b['xywhr']
            grasp_len = max(bw, bh) * 0.5
            dx = (grasp_len / 2.0) * math.cos(rad + math.pi / 2.0)
            dy = (grasp_len / 2.0) * math.sin(rad + math.pi / 2.0)
            p1 = (int(cx - dx), int(cy - dy))
            p2 = (int(cx + dx), int(cy + dy))
            cv2.line(annotated, p1, p2, (255, 255, 0), max(2, line_thick), cv2.LINE_AA)
            dx2 = (grasp_len / 3.0) * math.cos(rad)
            dy2 = (grasp_len / 3.0) * math.sin(rad)
            p3 = (int(cx - dx2), int(cy - dy2))
            p4 = (int(cx + dx2), int(cy + dy2))
            cv2.line(annotated, p3, p4, (0, 255, 255), max(2, line_thick), cv2.LINE_AA)
            cv2.circle(annotated, (int(cx), int(cy)), max(4, line_thick + 2), (0, 0, 255), -1, cv2.LINE_AA)

        # 5. 統一輸出固定 9:16 尺寸 (寬 540 x 高 960)，
        #    先讓標註影像左右貼齊邊界、上下以黑邊補滿並垂直置中，
        #    最後再把 HUD 資訊條疊在畫布底部（確保每幀都顯示檔名）
        canvas_w, canvas_h = 960, 540
        hud_h = 36
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        scale = canvas_w / float(annotated.shape[1])
        frame_h = int(round(annotated.shape[0] * scale))
        usable_h = canvas_h - hud_h
        if frame_h > usable_h:
            scale = usable_h / float(annotated.shape[0])
            frame_h = usable_h
            frame_w = int(round(annotated.shape[1] * scale))
            resized = cv2.resize(annotated, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
            x_offset = max(0, (canvas_w - frame_w) // 2)
            canvas[0:frame_h, x_offset:x_offset + frame_w] = resized
        else:
            resized = cv2.resize(annotated, (canvas_w, frame_h), interpolation=cv2.INTER_AREA)
            y_offset = (usable_h - frame_h) // 2
            canvas[y_offset:y_offset + frame_h, :] = resized

        hud_bar = np.zeros((hud_h, canvas_w, 3), dtype=np.uint8)
        hud_bar[:] = (25, 25, 25)
        cv2.putText(hud_bar, f"YOLOv11-OBB Inference Demo [{idx+1}/{total_imgs}] {filename}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        counts_str = f"Objects: {len(boxes_list)}"
        cv2.putText(hud_bar, counts_str, (canvas_w - 160, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        canvas[canvas_h - hud_h:, :] = hud_bar
        frame_resized = canvas
        
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
