import os
import glob
import re
import math
import argparse
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

CLASS_COLORS = {
    0: (46, 204, 113),   # plastic - 綠色
    1: (235, 152, 52),   # metal - 藍色
    2: (34, 126, 230),   # paper - 橘色
    3: (255, 0, 255)     # general_waste - 螢光洋紅/亮紫紅
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate demo GIF with side-by-side original and annotated comparison.")
    parser.add_argument("--path", type=str, default="custom_dataset/demo", help="Path to input images directory")
    parser.add_argument("--output", type=str, default="demo.gif", help="Path to output GIF file")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to trained YOLO model")
    parser.add_argument("--height", type=int, default=480, help="Target height for each image panel in pixels (width dynamically follows aspect ratio)")
    parser.add_argument("--ref-image", type=str, default=None, help="Optional specific image path to use as aspect ratio reference")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--duration", type=int, default=1000, help="Duration per frame in milliseconds")
    parser.add_argument("--no-grasp", action="store_true", help="Disable grasp pose overlay")
    return parser.parse_args()

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

def fit_to_panel(img, pw, ph):
    """將影像以等比例縮放至 (pw, ph) 面板中，若比例不完全相同則置中填補黑邊"""
    ih, iw = img.shape[:2]
    scale = min(pw / float(iw), ph / float(ih))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    ox = (pw - nw) // 2
    oy = (ph - nh) // 2
    panel[oy:oy + nh, ox:ox + nw] = resized
    return panel

def main():
    args = parse_args()
    
    # 支援替代模型路徑
    model_path = args.model
    if not os.path.exists(model_path):
        alt_weights = [
            "best.pt",
            "garbage_classification_runs/yolo11x_obb_model/weights/best.pt",
            "yolo11x-obb.pt"
        ]
        for alt in alt_weights:
            if os.path.exists(alt):
                model_path = alt
                break

    print(f"🚀 正在載入模型: {model_path} ...")
    model = YOLO(model_path)
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.path, ext)))
        image_paths.extend(glob.glob(os.path.join(args.path, ext.upper())))
        
    image_paths = sorted(image_paths, key=natural_sort_key)
    total_imgs = len(image_paths)
    if total_imgs == 0:
        print(f"❌ 在目錄 '{args.path}' 中找不到任何圖片！")
        return

    print(f"📸 找到 {total_imgs} 張測試圖片，開始計算畫布比例與繪製標註影像...")
    
    # 決定參考比例（彈性尺寸設定：由特定參考圖或第一張圖決定長寬比）
    ref_img_path = args.ref_image if (args.ref_image and os.path.exists(args.ref_image)) else image_paths[0]
    ref_bgr = cv2.imread(ref_img_path)
    if ref_bgr is not None:
        ref_h, ref_w = ref_bgr.shape[:2]
        aspect_ratio = ref_w / float(ref_h)
    else:
        aspect_ratio = 16.0 / 9.0

    # 單圖面板尺寸
    panel_h = args.height
    panel_w = int(round(panel_h * aspect_ratio))
    divider_w = max(3, int(panel_w / 200.0))
    hud_h = max(32, int(panel_h * 0.08))
    
    # 總畫面尺寸（左原圖 + 分隔線 + 右標註圖，底部為 HUD 資訊列）
    total_w = panel_w * 2 + divider_w
    total_h = panel_h + hud_h
    print(f"📐 畫布規格: 單圖 ({panel_w}x{panel_h}) | 左右並排總尺寸 ({total_w}x{total_h}) (參考圖: {os.path.basename(ref_img_path)})")

    gif_frames = []
    show_grasp = not args.no_grasp
    
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        
        # 1. 執行推論
        results = model(img_bgr, conf=args.conf, iou=args.iou, verbose=False)[0]
        
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
        boxes_list = apply_agnostic_nms(boxes_list, iou_thresh=args.iou)
        
        # 3. 繪製標註圖
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
            
            # 繪製醒目白色描邊（8 方向擴展純白外框，確保在任何背景皆有清晰白邊）
            text_x = max(4, badge_x1)
            text_y = badge_y2 - pad_y
            outline_radius = max(2, int(round(font_scale * 2.5)))
            
            # 先以多重偏移繪製純白實心描邊
            for dx in range(-outline_radius, outline_radius + 1):
                for dy in range(-outline_radius, outline_radius + 1):
                    if dx != 0 or dy != 0:
                        cv2.putText(annotated, badge_text, (text_x + dx, text_y + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick + 1, cv2.LINE_AA)
            
            # 再於中心繪製類別專屬顏色文字
            cv2.putText(annotated, badge_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

            # 抓取姿態標註 (Grasp Pose Overlay)
            if show_grasp:
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

        # 4. 建立左邊原圖面板與右邊標註圖面板（依比例縮放）
        left_panel = fit_to_panel(img_bgr, panel_w, panel_h)
        right_panel = fit_to_panel(annotated, panel_w, panel_h)

        # 加上清晰的左/右標題標籤
        tag_scale = max(0.65, min(1.1, panel_w / 600.0))
        tag_thick = max(2, int(round(tag_scale * 2.0)))
        
        # 左側標籤：Original
        orig_text = "Original"
        orig_pos = (14, int(30 * tag_scale + 4))
        cv2.putText(left_panel, orig_text, orig_pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (0, 0, 0), tag_thick + 2, cv2.LINE_AA)
        cv2.putText(left_panel, orig_text, orig_pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (255, 255, 255), tag_thick, cv2.LINE_AA)

        # 右側標籤：OBB Detection
        det_text = f"OBB Detection ({len(boxes_list)} objects)"
        det_pos = (14, int(30 * tag_scale + 4))
        cv2.putText(right_panel, det_text, det_pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (0, 0, 0), tag_thick + 2, cv2.LINE_AA)
        cv2.putText(right_panel, det_text, det_pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (0, 255, 255), tag_thick, cv2.LINE_AA)

        # 5. 左右拼接
        divider = np.full((panel_h, divider_w, 3), 45, dtype=np.uint8)
        combined_panels = np.hstack([left_panel, divider, right_panel])

        # 6. 底部 HUD 資訊列
        hud_bar = np.full((hud_h, total_w, 3), 25, dtype=np.uint8)
        hud_font_scale = max(0.5, min(0.8, total_w / 1400.0))
        hud_text_y = int(hud_h * 0.65)
        
        cv2.putText(hud_bar, f"YOLOv11-OBB Inference Demo [{idx+1}/{total_imgs}] {filename}", (16, hud_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        counts_str = f"Detections: {len(boxes_list)}"
        (cw, _), _ = cv2.getTextSize(counts_str, cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, 1)
        cv2.putText(hud_bar, counts_str, (total_w - cw - 16, hud_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 255, 255), 1, cv2.LINE_AA)

        frame = np.vstack([combined_panels, hud_bar])
        
        # 轉換為 RGB PIL Image 並做調色盤量化
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).convert('P', palette=Image.ADAPTIVE, colors=256)
        gif_frames.append(pil_frame)
        print(f"  [{idx+1}/{total_imgs}] {filename} -> 完成 (偵測到 {len(boxes_list)} 個垃圾)")

    if gif_frames:
        print(f"\n🎬 正在產生最佳化 GIF 動圖（每張 {args.duration/1000.0:.1f} 秒，共 {len(gif_frames)} 幀）...")
        gif_frames[0].save(
            args.output,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=args.duration,
            loop=0
        )
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"🎉 {args.output} 製作完成！(大小: {file_size_mb:.2f} MB)")
    else:
        print("❌ 沒有產生任何幀。")

if __name__ == '__main__':
    main()
