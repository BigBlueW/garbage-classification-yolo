import os
import glob
import cv2

IMAGES_DIR = "custom_dataset/train/images"
LABELS_DIR = "custom_dataset/train/labels"

CLASSES = {
    ord('0'): (0, "Plastic"),
    ord('1'): (1, "Metal"),
    ord('2'): (2, "Paper"),
    ord('3'): (3, "General Waste")
}

colors = [
    (0, 255, 0),    # 0: Plastic - 綠色
    (255, 255, 0),  # 1: Metal - 青色
    (0, 165, 255),  # 2: Paper - 橘色
    (200, 200, 200) # 3: General - 灰色
]

drawing = False
ix, iy = -1, -1
temp_box = None
boxes = []  # 存放 (class_id, x1, y1, x2, y2)
img = None
clone = None
w, h = 0, 0

def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, temp_box, clone, img, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        # 檢查點擊是否在任何已存在的框內
        clicked_box_idx = -1
        # 從最後一個畫的框開始檢查（最上層）
        for i in range(len(boxes)-1, -1, -1):
            cls_id, bx1, by1, bx2, by2 = boxes[i]
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                clicked_box_idx = i
                break
        
        if clicked_box_idx != -1:
            # 如果點在框內，直接刪除該框
            boxes.pop(clicked_box_idx)
            redraw_boxes()
        else:
            # 如果點在空白處，開始畫新框
            drawing = True
            ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            for b in boxes:
                cls_id, bx1, by1, bx2, by2 = b
                cv2.rectangle(img, (bx1, by1), (bx2, by2), colors[cls_id], 2)
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            x1, x2 = min(ix, x), max(ix, x)
            y1, y2 = min(iy, y), max(iy, y)
            if x2 - x1 > 10 and y2 - y1 > 10:
                temp_box = (x1, y1, x2, y2)
                redraw_boxes()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Press 0:Plastic, 1:Metal, 2:Paper, 3:General", (x1, max(30, y1-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def redraw_boxes():
    global img, clone
    img = clone.copy()
    for b in boxes:
        cls_id, x1, y1, x2, y2 = b
        c_color = colors[cls_id]
        c_name = [v[1] for k,v in CLASSES.items() if v[0] == cls_id][0]
        cv2.rectangle(img, (x1, y1), (x2, y2), c_color, 2)
        cv2.putText(img, c_name, (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_color, 2)

def load_yolo_labels(txt_path, img_w, img_h):
    loaded_boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    # 將 YOLO 的正規化座標轉換回像素座標
                    x1 = int((cx - bw/2) * img_w)
                    y1 = int((cy - bh/2) * img_h)
                    x2 = int((cx + bw/2) * img_w)
                    y2 = int((cy + bh/2) * img_h)
                    loaded_boxes.append((cls_id, x1, y1, x2, y2))
    return loaded_boxes

def save_yolo_labels(txt_path, img_w, img_h):
    # 若該圖片最後沒有任何框，則刪除 txt，避免產生空檔案
    if len(boxes) == 0:
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return

    with open(txt_path, "w") as f:
        for b in boxes:
            cls_id, x1, y1, x2, y2 = b
            center_x = ((x1 + x2) / 2.0) / img_w
            center_y = ((y1 + y2) / 2.0) / img_h
            bw_norm = (x2 - x1) / float(img_w)
            bh_norm = (y2 - y1) / float(img_h)
            f.write(f"{cls_id} {center_x:.6f} {center_y:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

def main():
    global img, clone, temp_box, boxes, w, h
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext.upper())))
    image_paths = sorted(image_paths)

    if not image_paths:
        print("沒有找到任何圖片。")
        return

    cv2.namedWindow("Label & Review Tool")
    cv2.setMouseCallback("Label & Review Tool", draw_bbox)

    print("\n" + "="*40)
    print(" 歡迎使用全能垃圾標註與檢查工具")
    print("="*40)
    print("操作說明:")
    print("  左鍵點擊已存在的框 : 直接刪除該框 (Delete)")
    print("  左鍵拖曳 : 畫出新的框，然後按 0~3 賦予標籤")
    print("\n導覽鍵:")
    print("  a : 上一張圖片 (會自動存檔)")
    print("  d 或 Space : 下一張圖片 (會自動存檔)")
    print("  q : 儲存並離開程式")
    print("="*40)

    img_idx = 0
    while img_idx < len(image_paths):
        img_path = image_paths[img_idx]
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        txt_path = os.path.join(LABELS_DIR, name + ".txt")

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            img_idx += 1
            continue
            
        h_raw, w_raw = raw_img.shape[:2]
        max_dim = 960
        scale = 1.0
        if max(h_raw, w_raw) > max_dim:
            scale = max_dim / max(h_raw, w_raw)
            raw_img = cv2.resize(raw_img, (int(w_raw*scale), int(h_raw*scale)))
        
        h, w = raw_img.shape[:2]
        clone = raw_img.copy()
        
        boxes = load_yolo_labels(txt_path, w, h)
        temp_box = None
        redraw_boxes()
        
        print(f"\n[{img_idx+1}/{len(image_paths)}] 審查中: {filename}")

        while True:
            display_img = img.copy()
            # 在畫面左上角印出進度與提示
            cv2.putText(display_img, f"[{img_idx+1}/{len(image_paths)}] A:Prev  D:Next  Click Box to Delete", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Label & Review Tool", display_img)
            
            key = cv2.waitKey(20) & 0xFF

            if key in CLASSES and temp_box is not None:
                cls_id = CLASSES[key][0]
                boxes.append((cls_id, *temp_box))
                temp_box = None
                redraw_boxes()
                
            elif key == ord('a'): # 上一張
                save_yolo_labels(txt_path, w, h)
                img_idx = max(0, img_idx - 1)
                break

            elif key == ord('d') or key == 13 or key == 32: # 下一張 (d, Enter, Space)
                save_yolo_labels(txt_path, w, h)
                img_idx += 1
                break

            elif key == ord('q'): # 退出
                save_yolo_labels(txt_path, w, h)
                print("退出審查工具。")
                cv2.destroyAllWindows()
                return

    print("\n審查完畢！如果發現剛才有修正錯誤的標籤，記得再跑一次 finetune.py 重新訓練！")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
