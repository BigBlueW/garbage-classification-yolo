import os
import random
import glob
import math
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Garbage Classification Inference Engine")
    parser.add_argument("--path", type=str, default="custom_dataset/test/images", help="Path to the test images directory")
    return parser.parse_args()

args = parse_args()

# Setup CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Config Paths
MODEL_PATH = "best.pt"
TEST_IMAGES_DIR = args.path if os.path.exists(args.path) else "custom_dataset/test/images"

# BGR Colors for OpenCV rendering
CLASS_COLORS = {
    0: (46, 204, 113),   # plastic - 翠綠色
    1: (235, 152, 52),   # metal - 藍青色
    2: (34, 126, 230),   # paper - 亮橘色
    3: (255, 0, 255)     # general_waste - 螢光洋紅/亮紫紅 (超高對比，極清晰)
}

def obb_iou(pts1, pts2):
    """計算兩個任意四邊形/旋轉邊界框之間的精確 IoU"""
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
    """跨類別 NMS：若多個不同類別的框重疊，只保留最高信心度的那一個"""
    if len(boxes_list) <= 1:
        return boxes_list
        
    boxes_list = sorted(boxes_list, key=lambda x: x['conf'], reverse=True)
    kept = []
    
    while boxes_list:
        best = boxes_list.pop(0)
        kept.append(best)
        boxes_list = [b for b in boxes_list if obb_iou(best['pts'], b['pts']) < iou_thresh]
        
    return kept

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Garbage Classification Inference Engine (YOLOv11-OBB)")
        self.geometry("1180x780")
        
        # Configure Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Garbage Vision", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 5))

        self.model_status_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Loading Model...", 
            text_color="gray70",
            font=ctk.CTkFont(size=12)
        )
        self.model_status_label.grid(row=1, column=0, padx=20, pady=(0, 15))

        # Confidence Slider
        self.conf_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Confidence: 0.60", 
            anchor="w"
        )
        self.conf_label.grid(row=2, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.conf_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0.05, 
            to=0.95, 
            number_of_steps=90, 
            command=self.update_sliders
        )
        self.conf_slider.set(0.60)
        self.conf_slider.grid(row=3, column=0, padx=20, pady=(5, 10))

        # NMS IoU Slider
        self.iou_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Agnostic NMS IoU: 0.40", 
            anchor="w"
        )
        self.iou_label.grid(row=4, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.iou_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0.10, 
            to=0.90, 
            number_of_steps=80, 
            command=self.update_sliders
        )
        self.iou_slider.set(0.40)
        self.iou_slider.grid(row=5, column=0, padx=20, pady=(5, 15))

        # Agnostic NMS Switch (跨類別抑制)
        self.agnostic_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="Agnostic NMS",
            command=self.run_inference
        )
        self.agnostic_switch.select()
        self.agnostic_switch.grid(row=6, column=0, padx=20, pady=(5, 10), sticky="w")

        # Grasp Pose Switch (預設開啟)
        self.grasp_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="Grasp Pose",
            command=self.run_inference
        )
        self.grasp_switch.select()
        self.grasp_switch.grid(row=7, column=0, padx=20, pady=(5, 15), sticky="w")

        self.load_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Load Random Image", 
            command=self.load_random_image,
            height=36,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.load_btn.grid(row=8, column=0, padx=20, pady=(5, 5))

        self.select_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Select Image File", 
            command=self.select_image_file,
            height=36,
            fg_color="#3a7ebf",
            hover_color="#326ba3",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.select_btn.grid(row=9, column=0, padx=20, pady=(5, 10))

        # Legend Box
        self.legend_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.legend_frame.grid(row=10, column=0, padx=20, pady=10, sticky="w")
        
        legend_items = [
            ("Plastic", "#2ecc71"),
            ("Metal", "#3498db"),
            ("Paper", "#e67e22"),
            ("General Waste", "#ff00ff")
        ]
        for idx, (c_name, c_hex) in enumerate(legend_items):
            lbl = ctk.CTkLabel(
                self.legend_frame, 
                text=f"■ {c_name}", 
                text_color=c_hex, 
                font=ctk.CTkFont(size=12, weight="bold")
            )
            lbl.grid(row=idx, column=0, sticky="w", pady=2)

        # --- Main View Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            self.main_frame, 
            bg="#1a1a1a", 
            highlightthickness=0
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        self.current_tk_image = None
        self.current_image_path = None
        self.raw_image_bgr = None

        # Load the custom trained model
        self.update()
        try:
            model_to_load = MODEL_PATH
            if not os.path.exists(model_to_load):
                alt_weights = [
                    "garbage_classification_runs/yolo11x_obb_model/weights/best.pt",
                    "yolo11x-obb.pt"
                ]
                for alt in alt_weights:
                    if os.path.exists(alt):
                        model_to_load = alt
                        break

            self.model = YOLO(model_to_load)
            self.model_status_label.configure(
                text=f"Ready: {os.path.basename(model_to_load)}", 
                text_color="#28a745"
            )
            self.load_random_image()
        except Exception as e:
            self.model_status_label.configure(text=f"Load Failed: {str(e)[:20]}", text_color="#dc3545")

    def update_sliders(self, value=None):
        conf_val = self.conf_slider.get()
        iou_val = self.iou_slider.get()
        self.conf_label.configure(text=f"Confidence: {conf_val:.2f}")
        self.iou_label.configure(text=f"Agnostic NMS IoU: {iou_val:.2f}")
        
        if self.current_image_path and self.raw_image_bgr is not None:
            self.run_inference()

    def load_image_from_path(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
            
        self.current_image_path = file_path
        filename = os.path.basename(self.current_image_path)
        
        if len(filename) > 22:
            display_name = filename[:19] + "..."
        else:
            display_name = filename
            
        self.model_status_label.configure(text=f"Image: {display_name}", text_color="gray70")
        
        # Read image supporting UTF-8 / non-ASCII paths
        try:
            img_array = np.fromfile(self.current_image_path, dtype=np.uint8)
            self.raw_image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            self.raw_image_bgr = None

        if self.raw_image_bgr is None:
            self.raw_image_bgr = cv2.imread(self.current_image_path)

        if self.raw_image_bgr is None:
            self.model_status_label.configure(text="Decode Error", text_color="#dc3545")
            return
            
        self.run_inference()

    def load_random_image(self):
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
            images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext.upper())))

        if not images:
            self.model_status_label.configure(text="No images in test/images", text_color="#dc3545")
            return
            
        selected = random.choice(images)
        self.load_image_from_path(selected)

    def select_image_file(self):
        initial_dir = os.path.abspath(TEST_IMAGES_DIR) if os.path.exists(TEST_IMAGES_DIR) else os.getcwd()
        filetypes = [
            ("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp *.JPG *.JPEG *.PNG *.WEBP *.BMP"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="選擇要推論的圖片檔案",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        if file_path:
            self.load_image_from_path(file_path)

    def run_inference(self):
        if not hasattr(self, 'model') or self.raw_image_bgr is None:
            return
            
        conf_thresh = self.conf_slider.get()
        iou_thresh = self.iou_slider.get()
        use_agnostic_nms = self.agnostic_switch.get() == 1
        show_grasp = self.grasp_switch.get() == 1
        
        # 1. Model inference
        results = self.model(self.raw_image_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        
        # 2. Extract OBB detections
        boxes_list = []
        if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
            xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
            xywhr = results.obb.xywhr.cpu().numpy()
            confs = results.obb.conf.cpu().numpy()
            clss = results.obb.cls.cpu().numpy().astype(int)
            
            for i in range(len(confs)):
                boxes_list.append({
                    'cls': clss[i],
                    'name': self.model.names[clss[i]],
                    'conf': float(confs[i]),
                    'pts': xyxyxyxy[i],
                    'xywhr': xywhr[i]
                })

        # 3. Apply Class-Agnostic NMS (消除跨類別重複框)
        if use_agnostic_nms and len(boxes_list) > 1:
            boxes_list = apply_agnostic_nms(boxes_list, iou_thresh=iou_thresh)

        # 4. Clean & Minimalist Rendering
        annotated = self.raw_image_bgr.copy()
        img_h, img_w = annotated.shape[:2]
        
        # Adaptive font scale and line thickness based on image resolution
        font_scale = max(0.85, min(1.4, img_w / 900.0))
        font_thick = max(2, int(round(font_scale * 2.0)))
        line_thick = max(2, int(round(img_w / 450.0)))
        
        for b in boxes_list:
            cls_id = b['cls']
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            pts = b['pts'].astype(np.int32).reshape((-1, 1, 2))
            
            # Clean oriented bounding box border
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=line_thick, lineType=cv2.LINE_AA)
            
            # Top-left corner point for badge
            top_pt = b['pts'][np.argmin(b['pts'][:, 1])]
            bx, by = int(top_pt[0]), int(top_pt[1])
            
            # Badge text: e.g. "plastic 0.91"
            badge_text = f" {b['name']} {b['conf']:.2f} "
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            # Draw badge background with neat padding
            pad_y = int(th * 0.35)
            badge_y1 = max(0, by - th - pad_y * 2)
            badge_y2 = badge_y1 + th + pad_y * 2
            badge_x1 = max(0, bx)
            badge_x2 = min(annotated.shape[1], badge_x1 + tw)
            
            # Draw solid badge background with slightly darker inner border
            # cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), 1)
            
            # White bold text inside badge
            text_y = badge_y2 - pad_y
            cv2.putText(annotated, badge_text, (badge_x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

            # Optional Grasp Pose Overlay
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

        self.show_image(annotated)

    def on_canvas_resize(self, event):
        if self.current_tk_image is not None and hasattr(self, 'last_annotated_img_rgb'):
            self.draw_on_canvas(self.last_annotated_img_rgb, event.width, event.height)

    def show_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.last_annotated_img_rgb = img_rgb
        
        self.update_idletasks()
        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600
            
        self.draw_on_canvas(img_rgb, canvas_w, canvas_h)

    def draw_on_canvas(self, img_rgb, canvas_w, canvas_h):
        img_h, img_w = img_rgb.shape[:2]
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(int(img_w * ratio), 1)
        new_h = max(int(img_h * ratio), 1)
        
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(resized)
        
        self.current_tk_image = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_w // 2, 
            canvas_h // 2, 
            anchor="center", 
            image=self.current_tk_image
        )

if __name__ == "__main__":
    app = App()
    app.mainloop()
