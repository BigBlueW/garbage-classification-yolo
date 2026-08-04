import os
import random
import glob
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from ultralytics import YOLO

# Setup CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Config Paths
MODEL_PATH = "best.pt"
TEST_IMAGES_DIR = "custom_dataset/test/images"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Garbage Classification Inference Engine")
        self.geometry("1100x750")
        
        # Configure Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Garbage Vision", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))

        self.model_status_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Loading Model...", 
            text_color="gray70",
            font=ctk.CTkFont(size=13)
        )
        self.model_status_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Confidence Slider
        self.conf_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Confidence: 0.25", 
            anchor="w"
        )
        self.conf_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.conf_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0.05, 
            to=0.95, 
            number_of_steps=90, 
            command=self.update_sliders
        )
        self.conf_slider.set(0.25)
        self.conf_slider.grid(row=3, column=0, padx=20, pady=(10, 20))

        # IoU Slider
        self.iou_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="NMS IoU: 0.45", 
            anchor="w"
        )
        self.iou_label.grid(row=4, column=0, padx=20, pady=(0, 0), sticky="w")
        
        self.iou_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0.05, 
            to=0.95, 
            number_of_steps=90, 
            command=self.update_sliders
        )
        self.iou_slider.set(0.45)
        self.iou_slider.grid(row=5, column=0, padx=20, pady=(10, 20))

        self.load_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Load Random Image", 
            command=self.load_random_image,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.load_btn.grid(row=6, column=0, padx=20, pady=20)
        
        # --- Main View Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            self.main_frame, 
            bg="#212121", 
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
            self.model = YOLO(MODEL_PATH)
            
            # Map original 6 classes to 4 target classes for the robotic system ONLY if it's the old 6-class model
            if len(self.model.model.names) == 6:
                self.model.model.names = {
                    0: "general_waste",
                    1: "paper",
                    2: "general_waste",
                    3: "metal",
                    4: "paper",
                    5: "plastic"
                }
            
            self.model_status_label.configure(
                text="System Ready", 
                text_color="#28a745"
            )
            self.load_random_image()
        except Exception as e:
            self.model_status_label.configure(text="Model Load Failed", text_color="#dc3545")

    def update_sliders(self, value=None):
        conf_val = self.conf_slider.get()
        iou_val = self.iou_slider.get()
        self.conf_label.configure(text=f"Confidence: {conf_val:.2f}")
        self.iou_label.configure(text=f"NMS IoU: {iou_val:.2f}")
        
        # Re-run inference on the same image if threshold changes
        if self.current_image_path and self.raw_image_bgr is not None:
            self.run_inference()

    def load_random_image(self):
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
            images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext.upper())))

        if not images:
            self.model_status_label.configure(text="No images found", text_color="#dc3545")
            return
            
        self.current_image_path = random.choice(images)
        filename = os.path.basename(self.current_image_path)
        
        # Truncate long filenames for UI
        if len(filename) > 20:
            display_name = filename[:17] + "..."
        else:
            display_name = filename
            
        self.model_status_label.configure(text=f"Image: {display_name}", text_color="gray70")
        
        self.raw_image_bgr = cv2.imread(self.current_image_path)
        if self.raw_image_bgr is None:
            self.model_status_label.configure(text="Decode Error", text_color="#dc3545")
            return
            
        self.run_inference()

    def run_inference(self):
        if not hasattr(self, 'model') or self.raw_image_bgr is None:
            return
            
        conf_thresh = self.conf_slider.get()
        iou_thresh = self.iou_slider.get()
        
        results = self.model(self.raw_image_bgr, conf=conf_thresh, iou=iou_thresh, agnostic_nms=True, verbose=False)[0]
        
        annotated_img = results.plot(line_width=2, font_size=16)
        self.show_image(annotated_img)

    def on_canvas_resize(self, event):
        # Triggered when window is resized to keep image scaled properly
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
