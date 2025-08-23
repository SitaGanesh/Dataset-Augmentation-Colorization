"""
GUI module for interactive image colorization.
Provides a simple interface for users to colorize images using trained models.
"""

import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml

import customtkinter as ctk
import torch
import numpy as np
from PIL import Image, ImageTk
from skimage import color

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the appearance mode and color theme
ctk.set_appearance_mode("light")   # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class ColorizationGUI:
    """
    Main GUI class for image colorization application.
    """

    def __init__(self):
        """Initialize the GUI application."""
        self.root = ctk.CTk()
        self.root.title("Image Colorization Tool")
        self.root.geometry("1200x800")

        # Initialize variables
        self.input_image = None
        self.colorized_image = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuration
        self.config_path = "config/config.yaml"
        self.load_config()

        # Setup GUI
        self.setup_gui()

        logger.info("GUI initialized successfully")

    def load_config(self):
        """Load configuration file from project root."""
        cwd = os.getcwd()
        project_root = os.path.dirname(cwd) if cwd.endswith("notebooks") else cwd

        config_path = os.path.join(project_root, "config", "config.yaml")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Config file not found at {config_path}: {e}. Using defaults.")
            self.config = {}

        # Ensure data and paths sections exist
        self.config.setdefault("data", {})
        self.config["data"].setdefault("input_size", [256, 256])

        self.config.setdefault("paths", {})
        # Always set absolute models_dir and results_dir
        self.config["paths"]["models_dir"] = os.path.join(project_root, "models")
        self.config["paths"]["results_dir"] = os.path.join(project_root, "results")

        # Store for later use
        self.config_path = config_path



    def setup_gui(self):
        """Setup the main GUI layout."""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.create_header_frame()
        self.create_main_frame()
        self.create_status_bar()

    def create_header_frame(self):
        """Create the header frame with controls."""
        header = ctk.CTkFrame(self.root)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(header, text="Image Colorization Tool", font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, columnspan=5, pady=10)

        ctk.CTkLabel(header, text="Model:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar()
        self.model_dropdown = ctk.CTkComboBox(
            header, variable=self.model_var, values=self.get_available_models(), command=self.load_model
        )
        self.model_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ctk.CTkButton(header, text="Load Image", command=self.load_image).grid(row=1, column=2, padx=5, pady=5)
        ctk.CTkButton(header, text="Colorize", command=self.colorize_image).grid(row=1, column=3, padx=5, pady=5)
        ctk.CTkButton(header, text="Save Result", command=self.save_result).grid(row=1, column=4, padx=5, pady=5)

    def create_main_frame(self):
        """Create the main content frame with image displays."""
        main = ctk.CTkFrame(self.root)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(1, weight=1)

        for idx, (label_text, attr) in enumerate(
            [("Input (Grayscale)", "input_canvas"), ("Output (Colorized)", "output_canvas")]
        ):
            frame = ctk.CTkFrame(main)
            frame.grid(row=0, column=idx, sticky="nsew", padx=5, pady=5)
            frame.grid_rowconfigure(1, weight=1)
            frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=5)
            canvas = tk.Canvas(frame, bg="white", relief="sunken", borderwidth=2)
            canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
            setattr(self, attr, canvas)

    def create_status_bar(self):
        """Create the status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status = ctk.CTkLabel(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", padx=10, pady=2)

    def get_available_models(self):
        """Return list of .pth files under models_dir and its subfolders."""
        models_dir = self.config["paths"]["models_dir"]
        if not os.path.isdir(models_dir):
            return ["No models found"]
        models = []
        for root, _, files in os.walk(models_dir):
            for f in files:
                if f.endswith(".pth"):
                    rel = os.path.relpath(os.path.join(root, f), models_dir)
                    models.append(rel)
        return models or ["No models found"]

    def load_model(self, _=None):
        """Load a selected model."""
        name = self.model_var.get()
        if name == "No models found":
            messagebox.showerror("Error", "No trained models available.")
            return
        self.status_var.set(f"Loading model: {name}...")
        self.root.update()
        try:
            from model_architecture import create_model
            path = os.path.join(self.config["paths"]["models_dir"], name)
            checkpoint = torch.load(path, map_location=self.device)
            self.model = create_model(self.config_path).to(self.device)
            state = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state)
            self.model.eval()
            self.status_var.set(f"Model loaded: {name}")
            logger.info(f"Loaded model: {path}")
        except Exception as e:
            self.status_var.set("Error loading model")
            messagebox.showerror("Error", f"Error loading model:\n{e}")
            logger.error(e)

    def load_image(self):
        """Load an image file."""
        file = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
        )
        if not file:
            return
        try:
            self.status_var.set("Loading image...")
            self.root.update()
            img = Image.open(file).convert("RGB")
            self.input_image = img
            self.display_input_image(img)
            self.status_var.set(f"Image loaded: {os.path.basename(file)}")
            logger.info(f"Loaded image: {file}")
        except Exception as e:
            self.status_var.set("Error loading image")
            messagebox.showerror("Error", f"Error loading image:\n{e}")
            logger.error(e)

    def display_input_image(self, image):
        """Show grayscale input."""
        gray = image.convert("L")
        w, h = gray.size
        cw, ch = self.input_canvas.winfo_width(), self.input_canvas.winfo_height()
        if cw > 1 and ch > 1:
            scale = min(cw / w, ch / h)
            gray = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        self.input_photo = ImageTk.PhotoImage(gray)
        self.input_canvas.delete("all")
        self.input_canvas.create_image(cw // 2, ch // 2, image=self.input_photo)

    def preprocess_image(self, image):
        """Convert RGB→LAB and return normalized L channel tensor."""
        from data_preprocessing import DataPreprocessor

        size = tuple(self.config["data"]["input_size"])
        img = image.resize(size, Image.Resampling.LANCZOS)
        arr = np.array(img)
        lab = color.rgb2lab(arr).astype(np.float32)
        L = lab[:, :, 0] / 100.0
        L = (L - 0.5) / 0.5
        return torch.from_numpy(L).unsqueeze(0).unsqueeze(0)

    def postprocess_image(self, L, AB):
        """Convert predicted LAB back to RGB numpy array."""
        from evaluation import ColorizationEvaluator

        ev = ColorizationEvaluator(self.config_path)
        return ev.lab_to_rgb(L, AB)[0]

    def colorize_image(self):
        """Run model inference and display colorized result."""
        if not self.input_image:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first.")
            return
        try:
            self.status_var.set("Colorizing image...")
            self.root.update()
            L = self.preprocess_image(self.input_image).to(self.device)
            with torch.no_grad():
                AB = self.model(L).cpu()
            rgb = self.postprocess_image(L.cpu(), AB)
            img = Image.fromarray((rgb * 255).astype(np.uint8))
            self.colorized_image = img
            self.display_output_image(img)
            self.status_var.set("Image colorized successfully!")
            logger.info("Image colorization completed")
        except Exception as e:
            self.status_var.set("Error during colorization")
            messagebox.showerror("Error", f"Error during colorization:\n{e}")
            logger.error(e)

    def display_output_image(self, image):
        """Show the colorized output."""
        w, h = image.size
        cw, ch = self.output_canvas.winfo_width(), self.output_canvas.winfo_height()
        if cw > 1 and ch > 1:
            scale = min(cw / w, ch / h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        self.output_photo = ImageTk.PhotoImage(image)
        self.output_canvas.delete("all")
        self.output_canvas.create_image(cw // 2, ch // 2, image=self.output_photo)

    def save_result(self):
        """Save the colorized image to disk."""
        if not self.colorized_image:
            messagebox.showwarning("Warning", "No colorized image to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save colorized image", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            self.colorized_image.save(path)
            self.status_var.set(f"Image saved: {os.path.basename(path)}")
            messagebox.showinfo("Success", "Image saved successfully!")
            logger.info(f"Saved image: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving image:\n{e}")
            logger.error(e)

    def run(self):
        """Start the GUI."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(100, lambda: None)  # Ensure canvases are initialized
        self.root.mainloop()

    def on_closing(self):
        logger.info("Closing GUI application")
        self.root.destroy()


class SimpleColorizationGUI:
    """
    Simplified GUI using tkinter only, for environments without CustomTkinter.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Colorization Tool")
        self.root.geometry("800x600")
        self.input_image = None
        self.photo = None
        self.status_var = tk.StringVar(value="Ready")
        self.setup_simple_gui()

    def setup_simple_gui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        ttk.Label(main, text="Image Colorization Tool", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Load Image", command=self.load_image_simple).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Colorize", command=self.colorize_simple).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Result", command=self.save_simple).pack(side=tk.LEFT, padx=5)

        self.image_label = ttk.Label(main, text="Load an image to start")
        self.image_label.grid(row=2, column=0, columnspan=3, pady=20)
        ttk.Label(main, textvariable=self.status_var).grid(row=3, column=0, columnspan=3)

    def load_image_simple(self):
        path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if not path:
            return
        img = Image.open(path)
        img.thumbnail((300, 300))
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo, text="")
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def colorize_simple(self):
        messagebox.showinfo("Info", "Colorization feature requires the full GUI with a trained model.")

    def save_simple(self):
        messagebox.showinfo("Info", "Save feature will be available after colorization.")

    def run(self):
        self.root.mainloop()


def main():
    try:
        app = ColorizationGUI()
        app.run()
    except Exception as e:
        logger.warning(f"Advanced GUI failed or not available: {e}")
        app = SimpleColorizationGUI()
        app.run()


if __name__ == "__main__":
    main()