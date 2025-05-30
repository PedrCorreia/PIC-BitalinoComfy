#!/usr/bin/env python3
#PYTHONPATH=/home/lugo/ComfyUI/custom_nodes /home/lugo/miniconda3/envs/PEDRO/bin/python /home/lugo/ComfyUI/custom_nodes/PIC-BitalinoComfy/sdxl_turbo_controlnet_depth.py
"""
SDXL Turbo + ControlNet Sphere Generator
Modern dark-themed UI for sphere-based diffusion control
"""

import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import torch
from rtd_comfy.sdxl_turbo.diffusion_engine import DiffusionEngine


class ModernDarkTheme:
    # Dark theme colors
    BG_DARK = "#1e1e1e"
    BG_MEDIUM = "#2d2d2d" 
    BG_LIGHT = "#3e3e3e"
    FG_PRIMARY = "#ffffff"
    FG_SECONDARY = "#cccccc"
    ACCENT_BLUE = "#007acc"
    ACCENT_GREEN = "#4caf50"
    ACCENT_ORANGE = "#ff9800"
    BORDER = "#404040"

class SDXLTurboSphereGenerator:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_modern_theme()
        self.root.title("SDXL Turbo Sphere Generator")
        self.root.geometry("1400x900")
        self.root.configure(bg=ModernDarkTheme.BG_DARK)

        # Pipeline variables
        self.diffusion_engine = None
        self.current_sphere = None
        self.depth_map = None
        self.generated_image = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = tk.StringVar(value="stabilityai/sdxl-turbo")
        self.generation_lock = threading.Lock()

        # Sphere parameters
        self.prompt = tk.StringVar(value="a futuristic crystalline sphere, highly detailed, 8k, photorealistic")
        self.negative_prompt = tk.StringVar(value="blurry, low quality, distorted, flat")
        self.sphere_radius = tk.DoubleVar(value=0.8)
        self.sphere_depth = tk.DoubleVar(value=0.6)
        self.surface_roughness = tk.DoubleVar(value=0.1)
        self.lighting_angle = tk.DoubleVar(value=45.0)
        self.metallic = tk.DoubleVar(value=0.5)
        self.depth_strength = tk.DoubleVar(value=1.2)
        self.guidance_scale = tk.DoubleVar(value=7.5)
        self.num_steps = tk.IntVar(value=4)

        # UI state
        self.processing = False
        self.auto_generate = tk.BooleanVar(value=True)

        self.setup_ui()
        self.setup_pipeline()

    def setup_modern_theme(self):
        """Configure modern dark theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme
        style.configure('TLabel', 
                       background=ModernDarkTheme.BG_DARK,
                       foreground=ModernDarkTheme.FG_PRIMARY,
                       font=('Segoe UI', 10))
        
        style.configure('TFrame',
                       background=ModernDarkTheme.BG_DARK,
                       borderwidth=0)
        
        style.configure('TLabelFrame',
                       background=ModernDarkTheme.BG_DARK,
                       foreground=ModernDarkTheme.FG_PRIMARY,
                       borderwidth=1,
                       relief='solid',
                       bordercolor=ModernDarkTheme.BORDER)
        
        style.configure('TButton',
                       background=ModernDarkTheme.ACCENT_BLUE,
                       foreground=ModernDarkTheme.FG_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('TButton',
                 background=[('active', '#005a9e'),
                            ('pressed', '#004080')])
        
        style.configure('TEntry',
                       background=ModernDarkTheme.BG_LIGHT,
                       foreground=ModernDarkTheme.FG_PRIMARY,
                       borderwidth=1,
                       bordercolor=ModernDarkTheme.BORDER,
                       insertcolor=ModernDarkTheme.FG_PRIMARY)
        
        style.configure('TScale',
                       background=ModernDarkTheme.BG_DARK,
                       troughcolor=ModernDarkTheme.BG_LIGHT,
                       borderwidth=0,
                       sliderlength=20)
        
        style.configure('TNotebook',
                       background=ModernDarkTheme.BG_DARK,
                       borderwidth=0)
        
        style.configure('TNotebook.Tab',
                       background=ModernDarkTheme.BG_MEDIUM,
                       foreground=ModernDarkTheme.FG_SECONDARY,
                       padding=[20, 10])
        
        style.map('TNotebook.Tab',
                 background=[('selected', ModernDarkTheme.ACCENT_BLUE)],
                 foreground=[('selected', ModernDarkTheme.FG_PRIMARY)])

    def setup_ui(self):
        """Setup the modern dark UI."""
        # Main container with dark background
        main_frame = tk.Frame(self.root, bg=ModernDarkTheme.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="SDXL Turbo Sphere Generator",
                              font=('Segoe UI', 24, 'bold'),
                              bg=ModernDarkTheme.BG_DARK,
                              fg=ModernDarkTheme.FG_PRIMARY)
        title_label.pack(pady=(0, 20))
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=ModernDarkTheme.BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        self.setup_control_panel(content_frame)
        
        # Right panel - Preview
        self.setup_preview_panel(content_frame)
        
        # Bottom status bar
        self.setup_status_bar(main_frame)
        
        # Generate initial sphere
        self.root.after(1000, self.generate_sphere_depth)

    def setup_control_panel(self, parent):
        """Setup the control panel."""
        control_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM, relief='solid', bd=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=0)
        control_frame.configure(width=400)
        control_frame.pack_propagate(False)
        
        # Title
        control_title = tk.Label(control_frame,
                                text="Sphere Controls",
                                font=('Segoe UI', 16, 'bold'),
                                bg=ModernDarkTheme.BG_MEDIUM,
                                fg=ModernDarkTheme.FG_PRIMARY)
        control_title.pack(pady=20)
        
        # Scrollable content
        canvas = tk.Canvas(control_frame, bg=ModernDarkTheme.BG_MEDIUM, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=ModernDarkTheme.BG_MEDIUM)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Prompt section
        self.create_prompt_section(scrollable_frame)
        
        # Sphere parameters
        self.create_sphere_controls(scrollable_frame)
        
        # Generation controls
        self.create_generation_controls(scrollable_frame)
        
        # Model path entry
        model_frame = tk.Frame(scrollable_frame, bg=ModernDarkTheme.BG_MEDIUM)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(model_frame, text="Model/Checkpoint Path:", bg=ModernDarkTheme.BG_MEDIUM, fg=ModernDarkTheme.FG_PRIMARY, font=('Segoe UI', 10)).pack(side=tk.LEFT)
        model_entry = tk.Entry(model_frame, textvariable=self.model_path, width=32, bg=ModernDarkTheme.BG_LIGHT, fg=ModernDarkTheme.FG_PRIMARY)
        model_entry.pack(side=tk.LEFT, padx=(8, 0))
        reload_btn = tk.Button(model_frame, text="Reload Model", command=self.reload_model, bg=ModernDarkTheme.ACCENT_ORANGE, fg=ModernDarkTheme.FG_PRIMARY, font=('Segoe UI', 10, 'bold'), relief='flat', bd=0, padx=10, pady=2)
        reload_btn.pack(side=tk.LEFT, padx=(8, 0))
        
        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")

    def create_prompt_section(self, parent):
        """Create prompt input section."""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 20))
        
        # Prompt
        tk.Label(section, text="Prompt:", 
                font=('Segoe UI', 12, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.FG_PRIMARY).pack(anchor=tk.W, pady=(0, 5))
        
        prompt_text = tk.Text(section, height=3, width=40,
                             bg=ModernDarkTheme.BG_LIGHT,
                             fg=ModernDarkTheme.FG_PRIMARY,
                             insertbackground=ModernDarkTheme.FG_PRIMARY,
                             font=('Segoe UI', 10))
        prompt_text.pack(fill=tk.X, pady=(0, 10))
        prompt_text.insert('1.0', self.prompt.get())
        
        def update_prompt(*args):
            self.prompt.set(prompt_text.get('1.0', 'end-1c'))
        prompt_text.bind('<KeyRelease>', update_prompt)
        
        # Negative prompt
        tk.Label(section, text="Negative Prompt:", 
                font=('Segoe UI', 12, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.FG_PRIMARY).pack(anchor=tk.W, pady=(0, 5))
        
        neg_prompt_text = tk.Text(section, height=2, width=40,
                                 bg=ModernDarkTheme.BG_LIGHT,
                                 fg=ModernDarkTheme.FG_PRIMARY,
                                 insertbackground=ModernDarkTheme.FG_PRIMARY,
                                 font=('Segoe UI', 10))
        neg_prompt_text.pack(fill=tk.X, pady=(0, 15))
        neg_prompt_text.insert('1.0', self.negative_prompt.get())
        
        def update_neg_prompt(*args):
            self.negative_prompt.set(neg_prompt_text.get('1.0', 'end-1c'))
        neg_prompt_text.bind('<KeyRelease>', update_neg_prompt)

    def create_sphere_controls(self, parent):
        """Create sphere parameter controls."""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(section, text="Sphere Parameters", 
                font=('Segoe UI', 14, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.ACCENT_BLUE).pack(anchor=tk.W, pady=(0, 15))
        
        # Create sliders for sphere parameters
        self.create_modern_slider(section, "Radius", self.sphere_radius, 0.1, 1.0)
        self.create_modern_slider(section, "Depth", self.sphere_depth, 0.1, 1.0)
        self.create_modern_slider(section, "Surface Roughness", self.surface_roughness, 0.0, 0.5)
        self.create_modern_slider(section, "Lighting Angle", self.lighting_angle, 0.0, 90.0)
        self.create_modern_slider(section, "Metallic", self.metallic, 0.0, 1.0)

    def create_generation_controls(self, parent):
        """Create generation parameter controls."""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(section, text="Generation Parameters", 
                font=('Segoe UI', 14, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.ACCENT_GREEN).pack(anchor=tk.W, pady=(0, 15))
        
        self.create_modern_slider(section, "Depth Strength", self.depth_strength, 0.5, 2.0)
        self.create_modern_slider(section, "Guidance Scale", self.guidance_scale, 1.0, 15.0)
        self.create_modern_slider(section, "Steps", self.num_steps, 1, 8, is_int=True)
        
        # Auto-generate checkbox
        auto_frame = tk.Frame(section, bg=ModernDarkTheme.BG_MEDIUM)
        auto_frame.pack(fill=tk.X, pady=(15, 10))
        
        auto_check = tk.Checkbutton(auto_frame,
                                   text="Auto-generate on parameter change",
                                   variable=self.auto_generate,
                                   bg=ModernDarkTheme.BG_MEDIUM,
                                   fg=ModernDarkTheme.FG_PRIMARY,
                                   selectcolor=ModernDarkTheme.BG_LIGHT,
                                   activebackground=ModernDarkTheme.BG_MEDIUM,
                                   activeforeground=ModernDarkTheme.FG_PRIMARY,
                                   font=('Segoe UI', 10))
        auto_check.pack(anchor=tk.W)
        
        # Generate button
        generate_btn = tk.Button(section,
                                text="Generate Sphere",
                                command=self.generate_image,
                                bg=ModernDarkTheme.ACCENT_BLUE,
                                fg=ModernDarkTheme.FG_PRIMARY,
                                font=('Segoe UI', 12, 'bold'),
                                relief='flat',
                                bd=0,
                                padx=20,
                                pady=10)
        generate_btn.pack(pady=(15, 10))
        
        # Save button  
        save_btn = tk.Button(section,
                            text="Save Result",
                            command=self.save_result,
                            bg=ModernDarkTheme.ACCENT_GREEN,
                            fg=ModernDarkTheme.FG_PRIMARY,
                            font=('Segoe UI', 11, 'bold'),
                            relief='flat',
                            bd=0,
                            padx=20,
                            pady=8)
        save_btn.pack(pady=5)
    def create_modern_slider(self, parent, label, variable, min_val, max_val, is_int=False):
        """Create a modern styled slider."""
        slider_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        slider_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Label with value display
        label_frame = tk.Frame(slider_frame, bg=ModernDarkTheme.BG_MEDIUM)
        label_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(label_frame, text=f"{label}:",
                font=('Segoe UI', 11, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.FG_PRIMARY).pack(side=tk.LEFT)
        
        value_label = tk.Label(label_frame, text=f"{variable.get():.2f}",
                              font=('Segoe UI', 11),
                              bg=ModernDarkTheme.BG_MEDIUM,
                              fg=ModernDarkTheme.ACCENT_BLUE)
        value_label.pack(side=tk.RIGHT)
        
        # Slider
        if is_int:
            slider = tk.Scale(slider_frame, from_=min_val, to=max_val, variable=variable,
                             orient=tk.HORIZONTAL, length=300,
                             bg=ModernDarkTheme.BG_MEDIUM,
                             fg=ModernDarkTheme.FG_PRIMARY,
                             highlightthickness=0,
                             troughcolor=ModernDarkTheme.BG_LIGHT,
                             activebackground=ModernDarkTheme.ACCENT_BLUE,
                             font=('Segoe UI', 9))
        else:
            slider = tk.Scale(slider_frame, from_=min_val, to=max_val, variable=variable,
                             orient=tk.HORIZONTAL, length=300, resolution=0.01,
                             bg=ModernDarkTheme.BG_MEDIUM,
                             fg=ModernDarkTheme.FG_PRIMARY,
                             highlightthickness=0,
                             troughcolor=ModernDarkTheme.BG_LIGHT,
                             activebackground=ModernDarkTheme.ACCENT_BLUE,
                             font=('Segoe UI', 9))
        
        slider.pack(fill=tk.X)
        
        # Update value label and auto-generate
        def on_change(*args):
            if is_int:
                value_label.config(text=f"{int(variable.get())}")
            else:
                value_label.config(text=f"{variable.get():.2f}")
            
            if self.auto_generate.get() and not self.processing:
                self.root.after(100, self.generate_sphere_depth)
                self.root.after(200, self.generate_image)
        
        variable.trace_add('write', on_change)
        
        return slider

    def setup_preview_panel(self, parent):
        """Setup the preview panel."""
        preview_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM, relief='solid', bd=1)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Title
        preview_title = tk.Label(preview_frame,
                                text="Live Preview",
                                font=('Segoe UI', 16, 'bold'),
                                bg=ModernDarkTheme.BG_MEDIUM,
                                fg=ModernDarkTheme.FG_PRIMARY)
        preview_title.pack(pady=20)
        
        # Create notebook for tabbed view
        self.notebook = ttk.Notebook(preview_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Sphere depth tab
        self.depth_frame = tk.Frame(self.notebook, bg=ModernDarkTheme.BG_DARK)
        self.notebook.add(self.depth_frame, text="Sphere Depth")
        
        self.depth_canvas = tk.Canvas(self.depth_frame, 
                                     bg=ModernDarkTheme.BG_DARK,
                                     highlightthickness=0,
                                     width=600, height=600)
        self.depth_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generated image tab
        self.result_frame = tk.Frame(self.notebook, bg=ModernDarkTheme.BG_DARK)
        self.notebook.add(self.result_frame, text="Generated Image")
        
        self.result_canvas = tk.Canvas(self.result_frame,
                                      bg=ModernDarkTheme.BG_DARK,
                                      highlightthickness=0,
                                      width=600, height=600)
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_status_bar(self, parent):
        """Setup status bar."""
        status_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_LIGHT, height=30)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(status_frame,
                               textvariable=self.status_var,
                               bg=ModernDarkTheme.BG_LIGHT,
                               fg=ModernDarkTheme.FG_SECONDARY,
                               font=('Segoe UI', 10))
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Progress indicator
        self.progress_frame = tk.Frame(status_frame, bg=ModernDarkTheme.BG_LIGHT)
        self.progress_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Create the actual progress bar widget
        self.progress = ttk.Progressbar(self.progress_frame, 
                                      mode='indeterminate',
                                      length=100)
        self.progress.pack()

    def generate_sphere_depth(self):
        """Generate a 3D sphere depth map using torch for speed and GPU support."""
        try:
            size = 512
            device = self.device
            radius = self.sphere_radius.get()
            depth = self.sphere_depth.get()
            roughness = self.surface_roughness.get()
            lighting_angle = self.lighting_angle.get()
            metallic = self.metallic.get()

            # Torch-based sphere depth map
            y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
            center = size // 2
            dx = x - center
            dy = y - center
            dist = torch.sqrt(dx.float()**2 + dy.float()**2)
            sphere_r = center * radius
            mask = dist <= sphere_r
            norm_dist = dist / sphere_r
            sphere_height = torch.zeros_like(dist)
            sphere_height[mask] = torch.sqrt(1 - norm_dist[mask]**2)
            depth_map = sphere_height * depth
            # Add surface roughness
            if roughness > 0:
                noise = torch.randn_like(depth_map) * roughness * 0.2
                depth_map = depth_map + noise * mask
            # Lighting
            light_angle_rad = torch.tensor(lighting_angle * 3.14159265 / 180, device=device)
            light_x = torch.cos(light_angle_rad)
            light_y = torch.sin(light_angle_rad)
            normal_x = dx / (sphere_r + 1e-6)
            normal_y = dy / (sphere_r + 1e-6)
            normal_z = torch.sqrt(1 - normal_x**2 - normal_y**2)
            normal_z[~mask] = 0
            lighting = (normal_x * light_x + normal_y * light_y + normal_z * 1.0)
            lighting = (lighting - lighting.min()) / (lighting.max() - lighting.min() + 1e-6)
            depth_map = depth_map * (0.7 + 0.3 * lighting)
            # Metallic effect
            if metallic > 0:
                metallic_mask = depth_map > (depth_map.max() * 0.7)
                depth_map[metallic_mask] *= (1 + metallic * 0.3)
            # Normalize
            depth_map = depth_map.clamp(0, 1)
            depth_map = (depth_map * 255).to(torch.uint8).cpu().numpy()
            self.depth_map = Image.fromarray(depth_map).convert("RGB")
            self.display_image(self.depth_map, self.depth_canvas)
            self.status_var.set("Sphere depth map generated (torch)")
        except RuntimeError as e:
            self.status_var.set(f"Torch error: {str(e)}")
            messagebox.showerror("Torch Error", str(e))
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def display_image(self, image, canvas):
        """Display an image in the specified canvas."""
        if image is None:
            return
        
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready, try again later
            self.root.after(100, lambda: self.display_image(image, canvas))
            return
        
        # Resize image to fit canvas while maintaining aspect ratio
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and add image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        
        # Keep a reference to prevent garbage collection
        canvas.image = photo
    
    def setup_pipeline(self):
        self.status_var.set("Loading models...")
        self.progress.start()
        def load_models():
            try:
                self.diffusion_engine = DiffusionEngine(
                    use_image2image=False,
                    height_diffusion_desired=512,
                    width_diffusion_desired=512,
                    device=self.device,
                    hf_model=self.model_path.get()
                )
                self.progress.stop()
                self.status_var.set("Ready - Sphere only mode. Image generation available.")
            except Exception as e:
                self.progress.stop()
                self.status_var.set(f"Error loading models: {str(e)}")
                messagebox.showerror("Model Loading Error", str(e))
        threading.Thread(target=load_models, daemon=True).start()

    def reload_model(self):
        if self.processing:
            messagebox.showinfo("Busy", "Please wait for current generation to finish.")
            return
        self.setup_pipeline()

    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and resize image
                self.current_image = Image.open(file_path).convert("RGB")
                
                # Resize for consistent processing (SDXL works well with 1024x1024)
                self.current_image = self.current_image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Display original image
                self.display_image(self.current_image, self.original_canvas)
                
                # Generate depth map
                self.generate_depth_map()
                
                self.status_var.set(f"Image loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def generate_depth_map(self):
        """Generate depth map from current image."""
        if self.current_image is None or self.depth_estimator is None:
            return
        
        self.status_var.set("Generating depth map...")
        
        def process_depth():
            try:
                # Generate depth map
                depth_result = self.depth_estimator(self.current_image)
                depth_image = depth_result["depth"]
                
                # Convert to numpy and normalize
                depth_array = np.array(depth_image)
                depth_normalized = ((depth_array - depth_array.min()) / 
                                  (depth_array.max() - depth_array.min() * 1.0) * 255).astype(np.uint8)
                
                # Apply radius and deformation transformations
                depth_transformed = self.apply_transformations(depth_normalized)
                
                # Convert back to PIL Image
                self.depth_map = Image.fromarray(depth_transformed).convert("RGB")
                
                # Display depth map
                self.display_image(self.depth_map, self.depth_canvas)
                
                self.status_var.set("Depth map generated")
                
            except Exception as e:
                self.status_var.set(f"Error generating depth map: {str(e)}")
        
        threading.Thread(target=process_depth, daemon=True).start()
    
    def apply_transformations(self, depth_array):
        """Apply radius and deformation transformations to depth map."""
        height, width = depth_array.shape
        center_x, center_y = width // 2, height // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalize distance
        normalized_distance = distance / max_distance
        
        # Apply radius effect
        radius_val = self.radius.get()
        if radius_val > 0:
            radius_mask = np.exp(-normalized_distance / radius_val)
            depth_array = depth_array * radius_mask
        
        # Apply deformation effect
        deformation_val = self.deformation.get()
        if deformation_val > 0:
            # Radial deformation
            angle = np.arctan2(y - center_y, x - center_x)
            radial_effect = np.sin(normalized_distance * np.pi * deformation_val * 4)
            
            # Apply sinusoidal deformation
            deformation_factor = 1 + radial_effect * deformation_val * 0.3
            depth_array = depth_array * deformation_factor
        
        # Ensure values stay in valid range
        depth_array = np.clip(depth_array, 0, 255)
        
        return depth_array.astype(np.uint8)
    
    def move_to_cpu(self, obj):
        """Recursively move all torch tensors in obj to CPU."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self.move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.move_to_cpu(v) for v in obj)
        else:
            return obj
    
    def generate_image(self):
        if self.diffusion_engine is None or self.depth_map is None:
            messagebox.showwarning("Warning", "Model not loaded or sphere not generated yet.")
            return
        if self.processing or self.generation_lock.locked():
            return
        def run_generation():
            with self.generation_lock:
                self.processing = True
                self.progress.start()
                self.status_var.set("Generating image...")
                try:
                    depth_tensor = torch.from_numpy(np.array(self.depth_map.convert('L'))).float() / 255.0
                    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
                    depth_tensor = depth_tensor.to(self.device)
                    # Always move to CPU and convert to numpy before passing to diffusion_engine
                    arr = depth_tensor.cpu().numpy()
                    if arr.shape[0] == 1 and arr.shape[1] == 1:
                        arr = arr[0, 0]  # shape now (512, 512)
                    self.diffusion_engine.set_input_image(arr)
                    self.diffusion_engine.set_num_inference_steps(self.num_steps.get())
                    self.diffusion_engine.set_guidance_scale(self.guidance_scale.get())
                    self.diffusion_engine.set_strength(self.depth_strength.get())
                    prompt = self.prompt.get()
                    negative_prompt = self.negative_prompt.get()
                    
                    # Set positive prompt
                    self.diffusion_engine.set_embeddings([prompt, prompt, prompt, prompt])
                    
                    # Store negative prompt array for use in pipeline
                    neg_prompt_array = [negative_prompt, negative_prompt, negative_prompt, negative_prompt]
                    
                    # Try to set negative embeddings if method exists
                    if hasattr(self.diffusion_engine, 'set_negative_embeddings'):
                        self.diffusion_engine.set_negative_embeddings(neg_prompt_array)
                    
                    # Store on diffusion engine for use in the patched pipeline
                    self.diffusion_engine.negative_prompt_array = neg_prompt_array
                    
                    # --- Apply direct monkeypatch to pipeline's check_inputs method ---
                    pipe = getattr(self.diffusion_engine, 'pipe', None)
                    if pipe is not None and not hasattr(pipe, '_check_inputs_patched'):
                        original_check_inputs = pipe.check_inputs
                        
                        # Create a new defensive check_inputs that prevents string comparison with tensors
                        def patched_check_inputs(self_pipe, prompt=None, height=None, width=None, *args, **kwargs):
                            # Remove negative_prompt entirely if prompt_embeds exists to avoid shape comparison
                            if 'prompt_embeds' in kwargs and 'negative_prompt' in kwargs:
                                kwargs.pop('negative_prompt')
                            
                            # Also remove negative_prompt if it wasn't removed already but exists
                            if 'negative_prompt' in kwargs and kwargs['negative_prompt'] is not None:
                                # Store it at pipeline level to reuse in __call__
                                neg_prompt = kwargs.pop('negative_prompt')
                                if isinstance(neg_prompt, str):
                                    neg_prompt = [neg_prompt] * 4
                                self_pipe._stored_negative_prompt = neg_prompt
                            
                            # Call original with cleaned kwargs
                            return original_check_inputs(prompt, height, width, *args, **kwargs)
                        
                        # Replace the check_inputs method
                        pipe.check_inputs = patched_check_inputs.__get__(pipe, type(pipe))
                        pipe._check_inputs_patched = True
                        
                        # Now patch __call__ to reapply the negative prompt after checks
                        original_call = pipe.__call__
                        
                        def patched_call(self_pipe, *args, **kwargs):
                            # If we have a stored negative prompt from check_inputs, use it
                            if hasattr(self_pipe, '_stored_negative_prompt'):
                                kwargs['negative_prompt'] = self_pipe._stored_negative_prompt
                                
                            # Always set our negative prompt array from the diffusion engine if available
                            if hasattr(self.diffusion_engine, 'negative_prompt_array'):
                                kwargs['negative_prompt'] = self.diffusion_engine.negative_prompt_array
                                
                            return original_call(*args, **kwargs)
                        
                        pipe.__call__ = patched_call.__get__(pipe, type(pipe))
                    # ------------------------------------------------------------------------

                    result = self.diffusion_engine.generate()
                    # Move all tensors in result to CPU before any numpy conversion
                    result = self.move_to_cpu(result)
                    if isinstance(result, list):
                        result = result[0]
                    if isinstance(result, torch.Tensor):
                        result = result.detach().cpu().clamp(0, 1)
                        result = (result * 255).to(torch.uint8).numpy().transpose(1, 2, 0)
                        result = Image.fromarray(result)
                    self.generated_image = result
                    self.display_image(self.generated_image, self.result_canvas)
                    self.notebook.select(1)  # Changed to ensure the right tab is selected
                    self.status_var.set("Image generated successfully")
                except RuntimeError as e:
                    self.status_var.set(f"Torch error: {str(e)}")
                    messagebox.showerror("Torch Error", str(e))
                    import traceback
                    traceback.print_exc()
                except Exception as e:
                    self.status_var.set(f"Error generating image: {str(e)}")
                    messagebox.showerror("Generation Error", str(e))
                    import traceback
                    traceback.print_exc()
                finally:
                    self.processing = False
                    self.progress.stop()
        threading.Thread(target=run_generation, daemon=True).start()
    
    def display_image(self, pil_image, canvas):
        """Display PIL image on canvas."""
        # Get canvas size
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 400, 400
        
        # Resize image to fit canvas while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        
        # Keep a reference to prevent garbage collection
        canvas.image = photo
    
    def save_result(self):
        """Save the generated result."""
        if self.generated_image is None:
            messagebox.showwarning("Warning", "No generated image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Generated Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.generated_image.save(file_path)
                self.status_var.set(f"Image saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def run(self):
        """Start the UI."""
        self.root.mainloop()


def main():
    """Main function to start the SDXL Turbo ControlNet Depth UI application."""
    print("SDXL Turbo + ControlNet Depth Interactive Demo")
    print("=" * 50)
    print("Features:")
    print("• Real-time depth-based diffusion control")
    print("• Adjustable radius and deformation parameters")
    print("• SDXL Turbo for fast generation")
    print("• Interactive UI with live preview")
    print()
    
    try:
        app = SDXLTurboSphereGenerator()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

def move_to_cpu(obj):
    """Recursively move all torch tensors in obj to CPU and detach them."""
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    return obj
