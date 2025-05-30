#!/usr/bin/env python3
"""
Recursive Sphere Diffusion Generator
Dark modern UI with continuous recursive diffusion on sphere geometry
Inspired by diffusion_engine.py for continuous generation
"""

import torch
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from diffusers import DPMSolverMultistepScheduler
import io
import base64
from queue import Queue
import random

class ModernDarkTheme:
    """Dark theme colors and styles"""
    BG_DARK = "#1e1e1e"
    BG_MEDIUM = "#2d2d2d" 
    BG_LIGHT = "#3e3e3e"
    FG_PRIMARY = "#ffffff"
    FG_SECONDARY = "#cccccc"
    ACCENT_BLUE = "#007acc"
    ACCENT_GREEN = "#4caf50"
    ACCENT_ORANGE = "#ff9800"
    ACCENT_PURPLE = "#9c27b0"
    BORDER = "#404040"

class RecursiveDiffusionEngine:
    """
    Recursive diffusion engine that continuously generates and feeds back
    Similar to the attached diffusion_engine.py but specialized for sphere generation
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipe = None
        self.current_latents = None
        self.recursion_queue = Queue(maxsize=3)
        self.generation_thread = None
        self.is_running = False
        self.strength = 0.7
        self.num_inference_steps = 4
        self.guidance_scale = 7.5
        
        # Sphere generation parameters
        self.sphere_cache = []
        self.recursion_depth = 0
        self.max_recursion = 50
        
    def initialize_pipeline(self):
        """Initialize SDXL Turbo pipeline"""
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            
            # Use tiny autoencoder for speed
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", 
                torch_dtype=torch.float16
            )
            
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, 
                use_karras_sigmas=True
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                self.pipe.enable_model_cpu_offload()
                
            return True
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            return False
    
    def generate_sphere_geometry(self, radius=0.8, depth=0.6, roughness=0.1, 
                                lighting_angle=45.0, metallic=0.5):
        """Generate 3D sphere depth map with parameters"""
        size = 512
        center = size // 2
        sphere_radius = int(center * radius)
        
        # Create depth map
        depth_map = np.zeros((size, size), dtype=np.float32)
        
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance <= sphere_radius:
                    # Sphere equation: z = sqrt(r² - x² - y²)
                    normalized_dist = distance / sphere_radius
                    sphere_height = np.sqrt(1 - normalized_dist*normalized_dist)
                    
                    # Apply depth scaling
                    depth_value = sphere_height * depth
                    
                    # Add surface roughness
                    if roughness > 0:
                        noise = np.random.normal(0, roughness * 0.1)
                        depth_value += noise
                    
                    # Lighting calculation
                    light_angle_rad = np.radians(lighting_angle)
                    light_x = np.cos(light_angle_rad)
                    light_y = np.sin(light_angle_rad)
                    
                    if distance > 0:
                        normal_x = dx / distance
                        normal_y = dy / distance
                        normal_z = sphere_height
                        
                        # Normalize
                        normal_length = np.sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z)
                        if normal_length > 0:
                            normal_x /= normal_length
                            normal_y /= normal_length
                            normal_z /= normal_length
                        
                        # Apply lighting
                        light_intensity = max(0, normal_x * light_x + normal_y * light_y + normal_z * 0.5)
                        depth_value *= (0.5 + 0.5 * light_intensity)
                    
                    depth_map[y, x] = depth_value
        
        # Apply metallic effect
        if metallic > 0:
            metallic_mask = depth_map > (depth_map.max() * 0.7)
            depth_map[metallic_mask] *= (1 + metallic * 0.3)
        
        # Normalize to 0-255
        depth_map = np.clip(depth_map * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(depth_map).convert("RGB")
    
    def recursive_generate(self, prompt, negative_prompt="", sphere_params=None, 
                          use_previous=True):
        """Generate with recursive feedback like diffusion_engine.py"""
        if self.pipe is None:
            return None
            
        try:
            # Generate sphere geometry
            if sphere_params is None:
                sphere_params = {
                    'radius': 0.8, 'depth': 0.6, 'roughness': 0.1,
                    'lighting_angle': 45.0, 'metallic': 0.5
                }
            
            sphere_depth = self.generate_sphere_geometry(**sphere_params)
            
            # For recursive generation, use previous result as init image
            init_image = None
            if use_previous and len(self.sphere_cache) > 0:
                init_image = self.sphere_cache[-1]
                
                # Blend with sphere depth for continuous evolution
                init_array = np.array(init_image)
                depth_array = np.array(sphere_depth)
                
                # Weighted blend - more sphere influence over time
                blend_weight = min(0.7, self.recursion_depth * 0.05)
                blended = (init_array * (1 - blend_weight) + depth_array * blend_weight).astype(np.uint8)
                sphere_depth = Image.fromarray(blended)
            
            # Generate with current settings
            if init_image is not None:
                # Image-to-image generation for recursion
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=self.strength,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    height=512,
                    width=512
                ).images[0]
            else:
                # Text-to-image for first generation
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    height=512,
                    width=512
                ).images[0]
            
            # Add to cache and manage size
            self.sphere_cache.append(result)
            if len(self.sphere_cache) > 5:
                self.sphere_cache.pop(0)
            
            self.recursion_depth += 1
            if self.recursion_depth >= self.max_recursion:
                self.recursion_depth = 0  # Reset for continuous generation
            
            return result, sphere_depth
            
        except Exception as e:
            print(f"Error in recursive generation: {e}")
            return None, None
    
    def start_continuous_generation(self, prompt, negative_prompt="", 
                                   sphere_params=None, callback=None):
        """Start continuous recursive generation thread"""
        self.is_running = True
        
        def generation_loop():
            while self.is_running:
                try:
                    result, depth = self.recursive_generate(
                        prompt, negative_prompt, sphere_params, 
                        use_previous=True
                    )
                    
                    if result and callback:
                        callback(result, depth)
                    
                    # Add some variation to parameters for evolution
                    if sphere_params:
                        sphere_params['roughness'] += random.uniform(-0.02, 0.02)
                        sphere_params['roughness'] = np.clip(sphere_params['roughness'], 0, 0.5)
                        
                        sphere_params['lighting_angle'] += random.uniform(-2, 2)
                        sphere_params['lighting_angle'] = np.clip(sphere_params['lighting_angle'], 0, 90)
                    
                    time.sleep(0.5)  # Control generation speed
                    
                except Exception as e:
                    print(f"Error in generation loop: {e}")
                    time.sleep(1)
        
        self.generation_thread = threading.Thread(target=generation_loop, daemon=True)
        self.generation_thread.start()
    
    def stop_continuous_generation(self):
        """Stop continuous generation"""
        self.is_running = False
        if self.generation_thread:
            self.generation_thread.join(timeout=2)

class SphereRecursiveUI:
    """Modern dark UI for recursive sphere diffusion"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_modern_theme()
        self.root.title("Recursive Sphere Diffusion")
        self.root.geometry("1600x1000")
        self.root.configure(bg=ModernDarkTheme.BG_DARK)
        
        # Initialize diffusion engine
        self.diffusion_engine = RecursiveDiffusionEngine()
        
        # UI Variables
        self.prompt = tk.StringVar(value="crystalline sphere, futuristic, holographic, highly detailed, 8k")
        self.negative_prompt = tk.StringVar(value="blurry, low quality, distorted, flat, ugly")
        
        # Sphere parameters
        self.sphere_radius = tk.DoubleVar(value=0.8)
        self.sphere_depth = tk.DoubleVar(value=0.6)
        self.surface_roughness = tk.DoubleVar(value=0.1)
        self.lighting_angle = tk.DoubleVar(value=45.0)
        self.metallic = tk.DoubleVar(value=0.5)
        
        # Generation parameters
        self.strength = tk.DoubleVar(value=0.7)
        self.guidance_scale = tk.DoubleVar(value=7.5)
        self.num_steps = tk.IntVar(value=4)
        self.recursion_speed = tk.DoubleVar(value=1.0)
        
        # State
        self.is_generating = False
        self.auto_recursive = tk.BooleanVar(value=False)
        
        self.setup_ui()
        self.initialize_pipeline()
        
    def setup_modern_theme(self):
        """Configure modern dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme configuration
        style.configure('TLabel', 
                       background=ModernDarkTheme.BG_DARK,
                       foreground=ModernDarkTheme.FG_PRIMARY,
                       font=('Segoe UI', 10))
        
        style.configure('TFrame', background=ModernDarkTheme.BG_DARK)
        
        style.configure('TButton',
                       background=ModernDarkTheme.ACCENT_BLUE,
                       foreground=ModernDarkTheme.FG_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'))
        
        style.map('TButton',
                 background=[('active', '#005a9e'),
                            ('pressed', '#004080')])
    
    def setup_ui(self):
        """Setup the modern dark UI"""
        # Main container
        main_frame = tk.Frame(self.root, bg=ModernDarkTheme.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="Recursive Sphere Diffusion Generator",
                              font=('Segoe UI', 24, 'bold'),
                              bg=ModernDarkTheme.BG_DARK,
                              fg=ModernDarkTheme.ACCENT_BLUE)
        title_label.pack(pady=(0, 30))
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=ModernDarkTheme.BG_DARK)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup panels
        self.setup_control_panel(content_frame)
        self.setup_preview_panel(content_frame)
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM, relief='solid', bd=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), pady=0)
        control_frame.configure(width=450)
        control_frame.pack_propagate(False)
        
        # Scrollable content
        canvas = tk.Canvas(control_frame, bg=ModernDarkTheme.BG_MEDIUM, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=ModernDarkTheme.BG_MEDIUM)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Control sections
        self.create_prompt_section(scrollable_frame)
        self.create_sphere_controls(scrollable_frame)
        self.create_generation_controls(scrollable_frame)
        self.create_recursion_controls(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        scrollbar.pack(side="right", fill="y")
    
    def create_prompt_section(self, parent):
        """Create prompt input section"""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 25))
        
        # Section title
        tk.Label(section, text="Prompts", 
                font=('Segoe UI', 16, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.ACCENT_PURPLE).pack(anchor=tk.W, pady=(0, 15))
        
        # Prompt
        tk.Label(section, text="Prompt:", 
                font=('Segoe UI', 12, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.FG_PRIMARY).pack(anchor=tk.W, pady=(0, 5))
        
        prompt_text = tk.Text(section, height=3, width=45,
                             bg=ModernDarkTheme.BG_LIGHT,
                             fg=ModernDarkTheme.FG_PRIMARY,
                             insertbackground=ModernDarkTheme.FG_PRIMARY,
                             font=('Segoe UI', 10),
                             wrap=tk.WORD)
        prompt_text.pack(fill=tk.X, pady=(0, 15))
        prompt_text.insert('1.0', self.prompt.get())
        
        def update_prompt(*args):
            self.prompt.set(prompt_text.get('1.0', 'end-1c'))
        prompt_text.bind('<KeyRelease>', update_prompt)
        
        # Negative prompt
        tk.Label(section, text="Negative Prompt:", 
                font=('Segoe UI', 12, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.FG_PRIMARY).pack(anchor=tk.W, pady=(0, 5))
        
        neg_text = tk.Text(section, height=2, width=45,
                          bg=ModernDarkTheme.BG_LIGHT,
                          fg=ModernDarkTheme.FG_PRIMARY,
                          insertbackground=ModernDarkTheme.FG_PRIMARY,
                          font=('Segoe UI', 10),
                          wrap=tk.WORD)
        neg_text.pack(fill=tk.X)
        neg_text.insert('1.0', self.negative_prompt.get())
        
        def update_neg_prompt(*args):
            self.negative_prompt.set(neg_text.get('1.0', 'end-1c'))
        neg_text.bind('<KeyRelease>', update_neg_prompt)
    
    def create_sphere_controls(self, parent):
        """Create sphere parameter controls"""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 25))
        
        tk.Label(section, text="Sphere Geometry", 
                font=('Segoe UI', 16, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.ACCENT_BLUE).pack(anchor=tk.W, pady=(0, 15))
        
        self.create_modern_slider(section, "Radius", self.sphere_radius, 0.1, 1.0)
        self.create_modern_slider(section, "Depth", self.sphere_depth, 0.1, 1.0)
        self.create_modern_slider(section, "Surface Roughness", self.surface_roughness, 0.0, 0.5)
        self.create_modern_slider(section, "Lighting Angle", self.lighting_angle, 0.0, 90.0)
        self.create_modern_slider(section, "Metallic", self.metallic, 0.0, 1.0)
    
    def create_generation_controls(self, parent):
        """Create generation controls"""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 25))
        
        tk.Label(section, text="Generation Settings", 
                font=('Segoe UI', 16, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.ACCENT_GREEN).pack(anchor=tk.W, pady=(0, 15))
        
        self.create_modern_slider(section, "Strength", self.strength, 0.1, 1.0)
        self.create_modern_slider(section, "Guidance Scale", self.guidance_scale, 1.0, 15.0)
        self.create_modern_slider(section, "Steps", self.num_steps, 1, 8, is_int=True)
    
    def create_recursion_controls(self, parent):
        """Create recursion controls"""
        section = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        section.pack(fill=tk.X, pady=(0, 25))
        
        tk.Label(section, text="Recursive Generation", 
                font=('Segoe UI', 16, 'bold'),
                bg=ModernDarkTheme.BG_MEDIUM,
                fg=ModernDarkTheme.ACCENT_ORANGE).pack(anchor=tk.W, pady=(0, 15))
        
        self.create_modern_slider(section, "Recursion Speed", self.recursion_speed, 0.1, 3.0)
        
        # Auto-recursive checkbox
        auto_check = tk.Checkbutton(section,
                                   text="Enable Auto-Recursive Generation",
                                   variable=self.auto_recursive,
                                   bg=ModernDarkTheme.BG_MEDIUM,
                                   fg=ModernDarkTheme.FG_PRIMARY,
                                   selectcolor=ModernDarkTheme.BG_LIGHT,
                                   activebackground=ModernDarkTheme.BG_MEDIUM,
                                   activeforeground=ModernDarkTheme.FG_PRIMARY,
                                   font=('Segoe UI', 11, 'bold'),
                                   command=self.toggle_auto_recursive)
        auto_check.pack(anchor=tk.W, pady=(15, 10))
        
        # Control buttons
        btn_frame = tk.Frame(section, bg=ModernDarkTheme.BG_MEDIUM)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.generate_btn = tk.Button(btn_frame,
                                     text="Generate Once",
                                     command=self.generate_single,
                                     bg=ModernDarkTheme.ACCENT_BLUE,
                                     fg=ModernDarkTheme.FG_PRIMARY,
                                     font=('Segoe UI', 12, 'bold'),
                                     relief='flat',
                                     bd=0,
                                     padx=20,
                                     pady=10)
        self.generate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.recursive_btn = tk.Button(btn_frame,
                                      text="Start Recursive",
                                      command=self.toggle_recursive_generation,
                                      bg=ModernDarkTheme.ACCENT_GREEN,
                                      fg=ModernDarkTheme.FG_PRIMARY,
                                      font=('Segoe UI', 12, 'bold'),
                                      relief='flat',
                                      bd=0,
                                      padx=20,
                                      pady=10)
        self.recursive_btn.pack(side=tk.LEFT)
    
    def create_modern_slider(self, parent, label, variable, min_val, max_val, is_int=False):
        """Create modern styled slider"""
        slider_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM)
        slider_frame.pack(fill=tk.X, pady=(0, 12))
        
        # Label with value
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
        resolution = 1 if is_int else 0.01
        slider = tk.Scale(slider_frame, from_=min_val, to=max_val, 
                         variable=variable, orient=tk.HORIZONTAL, 
                         length=350, resolution=resolution,
                         bg=ModernDarkTheme.BG_MEDIUM,
                         fg=ModernDarkTheme.FG_PRIMARY,
                         highlightthickness=0,
                         troughcolor=ModernDarkTheme.BG_LIGHT,
                         activebackground=ModernDarkTheme.ACCENT_BLUE,
                         font=('Segoe UI', 9))
        slider.pack(fill=tk.X)
        
        def on_change(*args):
            if is_int:
                value_label.config(text=f"{int(variable.get())}")
            else:
                value_label.config(text=f"{variable.get():.2f}")
            
            # Update diffusion engine parameters
            self.update_engine_params()
        
        variable.trace_add('write', on_change)
        return slider
    
    def setup_preview_panel(self, parent):
        """Setup preview panel"""
        preview_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_MEDIUM, relief='solid', bd=1)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Title
        preview_title = tk.Label(preview_frame,
                                text="Recursive Generation Preview",
                                font=('Segoe UI', 18, 'bold'),
                                bg=ModernDarkTheme.BG_MEDIUM,
                                fg=ModernDarkTheme.FG_PRIMARY)
        preview_title.pack(pady=20)
        
        # Preview area
        self.preview_canvas = tk.Canvas(preview_frame,
                                       bg=ModernDarkTheme.BG_DARK,
                                       highlightthickness=0,
                                       width=800, height=800)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Add generation counter
        self.generation_counter = tk.Label(preview_frame,
                                          text="Generation: 0",
                                          font=('Segoe UI', 14, 'bold'),
                                          bg=ModernDarkTheme.BG_MEDIUM,
                                          fg=ModernDarkTheme.ACCENT_ORANGE)
        self.generation_counter.pack(pady=(0, 10))
    
    def setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = tk.Frame(parent, bg=ModernDarkTheme.BG_LIGHT, height=35)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(status_frame,
                               textvariable=self.status_var,
                               bg=ModernDarkTheme.BG_LIGHT,
                               fg=ModernDarkTheme.FG_PRIMARY,
                               font=('Segoe UI', 11))
        status_label.pack(side=tk.LEFT, padx=15, pady=8)
    
    def initialize_pipeline(self):
        """Initialize the diffusion pipeline"""
        self.status_var.set("Loading diffusion models...")
        
        def load_models():
            success = self.diffusion_engine.initialize_pipeline()
            if success:
                self.status_var.set("Ready for recursive generation")
            else:
                self.status_var.set("Error loading models")
        
        threading.Thread(target=load_models, daemon=True).start()
    
    def update_engine_params(self):
        """Update diffusion engine parameters"""
        if self.diffusion_engine:
            self.diffusion_engine.strength = self.strength.get()
            self.diffusion_engine.guidance_scale = self.guidance_scale.get()
            self.diffusion_engine.num_inference_steps = self.num_steps.get()
    
    def get_sphere_params(self):
        """Get current sphere parameters"""
        return {
            'radius': self.sphere_radius.get(),
            'depth': self.sphere_depth.get(),
            'roughness': self.surface_roughness.get(),
            'lighting_angle': self.lighting_angle.get(),
            'metallic': self.metallic.get()
        }
    
    def display_result(self, image, depth_map=None):
        """Display generated result in preview"""
        if image is None:
            return
        
        # Update generation counter
        count = self.diffusion_engine.recursion_depth
        self.generation_counter.config(text=f"Generation: {count}")
        
        # Display image
        self.display_image_on_canvas(image, self.preview_canvas)
    
    def display_image_on_canvas(self, pil_image, canvas):
        """Display PIL image on canvas"""
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 800
        
        # Resize maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_image)
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        canvas.image = photo  # Keep reference
    
    def generate_single(self):
        """Generate a single image"""
        if self.diffusion_engine.pipe is None:
            messagebox.showwarning("Warning", "Pipeline not ready")
            return
        
        self.status_var.set("Generating single image...")
        
        def generate():
            try:
                result, depth = self.diffusion_engine.recursive_generate(
                    self.prompt.get(),
                    self.negative_prompt.get(),
                    self.get_sphere_params(),
                    use_previous=len(self.diffusion_engine.sphere_cache) > 0
                )
                
                if result:
                    self.root.after(0, lambda: self.display_result(result, depth))
                    self.root.after(0, lambda: self.status_var.set("Single generation complete"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
        
        threading.Thread(target=generate, daemon=True).start()
    
    def toggle_recursive_generation(self):
        """Toggle recursive generation"""
        if self.is_generating:
            self.stop_recursive_generation()
        else:
            self.start_recursive_generation()
    
    def start_recursive_generation(self):
        """Start recursive generation"""
        if self.diffusion_engine.pipe is None:
            messagebox.showwarning("Warning", "Pipeline not ready")
            return
        
        self.is_generating = True
        self.recursive_btn.config(text="Stop Recursive", bg=ModernDarkTheme.ACCENT_ORANGE)
        self.status_var.set("Starting recursive generation...")
        
        # Start continuous generation with callback
        self.diffusion_engine.start_continuous_generation(
            self.prompt.get(),
            self.negative_prompt.get(),
            self.get_sphere_params(),
            callback=lambda result, depth: self.root.after(0, lambda: self.display_result(result, depth))
        )
    
    def stop_recursive_generation(self):
        """Stop recursive generation"""
        self.is_generating = False
        self.recursive_btn.config(text="Start Recursive", bg=ModernDarkTheme.ACCENT_GREEN)
        self.diffusion_engine.stop_continuous_generation()
        self.status_var.set("Recursive generation stopped")
    
    def toggle_auto_recursive(self):
        """Toggle auto-recursive mode"""
        if self.auto_recursive.get():
            if not self.is_generating:
                self.start_recursive_generation()
        else:
            if self.is_generating:
                self.stop_recursive_generation()
    
    def run(self):
        """Start the application"""
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_generating:
            self.stop_recursive_generation()
        self.root.destroy()


def main():
    """Main function"""
    print("Recursive Sphere Diffusion Generator")
    print("=" * 50)
    print("Features:")
    print("• Continuous recursive diffusion generation")
    print("• Real-time sphere geometry evolution")
    print("• Modern dark UI")
    print("• Parameter-driven sphere morphing")
    print("• SDXL Turbo for fast iteration")
    print()
    
    try:
        app = SphereRecursiveUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
