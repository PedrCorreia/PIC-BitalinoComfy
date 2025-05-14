#!/usr/bin/env python
"""
PlotUnit Standalone Debug App

This is a completely self-contained debug app that doesn't rely on imports from
the existing module structure. It implements just enough functionality to test
the button clicks and tab navigation.
"""

import os
import sys
import time
import pygame
import threading
import numpy as np
from enum import Enum
from collections import deque

# Initialize pygame
pygame.init()

# === Constants ===
# Window dimensions and layout
WINDOW_WIDTH = 530
WINDOW_HEIGHT = 780
SIDEBAR_WIDTH = 50
STATUS_BAR_HEIGHT = 30

# UI Layout settings
PLOT_PADDING = 15
TAB_HEIGHT = 30
CONTROL_PANEL_HEIGHT = 40
TWIN_VIEW_SEPARATOR = 10

# Position for status bar - TOP instead of bottom
STATUS_BAR_TOP = True

# Additional UI spacing parameters
BUTTON_MARGIN = 8
SECTION_MARGIN = 20
TITLE_PADDING = 12
TEXT_MARGIN = 5
CONTROL_MARGIN = 10
ELEMENT_PADDING = 8

# Colors
BACKGROUND_COLOR = (14, 14, 14)
BUTTON_COLOR = (24, 24, 24)
BUTTON_COLOR_SETTINGS = (250, 24, 24)
SIDEBAR_COLOR = (24, 24, 24)
ACCENT_COLOR = (0, 120, 215)
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (40, 40, 40)

# Signal colors
RAW_SIGNAL_COLOR = (220, 180, 0)
PROCESSED_SIGNAL_COLOR = (0, 180, 220)
ECG_SIGNAL_COLOR = (255, 0, 0)
EDA_SIGNAL_COLOR = (220, 120, 0)

# Status colors
OK_COLOR = (0, 220, 0)
WARNING_COLOR = (220, 220, 0)
ERROR_COLOR = (220, 0, 0)

# Default settings
DEFAULT_SETTINGS = {
    'caps_enabled': True,
    'light_mode': False,
    'performance_mode': False,
    'connected_nodes': 0,
    'reset_plots': False,
    'reset_registry': False,
}

# === ViewMode Enum ===
class ViewMode(Enum):
    RAW = 0
    PROCESSED = 1
    TWIN = 2
    SETTINGS = 3

# === Base View Class ===
class BaseView:
    """Base class for all views."""
    
    def __init__(self, surface, data_lock, data, font):
        self.surface = surface
        self.data_lock = data_lock
        self.data = data
        self.font = font
        
        # Adjust plot rect based on status bar position (top/bottom)
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        self.plot_rect = pygame.Rect(
            SIDEBAR_WIDTH + PLOT_PADDING,
            PLOT_PADDING + status_bar_offset,
            WINDOW_WIDTH - SIDEBAR_WIDTH - PLOT_PADDING * 2,
            WINDOW_HEIGHT - STATUS_BAR_HEIGHT - PLOT_PADDING * 2
        )
    
    def draw(self):
        """Draw method to be implemented by subclasses."""
        pass

# === Settings View ===
class SettingsView(BaseView):
    """View for displaying and managing settings."""
    
    def __init__(self, surface, data_lock, data, font, settings):
        super().__init__(surface, data_lock, data, font)
        self.settings = settings
        self.settings_buttons = []  # Will store (rect, setting_key) pairs
        self.plot_unit = None  # Reference to PlotUnit for action handlers
        print("SettingsView initialized")
    
    def draw(self):
        # Clear the settings buttons list
        self.settings_buttons = []
        
        # Debug message
        print("Drawing settings view")
        
        # Draw settings header
        rect = self.plot_rect
        header = "Settings"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (rect.x + SECTION_MARGIN, rect.y + SECTION_MARGIN))
        
        # Calculate positions
        first_section_y = rect.y + SECTION_MARGIN + TITLE_PADDING + self.font.get_height()
        section_height = 120
        
        # Draw sections
        self._draw_visualization_settings(rect.x + SECTION_MARGIN, first_section_y, rect.width - (SECTION_MARGIN * 2))
        self._draw_appearance_settings(rect.x + SECTION_MARGIN, first_section_y + section_height, rect.width - (SECTION_MARGIN * 2))
        self._draw_action_buttons(rect.x + SECTION_MARGIN, first_section_y + section_height * 2, rect.width - (SECTION_MARGIN * 2))
    
    def _draw_visualization_settings(self, x, y, width):
        # Draw section header
        header = "Visualization"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw toggle
        self._draw_toggle(
            x, y + 40,
            "FPS Cap",
            "caps_enabled",
            self.settings['caps_enabled']
        )
        
        self._draw_toggle(
            x, y + 80,
            "Performance Mode",
            "performance_mode",
            self.settings['performance_mode']
        )
    
    def _draw_appearance_settings(self, x, y, width):
        # Draw section header
        header = "Appearance"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw toggle
        self._draw_toggle(
            x, y + 40,
            "Light Mode",
            "light_mode",
            self.settings['light_mode']
        )
    
    def _draw_action_buttons(self, x, y, width):
        # Draw section header
        header = "Actions"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw buttons
        self._draw_button(
            x, y + 40,
            "Reset Plots",
            "reset_plots",
            ACCENT_COLOR
        )
        
        self._draw_button(
            x, y + 80,
            "Reset Registry",
            "reset_registry",
            ACCENT_COLOR        )
    
    def _draw_toggle(self, x, y, label, setting_key, value):
        # Draw label
        label_surface = self.font.render(label, True, TEXT_COLOR)
        self.surface.blit(label_surface, (x, y))
        
        # Draw toggle switch
        switch_width = 48
        switch_height = 24
        switch_x = x + 200
        switch_y = y
        
        # Choose background color based on state - make colors more prominent
        bg_color = (50, 150, 255) if value else (80, 80, 80)  # Brighter blue when on, slightly lighter gray when off
        
        # Draw switch background without border
        switch_bg_rect = pygame.Rect(switch_x, switch_y, switch_width, switch_height)
        pygame.draw.rect(self.surface, bg_color, switch_bg_rect, border_radius=13)
        
        # Draw switch handle
        handle_size = 18
        handle_margin = 4
        handle_pos = switch_x + switch_width - handle_size - handle_margin if value else switch_x + handle_margin
        handle_rect = pygame.Rect(handle_pos, switch_y + handle_margin, handle_size, handle_size)
        pygame.draw.rect(self.surface, TEXT_COLOR, handle_rect, border_radius=9)
        
        # Store button rect and settings key for click handling
        self.settings_buttons.append((switch_bg_rect, setting_key))
        
        # Debug output
        print(f"Added toggle: {setting_key}, rect: {switch_bg_rect}, value: {value}") 
    def _draw_button(self, x, y, label, action_key, color):
        # Draw button - rectangle with no border
        button_width = 150
        button_height = 30
        button_rect = pygame.Rect(x, y, button_width, button_height)
        
        # Use a brighter red color for action buttons without border
        button_color = BUTTON_COLOR_SETTINGS  # Brighter red for reset buttons
        pygame.draw.rect(self.surface, button_color, button_rect, border_radius=5)
        
        # Draw label
        label_surface = self.font.render(label, True, TEXT_COLOR)
        label_rect = label_surface.get_rect(center=button_rect.center)
        self.surface.blit(label_surface, label_rect)
        
        # Store button rect and action key for click handling
        self.settings_buttons.append((button_rect, action_key))
        
        # Debug output
        print(f"Added action button: {action_key}, rect: {button_rect}")

# === Sidebar Class ===
class Sidebar:
    """Sidebar for navigation between different view modes."""
    
    def __init__(self, surface, font):
        self.surface = surface
        self.font = font
        self.width = SIDEBAR_WIDTH
        self.current_mode = ViewMode.RAW
        self.tab_names = {
            ViewMode.RAW: "R",
            ViewMode.PROCESSED: "P",
            ViewMode.TWIN: "T",
            ViewMode.SETTINGS: "S"
        }
    
    def draw(self):
        # Draw sidebar background
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        sidebar_rect = pygame.Rect(0, status_bar_offset, self.width, WINDOW_HEIGHT - status_bar_offset)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, sidebar_rect)
        
        # Draw tab buttons
        for i, mode in enumerate([ViewMode.RAW, ViewMode.PROCESSED, ViewMode.TWIN, ViewMode.SETTINGS]):
            self._draw_tab_button(i, mode == self.current_mode, self.tab_names[mode])
    def _draw_tab_button(self, index, active, label):
        # Adjust position based on status bar position
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        y_pos = PLOT_PADDING + status_bar_offset + index * (TAB_HEIGHT + BUTTON_MARGIN)
        button_rect = pygame.Rect(BUTTON_MARGIN, y_pos, self.width - BUTTON_MARGIN * 2, TAB_HEIGHT)
        
        # Draw button with accent color if active
        button_color = ACCENT_COLOR  if active else BUTTON_COLOR
        pygame.draw.rect(self.surface, button_color, button_rect, border_radius=5)
        
        # Create a larger font for the tab labels
        icon_font = pygame.font.SysFont(None, 28)  # Larger font size for icons
        
        # Draw label with larger font
        label_surface = icon_font.render(label, True, TEXT_COLOR)
        label_rect = label_surface.get_rect(center=button_rect.center)
        self.surface.blit(label_surface, label_rect)
    def handle_click(self, y):
        # Calculate which tab was clicked - adjusting for status bar position
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        for i in range(4):  # 4 view modes
            y_pos = PLOT_PADDING + status_bar_offset + i * (TAB_HEIGHT + BUTTON_MARGIN)
            if y_pos <= y <= y_pos + TAB_HEIGHT:
                print(f"Tab clicked: {i}")
                return i
        return None

# === Status Bar Class ===
class StatusBar:
    """Status bar for displaying system information."""
    
    def __init__(self, surface, font):
        self.surface = surface
        self.font = font
        self.height = STATUS_BAR_HEIGHT
    
    def draw(self, fps=0, nodes=0, runtime="00:04", latency="0.0 ms", signals=3, last_update="Just now"):
        # Draw status bar at top instead of bottom
        bar_rect = pygame.Rect(0, 0 if STATUS_BAR_TOP else WINDOW_HEIGHT - self.height, WINDOW_WIDTH, self.height)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, bar_rect)
        
        # Draw runtime info
        runtime_text = f"Runtime: {runtime}"
        runtime_surface = self.font.render(runtime_text, True, TEXT_COLOR)
        self.surface.blit(runtime_surface, (PLOT_PADDING, 5 if STATUS_BAR_TOP else WINDOW_HEIGHT - self.height + 5))
        
        # Draw latency info with green color
        latency_text = f"Latency: {latency}"
        latency_surface = self.font.render(latency_text, True, OK_COLOR)
        self.surface.blit(latency_surface, (PLOT_PADDING + 120, 5 if STATUS_BAR_TOP else WINDOW_HEIGHT - self.height + 5))
        
        # Draw signals count
        signals_text = f"Signals: {signals}"
        signals_surface = self.font.render(signals_text, True, TEXT_COLOR)
        self.surface.blit(signals_surface, (PLOT_PADDING + 240, 5 if STATUS_BAR_TOP else WINDOW_HEIGHT - self.height + 5))
        
        # Draw last update time
        update_text = f"Last update: {last_update}"
        update_surface = self.font.render(update_text, True, TEXT_COLOR)
        self.surface.blit(update_surface, (PLOT_PADDING + 320, 5 if STATUS_BAR_TOP else WINDOW_HEIGHT - self.height + 5))
        
        # Draw FPS counter on right side
        fps_text = f"FPS: {int(fps)}"
        fps_surface = self.font.render(fps_text, True, TEXT_COLOR)
        self.surface.blit(fps_surface, (WINDOW_WIDTH - 80, 5 if STATUS_BAR_TOP else WINDOW_HEIGHT - self.height + 5))

# === Event Handler Class ===
class EventHandler:
    """Event handler for the PlotUnit system."""
    
    def __init__(self, sidebar, settings_view=None):
        self.sidebar = sidebar
        self.settings_view = settings_view
        self.current_mode = sidebar.current_mode
        print(f"EventHandler initialized with current_mode: {self.current_mode.name}")
    
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    x, y = event.pos
                    self._handle_click(x, y)
        
        return True
    
    def _handle_click(self, x, y):
        print(f"Click detected at ({x}, {y})")
        if x < self.sidebar.width:
            # Click in sidebar area
            mode_index = self.sidebar.handle_click(y)
            if mode_index is not None:
                self._update_mode(mode_index)
        elif self.current_mode == ViewMode.SETTINGS and self.settings_view:
            # Pass click to settings view
            self._handle_settings_click(x, y)
    
    def _update_mode(self, mode_index):
        # Map index to ViewMode
        mode_mapping = {
            0: ViewMode.RAW,
            1: ViewMode.PROCESSED,
            2: ViewMode.TWIN,
            3: ViewMode.SETTINGS
        }
        
        if mode_index in mode_mapping:
            self.current_mode = mode_mapping[mode_index]
            self.sidebar.current_mode = self.current_mode
            print(f"Mode updated to: {self.current_mode.name}")
    
    def _handle_settings_click(self, x, y):
        if not self.settings_view:
            print("Settings view not available")
            return
        
        print(f"Settings view click at: ({x}, {y})")
        
        # Check if a settings button was clicked
        found_button = False
        for button_rect, setting_key in self.settings_view.settings_buttons:
            if button_rect.collidepoint(x, y):
                print(f"Button clicked: {setting_key}")
                self._toggle_setting(setting_key)
                found_button = True
                break
        
        if not found_button:
            print(f"No button found at position: ({x}, {y})")
            print(f"Available buttons: {len(self.settings_view.settings_buttons)}")
    def _toggle_setting(self, setting_key):
        print(f"Toggling setting: {setting_key}")
        
        # Special handling for action buttons
        if setting_key == 'reset_plots':
            print("Reset plots action triggered")
            # Flash the button by temporarily changing its color
            if hasattr(self, 'settings_view') and self.settings_view:
                for rect, key in self.settings_view.settings_buttons:
                    if key == 'reset_plots':
                        # We'll just trigger the action without visual feedback in this demo
                        break
            return True
        
        elif setting_key == 'reset_registry':
            print("Reset registry action triggered")
            # Same visual feedback approach
            return True
        
        # Toggle regular settings
        if setting_key in self.settings_view.settings:
            self.settings_view.settings[setting_key] = not self.settings_view.settings[setting_key]
            print(f"Setting '{setting_key}' toggled to: {self.settings_view.settings[setting_key]}")
        
        return False
    
    def get_current_mode(self):
        return self.current_mode

# === PlotUnit Main Class ===
class PlotUnit:
    """Main class for the PlotUnit visualization system."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Initialize basic attributes
        self.running = False
        self.initialized = False
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.settings = DEFAULT_SETTINGS.copy()
        
        # Set up threading-related attributes
        self.data_lock = threading.Lock()
        self.data = {}
        
        # Set up main thread
        self.thread = None
        
        # Store last frame time for FPS calculation
        self.last_frame_time = time.time()
        self.fps = 0
        
        print("PlotUnit initialized")
    
    def start(self):
        """Start the visualization thread."""
        if self.running:
            print("PlotUnit already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        print("PlotUnit thread started")
    
    def _run(self):
        """Main visualization loop."""
        try:            # Initialize pygame
            self.surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("ComfyUI - PlotUnit")
            
            # Create font for rendering text
            self.font = pygame.font.SysFont("Arial", 14)
            
            # Initialize UI components
            self.sidebar = Sidebar(self.surface, self.font)
            self.status_bar = StatusBar(self.surface, self.font)
            
            # Create views
            self.views = {
                ViewMode.RAW: BaseView(self.surface, self.data_lock, self.data, self.font),
                ViewMode.PROCESSED: BaseView(self.surface, self.data_lock, self.data, self.font),
                ViewMode.TWIN: BaseView(self.surface, self.data_lock, self.data, self.font),
                ViewMode.SETTINGS: SettingsView(self.surface, self.data_lock, self.data, self.font, self.settings)
            }
            
            # Create event handler
            self.event_handler = EventHandler(self.sidebar, self.views[ViewMode.SETTINGS])
            
            # Set as initialized
            self.initialized = True
            print("PlotUnit initialized successfully")
            
            # Main loop
            while self.running:
                # Process events
                self.running = self.event_handler.process_events()
                
                # Clear the screen
                self.surface.fill(BACKGROUND_COLOR)
                
                # Get current mode
                current_mode = self.event_handler.get_current_mode()
                
                # Draw current view
                self.views[current_mode].draw()
                  # Draw sidebar and status bar
                self.sidebar.draw()
                self.status_bar.draw(
                    self.fps, 
                    self.settings['connected_nodes'],
                    runtime="00:04", 
                    latency="0.0 ms", 
                    signals=3, 
                    last_update="Just now"
                )
                
                # Update the display
                pygame.display.flip()
                
                # Calculate FPS
                current_time = time.time()
                delta_time = current_time - self.last_frame_time
                self.fps = 1.0 / delta_time if delta_time > 0 else 0
                self.last_frame_time = current_time
                
                # Cap FPS to reduce CPU usage
                time.sleep(0.03)  # ~30 FPS
        
        except Exception as e:
            print(f"Error in visualization thread: {str(e)}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def load_test_signals(self):
        """Load test signals for visualization."""
        print("Test signals loaded (placeholder)")
    
    def _set_mode(self, mode):
        """Set the current view mode."""
        if self.initialized and hasattr(self, 'sidebar'):
            self.sidebar.current_mode = mode
            if hasattr(self, 'event_handler'):
                self.event_handler.current_mode = mode
            print(f"View mode set to: {mode.name}")
    
    def clear_plots(self):
        """Clear all plots."""
        print("Plots cleared")

# === Main Function ===
def main():
    """Main function to test the PlotUnit system."""
    print("\n=== PlotUnit Debug ===\n")
    
    # Set window title and icon
    pygame.display.set_caption("ComfyUI - PlotUnit")
    
    # Create PlotUnit instance
    plot = PlotUnit.get_instance()
    print("PlotUnit instance created")
    
    # Start visualization
    plot.start()
    print("PlotUnit visualization started")
    
    # Wait for initialization
    wait_seconds = 0
    while not plot.initialized and wait_seconds < 5:
        print(f"Waiting for initialization... ({wait_seconds+1}/5)")
        time.sleep(1.0)
        wait_seconds += 1
    
    if not plot.initialized:
        print("PlotUnit failed to initialize within the timeout period")
        return False
    
    print("PlotUnit initialized successfully")
    
    # Switch to SETTINGS view
    print("\nSwitching to SETTINGS view...")
    plot._set_mode(ViewMode.SETTINGS)
    time.sleep(0.5)
    
    # Print instructions
    print("\n=== Instructions ===")
    print("1. Look at the PlotUnit window that opened")
    print("2. Click on tabs in the left sidebar to switch views")
    print("3. Try clicking on toggle buttons and action buttons in the Settings view")
    print("4. Watch this console for debug messages")
    print("\nPress Ctrl+C to exit")
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting debug program...")
    
    return True

if __name__ == "__main__":
    main()
