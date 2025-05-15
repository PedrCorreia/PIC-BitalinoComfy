"""
UI components for PlotUnit visualization system

This module provides the UI components used by the PlotUnit visualization system.
"""

import pygame

# Import local modules
from modules.constants import *
from modules.view_mode import ViewMode


class BaseView:
    """Base class for all views."""
    
    def __init__(self, surface, data_lock, data, font):
        """Initialize the base view.
        
        Args:
            surface: The pygame surface to draw on
            data_lock: Threading lock for accessing the data
            data: The data to visualize
            font: The font to use for text
        """
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


class SettingsView(BaseView):
    """View for displaying and managing settings."""
    
    def __init__(self, surface, data_lock, data, font, settings):
        """Initialize the settings view.
        
        Args:
            surface: The pygame surface to draw on
            data_lock: Threading lock for accessing the data
            data: The data to visualize
            font: The font to use for text
            settings: The current settings dictionary
        """
        super().__init__(surface, data_lock, data, font)
        self.settings = settings
        self.settings_buttons = []  # Will store (rect, setting_key) pairs
        self.plot_unit = None  # Reference to PlotUnit for action handlers
        print("SettingsView initialized")
    
    def draw(self):
        """Draw the settings view."""
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
        """Draw visualization settings section.
        
        Args:
            x: The x position to draw at
            y: The y position to draw at
            width: The width of the section
        """
        # Draw section header
        header = "Visualization"
        header_surface = self.font.render(header, True, TEXT_COLOR)
        self.surface.blit(header_surface, (x, y))
        
        # Draw toggle buttons
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
        """Draw appearance settings section.
        
        Args:
            x: The x position to draw at
            y: The y position to draw at
            width: The width of the section
        """
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
        """Draw action buttons section.
        
        Args:
            x: The x position to draw at
            y: The y position to draw at
            width: The width of the section
        """
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
            ACCENT_COLOR
        )
    
    def _draw_toggle(self, x, y, label, setting_key, value):
        """Draw a toggle button.
        
        Args:
            x: The x position to draw at
            y: The y position to draw at
            label: The label text
            setting_key: The setting key in the settings dictionary
            value: The current value of the setting
        """
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
        """Draw an action button.
        
        Args:
            x: The x position to draw at
            y: The y position to draw at
            label: The label text
            action_key: The action key
            color: The button color
        """
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


class Sidebar:
    """Sidebar for navigation between different view modes."""
    
    def __init__(self, surface, font):
        """Initialize the sidebar.
        
        Args:
            surface: The pygame surface to draw on
            font: The font to use for text
        """
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
        """Draw the sidebar."""
        # Draw sidebar background
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        sidebar_rect = pygame.Rect(0, status_bar_offset, self.width, WINDOW_HEIGHT - status_bar_offset)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, sidebar_rect)
        
        # Draw tab buttons
        for i, mode in enumerate([ViewMode.RAW, ViewMode.PROCESSED, ViewMode.TWIN, ViewMode.SETTINGS]):
            self._draw_tab_button(i, mode == self.current_mode, self.tab_names[mode])
            
    def _draw_tab_button(self, index, active, label):
        """Draw a tab button.
        
        Args:
            index: The index of the tab
            active: Whether the tab is active
            label: The tab label text
        """
        # Adjust position based on status bar position
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        y_pos = PLOT_PADDING + status_bar_offset + index * (TAB_HEIGHT + BUTTON_MARGIN)
        button_rect = pygame.Rect(BUTTON_MARGIN, y_pos, self.width - BUTTON_MARGIN * 2, TAB_HEIGHT)
        
        # Draw button with accent color if active
        button_color = ACCENT_COLOR if active else BUTTON_COLOR
        pygame.draw.rect(self.surface, button_color, button_rect, border_radius=5)
        
        # Create a larger font for the tab labels
        icon_font = pygame.font.SysFont(None, TAB_ICON_FONT_SIZE)
        
        # Draw label with larger font
        label_surface = icon_font.render(label, True, TEXT_COLOR)
        label_rect = label_surface.get_rect(center=button_rect.center)
        self.surface.blit(label_surface, label_rect)
        
    def handle_click(self, y):
        """Handle a click on the sidebar.
        
        Args:
            y: The y position of the click
            
        Returns:
            The index of the clicked tab, or None if no tab was clicked
        """
        # Calculate which tab was clicked - adjusting for status bar position
        status_bar_offset = STATUS_BAR_HEIGHT if STATUS_BAR_TOP else 0
        for i in range(4):  # 4 view modes
            y_pos = PLOT_PADDING + status_bar_offset + i * (TAB_HEIGHT + BUTTON_MARGIN)
            if y_pos <= y <= y_pos + TAB_HEIGHT:
                print(f"Tab clicked: {i}")
                return i
        return None


class StatusBar:
    """Status bar for displaying system information."""
    
    def __init__(self, surface, font):
        """Initialize the status bar.
        
        Args:
            surface: The pygame surface to draw on
            font: The font to use for text
        """
        self.surface = surface
        self.font = font
        self.height = STATUS_BAR_HEIGHT
    
    def draw(self, fps=0, nodes=0, runtime="00:04", latency="0.0 ms", signals=3, last_update="Just now"):
        """Draw the status bar.
        
        Args:
            fps: The current FPS
            nodes: The number of connected nodes
            runtime: The runtime text
            latency: The latency text
            signals: The number of signals
            last_update: The last update text
        """
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
