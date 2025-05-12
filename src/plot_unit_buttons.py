import pygame
from .plot_unit import PlotUnit
import threading
import time

# Custom Button class for the PlotUnit GUI
class ResetButton:
    def __init__(self, x, y, width, height, text, color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255))
        self.active_color = color
        self.action = action
        self.is_hovered = False
        
    def draw(self, surface):
        # Choose color based on hover state
        current_color = self.hover_color if self.is_hovered else self.active_color
        
        # Draw button rectangle
        pygame.draw.rect(surface, current_color, self.rect, border_radius=5)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2, border_radius=5)  # Button border
        
        # Draw text
        font = pygame.font.SysFont('Arial', 16)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
    
    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                self.action()
                return True
        return False

# Extended PlotUnit class with buttons
class PlotUnitWithButtons(PlotUnit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Wait for pygame to be initialized before creating buttons
        self._buttons_initialized = False
        self._buttons = []
        
        # Add a deferred initialization method to be called once pygame is initialized
        self._add_gui_elements_thread = threading.Thread(target=self._delayed_init_buttons)
        self._add_gui_elements_thread.daemon = True
        self._add_gui_elements_thread.start()
    
    def _delayed_init_buttons(self):
        """Initialize buttons after a delay to ensure pygame is ready"""
        time.sleep(2)  # Wait for pygame to initialize
        self._init_buttons()
    
    def _init_buttons(self):
        """Initialize the buttons for the GUI"""
        if self._buttons_initialized:
            return
            
        print("[DEBUG-GUI] Initializing PlotUnit buttons")
        
        # Create buttons
        button_width = 120
        button_height = 30
        button_margin = 10
        
        # Button positions - positioned at the top of the window
        buttons_y = 10
        
        # Clear plots button
        clear_plots_button = ResetButton(
            button_margin, buttons_y, button_width, button_height,
            "Clear Plots", (0, 80, 150), self.clear_plots
        )
        
        # Clear registry button
        clear_registry_button = ResetButton(
            button_margin * 2 + button_width, buttons_y, button_width, button_height,
            "Clear Registry", (150, 80, 0), self.clear_registry
        )
        
        # Add buttons to the list
        self._buttons = [clear_plots_button, clear_registry_button]
        self._buttons_initialized = True
        print(f"[DEBUG-GUI] {len(self._buttons)} buttons initialized")
    
    def clear_plots(self):
        """Clear all plots"""
        print("[DEBUG-GUI] Clear plots button clicked")
        if hasattr(self, 'signals'):
            self.signals = {}
            print("[DEBUG-GUI] Cleared all plot signals")
    
    def clear_registry(self):
        """Clear signal registry"""
        print("[DEBUG-GUI] Clear registry button clicked")
        try:
            from ..comfy.mock_signal_node import SignalRegistry
            SignalRegistry.reset()
            print("[DEBUG-GUI] Signal registry reset")
        except ImportError:
            print("[ERROR-GUI] Could not import SignalRegistry")
        except Exception as e:
            print(f"[ERROR-GUI] Failed to reset registry: {str(e)}")
    
    def draw(self):
        """Override the draw method to include buttons"""
        # Call the parent class draw method
        super().draw()
        
        # Check if buttons are initialized and pygame is running
        if not self._buttons_initialized or not hasattr(self, 'screen'):
            return
        
        # Process mouse events for hover effects
        mouse_pos = pygame.mouse.get_pos()
        for button in self._buttons:
            button.check_hover(mouse_pos)
            button.draw(self.screen)
    
    def process_event(self, event):
        """Process pygame events"""
        # Let parent class handle the event first
        super().process_event(event)
        
        # Then handle button events
        if self._buttons_initialized:
            for button in self._buttons:
                if button.handle_event(event):
                    break  # Stop after first button that handles the event
    
    def clear_plots(self):
        """Clear all plots"""
        print("[DEBUG-GUI] Clear plots button clicked")
        if hasattr(self, 'signals'):
            self.signals = {}
            print("[DEBUG-GUI] Cleared all plot signals")

# Replace the original PlotUnit with our extended version
# This line will be imported by the PlotUnit singleton mechanism
PlotUnit = PlotUnitWithButtons
