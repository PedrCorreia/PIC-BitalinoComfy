"""
Button controller for the PlotUnit visualization system.

This module connects UI button components with the PlotUnit system,
providing a clean interface for button-based interactions.
"""

import pygame
import logging
import threading
import time
from ..constants import *
from ..ui import ResetButton, ToggleButton
# Configure logger
logger = logging.getLogger('ButtonController')

class ButtonController:
    """
    Controller for managing buttons in the PlotUnit UI.
    
    This class manages the creation, rendering, and event handling for
    UI buttons, creating a clean interface between the button components
    and the main PlotUnit system.
    
    Attributes:
        plot_unit (PlotUnit): Reference to the parent PlotUnit instance
        buttons (list): List of button instances
        initialized (bool): Whether the buttons have been initialized
    """
    
    def __init__(self, plot_unit):
        """
        Initialize the button controller.
        
        Args:
            plot_unit (PlotUnit): Reference to the parent PlotUnit instance
        """
        self.plot_unit = plot_unit
        self.buttons = []
        self.initialized = False
        
        # Start a delayed initialization thread
        self._init_thread = threading.Thread(target=self._delayed_init)
        self._init_thread.daemon = True
        self._init_thread.start()    
    
    def _delayed_init(self):
        """
        Initialize buttons after a delay to ensure pygame is ready.
        
        This method is called from a separate thread to ensure PyGame
        is fully initialized before creating buttons.
        """
        # Wait for pygame to initialize
        time.sleep(2)
        self._init_buttons()
        
    def _init_buttons(self):
        """
        Initialize the buttons for the right sidebar.
        """
        if self.initialized:
            return
        
        logger.info("Initializing right sidebar UI buttons")

        # Sidebar config (match left sidebar)
        sidebar_width = SIDEBAR_WIDTH
        sidebar_height = self.plot_unit.height
        button_height = TAB_HEIGHT + 10  # Match sidebar button height
        button_margin = 15               # Match sidebar button spacing
        button_width = sidebar_width - 2 * ELEMENT_PADDING
        num_buttons = 3

        # Calculate vertical positions for stacked buttons (starting from top)
        buttons_y_start = button_margin + self.plot_unit.status_bar_height
        buttons_x = self.plot_unit.width - sidebar_width + ELEMENT_PADDING

        # Create buttons stacked vertically
        clear_plots_button = ResetButton(
            buttons_x,
            buttons_y_start,
            button_width,
            button_height,
            "Clear Plots",
            self._clear_plots
        )
        clear_registry_button = ResetButton(
            buttons_x,
            buttons_y_start + (button_height + button_margin) * 1,
            button_width,
            button_height,
            "Clear Registry",
            self._clear_registry
        )
        perf_toggle = ToggleButton(
            buttons_x,
            buttons_y_start + (button_height + button_margin) * 2,
            button_width,
            button_height,
            "Performance",
            self.plot_unit.settings.get('performance_mode', False),
            self._toggle_performance
        )

        self.buttons = [
            clear_plots_button,
            clear_registry_button,
            perf_toggle
        ]

        self.initialized = True
        logger.info(f"Initialized {len(self.buttons)} right sidebar buttons")
    def draw(self, surface):
        """
        Draw the right sidebar and its buttons.
        """
        if not self.initialized:
            return

        # Draw sidebar background - full height right sidebar
        sidebar_rect = pygame.Rect(
            self.plot_unit.width - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, self.plot_unit.height
        )
        pygame.draw.rect(surface, SIDEBAR_COLOR, sidebar_rect)
        
        # Draw a divider line to match left sidebar aesthetics
        pygame.draw.line(
            surface, 
            TEXT_COLOR, 
            (self.plot_unit.width - SIDEBAR_WIDTH, 0), 
            (self.plot_unit.width - SIDEBAR_WIDTH, self.plot_unit.height),
            1
        )

        # Draw each button
        font = self.plot_unit.font if hasattr(self.plot_unit, 'font') else None
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.check_hover(mouse_pos)
            button.draw(surface, font)
    
    def handle_events(self, event):
        """
        Handle pygame events for buttons.
        
        Args:
            event (pygame.Event): The pygame event to handle
            
        Returns:
            bool: True if an event was handled, False otherwise
        """
        if not self.initialized:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            for button in self.buttons:
                if button.rect.collidepoint(mouse_pos):
                    return button.handle_click()
                    
        return False

    def _clear_plots(self):
        """
        Clear all plots from the visualization.
        """
        logger.info("Clear plots button clicked")
        if hasattr(self.plot_unit, 'clear_plots'):
            self.plot_unit.clear_plots()
    
    def _clear_registry(self):
        """
        Clear the signal registry.
        """
        logger.info("Clear registry button clicked")
        
        try:
            # Import here to avoid circular imports
            from ....src.registry.signal_registry import SignalRegistry
            SignalRegistry.get_instance().reset()
            logger.info("Signal registry reset")
        except ImportError:
            logger.error("Could not import SignalRegistry")
        except Exception as e:
            logger.error(f"Failed to reset registry: {str(e)}")
    
    def _toggle_performance(self, is_on):
        """
        Toggle performance mode.
        
        Args:
            is_on (bool): Whether performance mode is enabled
        """
        if hasattr(self.plot_unit, 'settings'):
            self.plot_unit.settings['performance_mode'] = is_on
            logger.info(f"Performance mode {'enabled' if is_on else 'disabled'}")
            
        # Update related settings
        if hasattr(self.plot_unit, 'settings'):
            self.plot_unit.settings['smart_downsample'] = is_on
            self.plot_unit.settings['line_thickness'] = 1 if is_on else 2
