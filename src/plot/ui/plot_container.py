from ..constants import *
import pygame

DEFAULT_MARGIN = 15  # fallback if not in constants

class PlotContainer:
    """
    Container for dynamically stacking and sizing plot areas in the PlotUnit visualization system.

    Supports both single (vertical stack) and twin (side-by-side) view modes. Automatically resizes and arranges plots
    based on window/sidebar dimensions and the number of plots.
    """
    def __init__(self, window_rect, sidebar_width, margin=None, twin_view=False):
        if margin is None:
            margin = PLOT_PADDING if 'PLOT_PADDING' in globals() else DEFAULT_MARGIN
        self.window_rect = window_rect  # pygame.Rect of the main window
        self.sidebar_width = sidebar_width
        self.margin = margin
        self.twin_view = twin_view
        self.plots = []  # List of plot area objects

    def add_plot(self, plot):
        self.plots.append(plot)
        self.update_layout()

    def remove_plot(self, plot):
        if plot in self.plots:
            self.plots.remove(plot)
            self.update_layout()

    def set_twin_view(self, twin):
        self.twin_view = twin
        self.update_layout()

    def set_window_rect(self, window_rect, sidebar_width=None):
        self.window_rect = window_rect
        if sidebar_width is not None:
            self.sidebar_width = sidebar_width
            self.update_layout()
    
    def update_layout(self):
        n = len(self.plots)
        if n == 0:
            return
        # Calculate available area for plots
        x0 = self.sidebar_width + self.margin
        y0 = self.margin + STATUS_BAR_HEIGHT  # leave space for status bar
        width = self.window_rect.width - self.sidebar_width - 2 * self.margin
        height = self.window_rect.height - y0 - self.margin
        
        # Make sure width and height are multiples of the grid size (50px)
        # This ensures perfect alignment with grid
        grid_size = 50
        width = (width // grid_size) * grid_size
        height = (height // grid_size) * grid_size

        if self.twin_view:
            # Split horizontally for twin view - ensure both sides have equal grid alignment
            col_width = (width - self.margin) // 2
            col_width = (col_width // grid_size) * grid_size  # Make sure it's a multiple of grid size
            
            # Count how many signals we're dealing with
            signal_pairs = 0
            for plot in self.plots:
                if hasattr(plot, 'get_available_signals'):
                    try:
                        raw_signals = plot.get_available_signals(filter_processed=True)
                        raw_count = len(raw_signals)
                        signal_pairs = max(signal_pairs, raw_count)
                    except:
                        # Default if we can't get signal count
                        signal_pairs = 3
                else:
                    signal_pairs = 3
            
            # Minimum of 1 pair
            signal_pairs = max(1, signal_pairs)
            
            # Calculate height per signal pair - ensure it's a multiple of grid size
            # We want at least 2 grid cells per signal
            min_height_per_signal = grid_size * 3
            
            # Calculate available height and distribute equally
            available_height = height - ((signal_pairs - 1) * self.margin)
            height_per_signal = max(available_height // signal_pairs, min_height_per_signal)
            height_per_signal = (height_per_signal // grid_size) * grid_size
            
            # Track the current y position as we add plots
            current_y = y0
            
            for plot in self.plots:
                # Create the rectangles for this plot
                total_plot_height = height_per_signal * signal_pairs + (signal_pairs - 1) * self.margin
                
                left_rect = pygame.Rect(x0, current_y, col_width, total_plot_height)
                right_rect = pygame.Rect(x0 + col_width + self.margin, current_y, col_width, total_plot_height)
                
                # Set the rectangles on the plot
                if hasattr(plot, 'set_rects'):
                    plot.set_rects([left_rect, right_rect])
                else:
                    plot.set_rect(left_rect)  # fallback
                
                # Move to the next y position
                current_y += total_plot_height + self.margin
        else:
            # For single view, distribute space evenly
            # Make sure each plot gets a height that's a multiple of the grid size
            
            # Calculate minimum height per plot
            min_height = grid_size * 3  # At least 3 grid cells tall
            
            # Calculate available height and distribute equally
            available_height = height - ((n - 1) * self.margin)
            height_per_plot = max(available_height // n, min_height)
            height_per_plot = (height_per_plot // grid_size) * grid_size
            
            for i, plot in enumerate(self.plots):
                # Create rectangle with equal height allocation
                rect = pygame.Rect(x0, y0 + i * (height_per_plot + self.margin), width, height_per_plot)
                plot.set_rect(rect)

    def draw(self):
        for plot in self.plots:
            # View classes manage their own surface, no need to pass it
            plot.draw()

    def handle_event(self, event):
        for plot in self.plots:
            if hasattr(plot, 'handle_event'):
                plot.handle_event(event)

    def draw_grid(self, surface, x, y, width, height, bg_color=None, grid_color=None):
        """
        Draw a grid background for signal plots.
        
        Args:
            surface (pygame.Surface): Surface to draw on
            x (int): X coordinate of the grid top-left corner
            y (int): Y coordinate of the grid top-left corner
            width (int): Width of the grid
            height (int): Height of the grid
            bg_color (tuple, optional): Background color (RGB). Uses BACKGROUND_COLOR if None.
            grid_color (tuple, optional): Grid line color (RGB). Uses GRID_COLOR if None.
        """
        # Determine background color based on light/dark mode
        bg_color = bg_color if bg_color is not None else BACKGROUND_COLOR
        grid_color = grid_color if grid_color is not None else GRID_COLOR
        
        # Draw background
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, bg_color, rect)
        
        # Calculate grid spacing to ensure we have even divisions
        h_spacing = 50  # Fixed grid size
        v_spacing = 50  # Fixed grid size
        
        # Draw grid lines - horizontal
        for i in range(0, width + 1, h_spacing):
            line_x = x + i
            pygame.draw.line(surface, grid_color, (line_x, y), (line_x, y + height))
        
        # Draw grid lines - vertical
        for i in range(0, height + 1, v_spacing):
            line_y = y + i
            pygame.draw.line(surface, grid_color, (x, line_y), (x + width, line_y))

