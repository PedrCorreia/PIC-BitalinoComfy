def _draw_twin_view(self):
    """Draw raw and processed signals side by side"""
    # Get available signal IDs
    raw_signals = []
    processed_signals = []
    
    with self.data_lock:
        # Find up to 3 raw signals with corresponding processed versions
        for signal_id in self.data:
            if signal_id != 'filtered' and signal_id != 'processed' and signal_id != 'raw':
                # Check if we also have a processed version
                processed_id = f"{signal_id}_processed"
                if processed_id in self.data:
                    raw_signals.append(signal_id)
                    processed_signals.append(processed_id)
                    
                if len(raw_signals) >= 3:  # Limit to 3 signals
                    break
    
    # Fall back to default signals if no paired signals found
    if not raw_signals:
        with self.data_lock:
            raw_data = np.copy(self.data.get('raw', np.zeros(100)))
            processed_data = np.copy(self.data.get('filtered', np.zeros(100)))
        
        # Draw split screen with raw and processed signals
        self._draw_signal_panel(raw_data, "Raw Signal", (220, 180, 0), 
                                self.sidebar_width, 0, self.plot_width // 2 - 5)
        
        self._draw_signal_panel(processed_data, "Processed Signal", (0, 180, 220),
                                self.sidebar_width + self.plot_width // 2 + 5, 0, 
                                self.plot_width // 2 - 5)
    else:
        # Left panel - Raw signals
        panel_width = self.plot_width // 2 - 5
        left_panel_x = self.sidebar_width
        
        # Draw raw panel with title
        title_text = self.font.render("Raw Signals", True, self.text_color)
        self.surface.blit(title_text, (left_panel_x + 10, 10))
        
        # Right panel - Processed signals
        right_panel_x = self.sidebar_width + panel_width + 10
        
        # Draw processed panel with title
        title_text = self.font.render("Processed Signals", True, self.text_color)
        self.surface.blit(title_text, (right_panel_x + 10, 10))
        
        # Calculate height for each signal
        panel_height = (self.height - 80) // min(len(raw_signals), 3)
        
        # Draw each pair of signals
        for i, (raw_id, proc_id) in enumerate(zip(raw_signals, processed_signals)):
            if i >= 3:  # Safety check
                break
            
            # Get signal colors
            raw_color = self.signal_colors.get(raw_id, (220, 180, 0))
            proc_color = self.signal_colors.get(proc_id, (0, 180, 220))
            
            # Calculate vertical position
            y_offset = 40 + (panel_height * i)
            
            # Get signal data
            with self.data_lock:
                raw_data = np.copy(self.data.get(raw_id, np.zeros(100)))
                proc_data = np.copy(self.data.get(proc_id, np.zeros(100)))
            
            # Draw raw signal on left panel
            panel_title = f"{raw_id}"
            self._draw_signal_panel(raw_data, panel_title, raw_color, 
                                    left_panel_x, y_offset, panel_width, 
                                    panel_height - 10)
                                    
            # Draw processed signal on right panel
            panel_title = f"{proc_id}"
            self._draw_signal_panel(proc_data, panel_title, proc_color, 
                                    right_panel_x, y_offset, panel_width, 
                                    panel_height - 10)
                                    
def _draw_multi_plot(self, signal_ids, title, is_processed=False):
    """Draw multiple stacked signals in a single view"""
    if not signal_ids:
        return
        
    # Plot area dimensions
    plot_area_x = self.sidebar_width
    plot_area_y = 40
    plot_area_width = self.plot_width
    
    # Draw title
    title_text = self.font.render(title, True, self.text_color)
    self.surface.blit(title_text, (plot_area_x + 10, 10))
    
    # Calculate height for each signal panel
    panel_count = min(len(signal_ids), 3)  # Limit to max 3 panels
    panel_height = (self.height - 80) // panel_count
    
    # Draw stacked signal panels
    for i, signal_id in enumerate(signal_ids[:panel_count]):
        # Calculate vertical position
        y_offset = plot_area_y + (i * panel_height)
        
        # Get signal color
        color = (0, 180, 220) if is_processed else (220, 180, 0)  # Default colors
        if hasattr(self, 'signal_colors') and signal_id in self.signal_colors:
            color = self.signal_colors[signal_id]
        
        # Get signal data
        with self.data_lock:
            data = np.copy(self.data.get(signal_id, np.zeros(100)))
        
        # Draw panel for this signal
        self._draw_signal_panel(
            data=data,
            title=signal_id,
            color=color,
            x=plot_area_x,
            y=y_offset,
            width=plot_area_width,
            height=panel_height - 10
        )

def _draw_signal_panel(self, data, title, color, x, y, width, height=None):
    """Draw a signal panel with data, grid and stats"""
    if height is None:
        height = self.height - 80
        
    # Panel title (if smaller panels, use smaller font)
    small_font = height < 150
    if small_font:
        title_font = pygame.font.SysFont("Arial", 12)
    else:
        title_font = self.font
        
    title_text = title_font.render(title, True, self.text_color)
    self.surface.blit(title_text, (x + 5, y + 5))
    
    # Draw panel background
    panel_y = y + 25  # Leave space for title
    panel_height = height - 25  # Adjust height to leave space for title
    
    pygame.draw.rect(self.surface, 
                    (25, 25, 25) if not self.settings['light_mode'] else (230, 230, 230), 
                    (x, panel_y, width, panel_height))
    
    # Draw grid
    self._draw_grid(x, panel_y, width, panel_height)
    
    # Draw actual data plot
    if len(data) > 1:
        # Apply caps if enabled
        if self.settings['caps_enabled']:
            data_min = np.min(data)
            data_max = np.max(data)
            
            # If min and max are the same, adjust to avoid division by zero
            if data_min == data_max:
                data_min -= 0.5
                data_max += 0.5
        else:
            # Use fixed range when caps disabled
            data_min = -1.0
            data_max = 1.0
        
        # Calculate scaling factors
        x_scale = width / (len(data) - 1)
        y_scale = panel_height / (data_max - data_min)
        
        # Draw line connecting data points
        points = []
        for i, value in enumerate(data):
            point_x = x + i * x_scale
            point_y = panel_y + panel_height - (value - data_min) * y_scale
            points.append((point_x, point_y))
        
        if len(points) > 1:
            pygame.draw.lines(self.surface, color, False, points, 2)
    
    # Draw stats if there's enough space
    if height > 120:
        stats_y = y + height - 20
        stats_font = pygame.font.SysFont("Arial", 12) if small_font else self.font
        
        # Draw min/max/avg values at bottom
        min_text = stats_font.render(f"Min: {np.min(data):.2f}", True, self.text_color)
        max_text = stats_font.render(f"Max: {np.max(data):.2f}", True, self.text_color)
        avg_text = stats_font.render(f"Avg: {np.mean(data):.2f}", True, self.text_color)
        
        self.surface.blit(min_text, (x + 5, stats_y))
        self.surface.blit(max_text, (x + 90, stats_y))
        self.surface.blit(avg_text, (x + 175, stats_y))
