import pygame
from src.plot.constants import ACCENT_COLOR, TEXT_COLOR, BUTTON_COLOR_SETTINGS, TARGET_FPS, STATUS_BAR_HEIGHT, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT

class SettingsPanel:
    def __init__(self, font, plot_registry):
        self.font = font
        self.plot_registry = plot_registry
        self.fps_cap_on = True
        self.window_sec = 2

    def draw(self, screen, plot_x, plot_y, plot_width, plot_height, plotted_signals=None):
        # Calculate available area
        available_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * plot_x
        available_height = WINDOW_HEIGHT - STATUS_BAR_HEIGHT - plot_y - 20
        # FPS Cap Toggle
        cap_label = self.font.render(f"FPS Cap ({TARGET_FPS} FPS):", True, TEXT_COLOR)
        cap_box = pygame.Rect(plot_x + 20, plot_y + 20, 20, 20)
        pygame.draw.rect(screen, (200, 200, 200), cap_box, border_radius=4)
        if self.fps_cap_on:
            pygame.draw.circle(screen, (0, 220, 0), cap_box.center, 8)
        screen.blit(cap_label, (cap_box.right + 10, cap_box.y))
        # Table layout
        table_x = plot_x + 20
        table_y = cap_box.bottom + 20
        # Compact columns: Type, Signal ID, Rate, Len
        col_widths = [70, 210, 80, 60]  # More compact
        row_height = max(22, int(plot_height * 0.045))
        header = ["Type", "Signal ID", "Rate", "Len"]
        for i, h in enumerate(header):
            h_label = self.font.render(h, True, ACCENT_COLOR)
            screen.blit(h_label, (table_x + sum(col_widths[:i]), table_y))
        row = 1
        signals_drawn = 0
        # Calculate max_rows, always allow at least 3
        max_rows = max(3, int((plot_height - 180) // row_height) - 2)
        all_ids = self.plot_registry.get_all_signal_ids()
        print(f"[DEBUG][SettingsPanel] all_ids: {all_ids}, max_rows: {max_rows}")
        for sid in all_ids:
            meta = self.plot_registry.get_signal_metadata(sid) or {}
            print(f"[DEBUG][SettingsPanel] sid: {sid}, meta: {meta}, row: {row}, max_rows: {max_rows}")
            meta_type = meta.get('type')
            if meta_type == 'processed':
                sig_type = 'processed'
            else:
                sig_type = 'raw'
            sig_id = str(sid)
            rate = meta.get('sampling_rate', '-')
            # Get signal length robustly
            data = self.plot_registry.get_signal(sid)
            sig_len = 0
            if isinstance(data, dict) and 'v' in data:
                sig_len = len(data['v'])
            elif hasattr(data, '__len__'):
                sig_len = len(data)
            t_label = self.font.render(sig_type, True, (180,180,180))
            display_id = sig_id if len(sig_id) < 32 else sig_id[:29] + '...'
            id_label = self.font.render(display_id, True, (220,220,220))
            rate_label = self.font.render(str(rate), True, (180,200,180))
            len_label = self.font.render(str(sig_len), True, (200,200,255))
            y = table_y + row * row_height
            if row > max_rows:
                print(f"[DEBUG][SettingsPanel] row {row} > max_rows {max_rows}, breaking loop")
                break
            screen.blit(t_label, (table_x, y))
            screen.blit(id_label, (table_x + col_widths[0], y))
            screen.blit(rate_label, (table_x + col_widths[0] + col_widths[1], y))
            screen.blit(len_label, (table_x + col_widths[0] + col_widths[1] + col_widths[2], y))
            row += 1
            signals_drawn += 1
        # If no signals, show a message
        if signals_drawn == 0:
            empty_label = self.font.render("No signals in registry", True, (180,80,80))
            screen.blit(empty_label, (table_x, table_y + row_height))
        # Place rolling window controls below the table, always visible and not overlapping
        controls_y = table_y + (max(row, 2) + 1) * row_height + 20
        if controls_y + 40 > plot_y + plot_height:
            controls_y = plot_y + plot_height - 50
        window_label = self.font.render(f"Rolling Window: {self.window_sec} s", True, TEXT_COLOR)
        screen.blit(window_label, (plot_x + 50, controls_y))
        # Button layout: ensure they fit and are visually robust
        btn_size = min(30, max(20, int(row_height * 1.1)))
        minus_rect = pygame.Rect(plot_x + 50, controls_y + 28, btn_size, btn_size)
        plus_rect = pygame.Rect(plot_x + 120, controls_y + 28, btn_size, btn_size)
        # Get mouse position for visual feedback
        mx, my = pygame.mouse.get_pos()
        # Visual feedback for buttons
        mouse_over_plus = plus_rect.collidepoint(mx, my)
        mouse_over_minus = minus_rect.collidepoint(mx, my)
        plus_color = (0, 255, 120) if mouse_over_plus else ACCENT_COLOR
        minus_color = (255, 120, 0) if mouse_over_minus else ACCENT_COLOR
        pygame.draw.rect(screen, plus_color, plus_rect, border_radius=6)
        pygame.draw.rect(screen, minus_color, minus_rect, border_radius=6)
        plus_label = self.font.render("+", True, (0,0,0))
        minus_label = self.font.render("-", True, (0,0,0))
        screen.blit(plus_label, (plus_rect.x + btn_size//3, plus_rect.y + btn_size//6))
        screen.blit(minus_label, (minus_rect.x + btn_size//3, minus_rect.y + btn_size//6))
        # Add a Refresh button further below
        refresh_btn_width = 100
        refresh_btn_height = 32
        refresh_btn_x = plot_x + 50
        refresh_btn_y = controls_y + btn_size + 40  # More space below controls
        refresh_rect = pygame.Rect(refresh_btn_x, refresh_btn_y, refresh_btn_width, refresh_btn_height)
        mouse_over_refresh = refresh_rect.collidepoint(mx, my)
        refresh_color = (120, 200, 255) if mouse_over_refresh else ACCENT_COLOR
        pygame.draw.rect(screen, refresh_color, refresh_rect, border_radius=8)
        refresh_label = self.font.render("Refresh", True, (0,0,0))
        screen.blit(refresh_label, (refresh_rect.x + 16, refresh_rect.y + 6))
        # Return refresh_rect for event handling
        return cap_box, plus_rect, minus_rect, row, refresh_rect

    def handle_event(self, event, plot_x, plot_y, row, row_height, plot_height, plotted_signals=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            cap_box = pygame.Rect(plot_x + 20, plot_y + 20, 20, 20)
            available_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * plot_x
            available_height = WINDOW_HEIGHT - STATUS_BAR_HEIGHT - plot_y - 20
            col_widths = [max(120, int(available_width * 0.25)), max(180, int(available_width * 0.55)), max(60, int(available_width * 0.15))]
            table_y = cap_box.bottom + 20
            label_y = table_y + (row + 1) * row_height
            if label_y + row_height > plot_y + plot_height - 20:
                label_y = plot_y + plot_height - 50
            plus_rect = pygame.Rect(plot_x + 120, label_y - 5, 30, 30)
            minus_rect = pygame.Rect(plot_x + 50, label_y - 5, 30, 30)
            # Add refresh_rect for event handling
            refresh_btn_width = 100
            refresh_btn_height = 32
            refresh_btn_x = plot_x + 50
            refresh_btn_y = label_y + 30
            refresh_rect = pygame.Rect(refresh_btn_x, refresh_btn_y, refresh_btn_width, refresh_btn_height)
            if cap_box.collidepoint(mx, my):
                self.fps_cap_on = not self.fps_cap_on
            if plus_rect.collidepoint(mx, my):
                self.window_sec = min(self.window_sec + 1, 60)
            if minus_rect.collidepoint(mx, my):
                self.window_sec = max(self.window_sec - 1, 1)
            if refresh_rect.collidepoint(mx, my):
                # No-op: table will refresh on next draw automatically
                pass
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.window_sec = max(self.window_sec - 1, 1)
            elif event.key == pygame.K_RIGHT:
                self.window_sec = min(self.window_sec + 1, 60)
