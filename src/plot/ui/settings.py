import pygame
from src.plot.constants import ACCENT_COLOR, TEXT_COLOR, BUTTON_COLOR_SETTINGS, TARGET_FPS, STATUS_BAR_HEIGHT, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT

class SettingsPanel:
    def __init__(self, font, plot_registry):
        self.font = font
        self.plot_registry = plot_registry
        self.fps_cap_on = True
        self.window_sec = 2

    def draw(self, screen, plot_x, plot_y, plot_width, plot_height):
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
        col_widths = [max(80, int(plot_width * 0.18)), max(180, int(plot_width * 0.60)), max(60, int(plot_width * 0.18))]
        row_height = max(22, int(plot_height * 0.045))
        header = ["Type", "Signal ID", "Size"]
        for i, h in enumerate(header):
            h_label = self.font.render(h, True, ACCENT_COLOR)
            screen.blit(h_label, (table_x + sum(col_widths[:i]), table_y))
        types = ["raw", "processed"]
        row = 1
        signals_drawn = 0
        max_rows = int((plot_height - 180) // row_height) - 2
        # Use debug=False to avoid interfering with registry state
        for sig_type in types:
            signals = self.plot_registry.get_signals_by_type(None, sig_type, debug=False)
            for sig in signals:
                sig_id = str(sig['id'])
                size = len(sig['v']) if hasattr(sig['v'], '__len__') else '?'
                t_label = self.font.render(sig_type, True, (180,180,180))
                display_id = sig_id if len(sig_id) < 32 else sig_id[:29] + '...'
                id_label = self.font.render(display_id, True, (220,220,220))
                size_label = self.font.render(str(size), True, (180,220,180))
                y = table_y + row * row_height
                if row > max_rows:
                    break
                screen.blit(t_label, (table_x, y))
                screen.blit(id_label, (table_x + col_widths[0], y))
                screen.blit(size_label, (table_x + col_widths[0] + col_widths[1], y))
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
        # Use accent color for buttons
        pygame.draw.rect(screen, ACCENT_COLOR, plus_rect, border_radius=6)
        pygame.draw.rect(screen, ACCENT_COLOR, minus_rect, border_radius=6)
        plus_label = self.font.render("+", True, (0,0,0))
        minus_label = self.font.render("-", True, (0,0,0))
        screen.blit(plus_label, (plus_rect.x + btn_size//3, plus_rect.y + btn_size//6))
        screen.blit(minus_label, (minus_rect.x + btn_size//3, minus_rect.y + btn_size//6))
        return cap_box, plus_rect, minus_rect, row

    def handle_event(self, event, plot_x, plot_y, row, row_height, plot_height):
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
            if cap_box.collidepoint(mx, my):
                self.fps_cap_on = not self.fps_cap_on
            if plus_rect.collidepoint(mx, my):
                self.window_sec = min(self.window_sec + 1, 60)
            if minus_rect.collidepoint(mx, my):
                self.window_sec = max(self.window_sec - 1, 1)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.window_sec = max(self.window_sec - 1, 1)
            elif event.key == pygame.K_RIGHT:
                self.window_sec = min(self.window_sec + 1, 60)
