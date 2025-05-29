import pygame
from src.plot.constants import ACCENT_COLOR, TEXT_COLOR, BUTTON_COLOR_SETTINGS, TARGET_FPS, STATUS_BAR_HEIGHT, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT, SECTION_MARGIN, CONTROL_MARGIN, TITLE_PADDING, ELEMENT_PADDING
from src.plot.view_mode import ViewMode

class SettingsPanel:
    def __init__(self, font, plot_registry):
        self.font = font
        self.plot_registry = plot_registry
        self.fps_cap_on = True

    def draw(self, screen, plot_x, plot_y, plot_width, plot_height, window_sec_dict, current_viewmode, plotted_signals=None, signal_window_sec=None):
        mx, my = pygame.mouse.get_pos()
        y_cursor = plot_y + SECTION_MARGIN
        section_header_font = pygame.font.SysFont("consolas", 16, bold=True)
        # --- Performance Section ---
        perf_header = section_header_font.render("Performance", True, ACCENT_COLOR)
        screen.blit(perf_header, (plot_x + SECTION_MARGIN, y_cursor))
        y_cursor += perf_header.get_height() + CONTROL_MARGIN
        # FPS Cap Toggle (large, clear button)
        cap_label = self.font.render(f"FPS Cap ({TARGET_FPS} FPS):", True, TEXT_COLOR)
        cap_box = pygame.Rect(plot_x + SECTION_MARGIN, y_cursor, 28, 28)
        pygame.draw.rect(screen, (200, 200, 200), cap_box, border_radius=6)
        if self.fps_cap_on:
            pygame.draw.circle(screen, (0, 220, 0), cap_box.center, 12)
        screen.blit(cap_label, (cap_box.right + CONTROL_MARGIN, cap_box.y + 4))
        y_cursor += cap_box.height + CONTROL_MARGIN
        # Draw horizontal line
        pygame.draw.line(screen, ACCENT_COLOR, (plot_x + SECTION_MARGIN, y_cursor), (plot_x + plot_width - SECTION_MARGIN, y_cursor), 2)
        y_cursor += CONTROL_MARGIN
        # --- Registry Section ---
        reg_header = section_header_font.render("Registry", True, ACCENT_COLOR)
        screen.blit(reg_header, (plot_x + SECTION_MARGIN, y_cursor))
        y_cursor += reg_header.get_height() + CONTROL_MARGIN
        # Refresh button
        refresh_btn_width = 120
        refresh_btn_height = 36
        refresh_btn_x = plot_x + SECTION_MARGIN
        refresh_btn_y = y_cursor
        refresh_rect = pygame.Rect(refresh_btn_x, refresh_btn_y, refresh_btn_width, refresh_btn_height)
        mouse_over_refresh = refresh_rect.collidepoint(mx, my)
        refresh_color = (120, 200, 255) if mouse_over_refresh else ACCENT_COLOR
        pygame.draw.rect(screen, refresh_color, refresh_rect, border_radius=8)
        refresh_label = self.font.render("Refresh", True, (0,0,0))
        screen.blit(refresh_label, (refresh_rect.x + 24, refresh_rect.y + 8))
        y_cursor += refresh_btn_height + CONTROL_MARGIN
        # --- Signal Table Section ---
        table_header = section_header_font.render("Signal Registry Table", True, ACCENT_COLOR)
        screen.blit(table_header, (plot_x + SECTION_MARGIN, y_cursor))
        y_cursor += table_header.get_height() + CONTROL_MARGIN
        # Draw table headers
        table_x = plot_x + SECTION_MARGIN
        table_y = y_cursor
        col_widths = [70, 210, 80, 60]
        row_height = 24
        header = ["Type", "Signal ID", "Rate", "Len"]
        for i, h in enumerate(header):
            h_label = self.font.render(h, True, ACCENT_COLOR)
            screen.blit(h_label, (table_x + sum(col_widths[:i]), table_y))
        row = 1
        signals_drawn = 0
        max_rows = 6
        all_ids = self.plot_registry.get_all_signal_ids()
        for sid in all_ids:
            meta = self.plot_registry.get_signal_metadata(sid) or {}
            meta_type = meta.get('type')
            if meta_type == 'processed':
                sig_type = 'processed'
            else:
                sig_type = 'raw'
            sig_id = str(sid)
            rate = meta.get('sampling_rate', '-')
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
                break
            screen.blit(t_label, (table_x, y))
            screen.blit(id_label, (table_x + col_widths[0], y))
            screen.blit(rate_label, (table_x + col_widths[0] + col_widths[1], y))
            screen.blit(len_label, (table_x + col_widths[0] + col_widths[1] + col_widths[2], y))
            row += 1
            signals_drawn += 1
        if signals_drawn == 0:
            empty_label = self.font.render("No signals in registry", True, (180,80,80))
            screen.blit(empty_label, (table_x, table_y + row_height))
        y_cursor = table_y + (max(row, 2) + 1) * row_height + CONTROL_MARGIN
        # Draw horizontal line
        pygame.draw.line(screen, ACCENT_COLOR, (plot_x + SECTION_MARGIN, y_cursor), (plot_x + plot_width - SECTION_MARGIN, y_cursor), 2)
        y_cursor += CONTROL_MARGIN
        # --- Per-Signal Window Controls Section ---
        sig_header = section_header_font.render("Per-Signal Window Controls", True, ACCENT_COLOR)
        screen.blit(sig_header, (plot_x + SECTION_MARGIN, y_cursor))
        y_cursor += sig_header.get_height() + CONTROL_MARGIN
        # Place per-signal controls on the right, below the section header
        sig_ctrl_x = plot_x + plot_width - 320  # Increase left margin for more space
        sig_ctrl_y = y_cursor
        btn_size = 24
        self._sig_window_btn_rects = []  # (minus_rect, plus_rect, sid)
        for i, sid in enumerate(all_ids):
            y = sig_ctrl_y + i * (btn_size + ELEMENT_PADDING)
            display_id = str(sid) if len(str(sid)) < 18 else str(sid)[:15] + '...'
            label = self.font.render(f"{display_id}", True, (200,220,255))
            screen.blit(label, (sig_ctrl_x, y))
            # Determine view mode for this signal (raw/processed/twin/metrics)
            meta = self.plot_registry.get_signal_metadata(sid) or {}
            meta_type = meta.get('type')
            if meta_type == 'processed':
                view_mode = ViewMode.PROCESSED
            else:
                view_mode = ViewMode.RAW
            win_sec = 2
            if signal_window_sec is not None:
                win_sec = signal_window_sec.get(sid, window_sec_dict.get(view_mode, 2))
            # Add more space between label and window size
            win_label = self.font.render(f"{win_sec} s", True, TEXT_COLOR)
            screen.blit(win_label, (sig_ctrl_x + 200, y))
            minus_rect = pygame.Rect(sig_ctrl_x + 220, y, btn_size, btn_size)
            plus_rect = pygame.Rect(sig_ctrl_x + 260, y, btn_size, btn_size)
            mouse_over_plus = plus_rect.collidepoint(mx, my)
            mouse_over_minus = minus_rect.collidepoint(mx, my)
            plus_color = (0, 255, 120) if mouse_over_plus else ACCENT_COLOR
            minus_color = (255, 120, 0) if mouse_over_minus else ACCENT_COLOR
            pygame.draw.rect(screen, plus_color, plus_rect, border_radius=6)
            pygame.draw.rect(screen, minus_color, minus_rect, border_radius=6)
            plus_label = self.font.render("+", True, (0,0,0))
            minus_label = self.font.render("-", True, (0,0,0))
            screen.blit(plus_label, (plus_rect.x + 6, plus_rect.y + 2))
            screen.blit(minus_label, (minus_rect.x + 6, minus_rect.y + 2))
            self._sig_window_btn_rects.append((minus_rect, plus_rect, sid))
        # Return rects for event handling
        return None  # Not used

    def handle_event(self, event, plot_x, plot_y, row, row_height, plot_height, window_sec_dict, current_viewmode, plotted_signals=None, signal_window_sec=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # FPS cap toggle
            cap_box = pygame.Rect(plot_x + SECTION_MARGIN, plot_y + SECTION_MARGIN, 28, 28)
            if cap_box.collidepoint(mx, my):
                self.fps_cap_on = not self.fps_cap_on
                return
            # Only check per-signal window +/- buttons
            if hasattr(self, '_sig_window_btn_rects') and signal_window_sec is not None:
                for minus_rect, plus_rect, sid in self._sig_window_btn_rects:
                    if minus_rect.collidepoint(mx, my):
                        signal_window_sec[sid] = max(signal_window_sec.get(sid, 2) - 1, 1)
                        return
                    if plus_rect.collidepoint(mx, my):
                        signal_window_sec[sid] = min(signal_window_sec.get(sid, 2) + 1, 60)
                        return
            # Check refresh button
            refresh_btn_x = plot_x + 50
            refresh_btn_y = plot_y + row_height * 8  # Approximate, not critical
            refresh_rect = pygame.Rect(refresh_btn_x, refresh_btn_y, 100, 32)
            if refresh_rect.collidepoint(mx, my):
                pass  # No-op
        elif event.type == pygame.KEYDOWN:
            pass  # (Optional: add keyboard shortcuts for per-signal adjustment)
