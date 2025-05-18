#!/usr/bin/env python
"""
Minimal Working Version: PIC-2025 Visualization Interface
- Real-time plotting of synthetic signals (RAW, PROCESSED, TWIN, SETTINGS tabs)
- No registry/adapter complexity, just direct synthetic signal generation
- Maintains the same layout and UI structure as robust_view_switcher.py
"""
import pygame
import numpy as np
import time
from collections import deque
from src.plot.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, SIDEBAR_WIDTH, STATUS_BAR_HEIGHT,
    BACKGROUND_COLOR, SIDEBAR_COLOR, STATUS_COLOR, BUTTON_COLOR, BUTTON_COLOR_SETTINGS,
    ACCENT_COLOR, TEXT_COLOR, VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS,
    PLOT_PADDING, TAB_HEIGHT, CONTROL_PANEL_HEIGHT, TWIN_VIEW_SEPARATOR,
    BUTTON_MARGIN, SECTION_MARGIN, TITLE_PADDING, TEXT_MARGIN, CONTROL_MARGIN, ELEMENT_PADDING,
    RAW_SIGNAL_COLOR, PROCESSED_SIGNAL_COLOR
)

TABS = [VIEW_MODE_RAW, VIEW_MODE_PROCESSED, VIEW_MODE_TWIN, VIEW_MODE_SETTINGS]
TAB_ICONS = ['R', 'P', 'T', 'S']

# --- Synthetic Signal Generator ---
class Signal:
    def __init__(self, name, color, kind, freq, noise, phase=0.0, amp=1.0):
        self.name = name
        self.color = color
        self.kind = kind
        self.freq = freq
        self.noise = noise
        self.phase = phase
        self.amp = amp
        self.t_deque = deque(maxlen=1200)
        self.v_deque = deque(maxlen=1200)
    def update(self, t):
        if self.kind == 'sine':
            v = self.amp * np.sin(2 * np.pi * self.freq * t + self.phase) + np.random.normal(0, self.noise)
        elif self.kind == 'square':
            v = self.amp * np.sign(np.sin(2 * np.pi * self.freq * t + self.phase)) + np.random.normal(0, self.noise)
        elif self.kind == 'sawtooth':
            v = self.amp * (2 * ((self.freq * t + self.phase/(2*np.pi)) % 1) - 1) + np.random.normal(0, self.noise)
        else:
            v = np.random.normal(0, self.noise)
        self.t_deque.append(t)
        self.v_deque.append(v)
    def get_window(self, window_sec):
        t = np.array(self.t_deque)
        v = np.array(self.v_deque)
        if len(t) < 2:
            return t, v
        t0 = t[-1] - window_sec
        mask = t >= t0
        return t[mask], v[mask]

# --- Main App ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PIC-2025 Minimal Visualization")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    # --- Signals ---
    signals = [
        Signal('SINE_RAW', RAW_SIGNAL_COLOR, 'sine', 0.5, 0.08, 0, 0.9),
        Signal('SQUARE_RAW', (255,120,0), 'square', 0.3, 0.1, 0, 0.8),
        Signal('SAWTOOTH_RAW', (255,60,0), 'sawtooth', 0.4, 0.07, 0, 0.7),
        Signal('SINE_PROCESSED', PROCESSED_SIGNAL_COLOR, 'sine', 0.5, 0.01, 0.2, 0.9),
        Signal('SQUARE_PROCESSED', (0,120,255), 'square', 0.3, 0.01, 0.1, 0.8),
        Signal('SAWTOOTH_PROCESSED', (0,60,255), 'sawtooth', 0.4, 0.01, 0.1, 0.7),
    ]
    t0 = time.time()
    selected_tab = 0
    window_sec = 10
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i in range(len(TABS)):
                    tab_y = 50 + i * (TAB_HEIGHT + BUTTON_MARGIN)
                    if 0 <= mx < SIDEBAR_WIDTH and tab_y <= my < tab_y + TAB_HEIGHT:
                        selected_tab = i
        # --- Update signals ---
        now = time.time() - t0
        for s in signals:
            s.update(now)
        # --- Draw UI ---
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, SIDEBAR_COLOR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(screen, STATUS_COLOR, (0, 0, WINDOW_WIDTH, STATUS_BAR_HEIGHT))
        plot_x = SIDEBAR_WIDTH + PLOT_PADDING
        plot_y = STATUS_BAR_HEIGHT + PLOT_PADDING
        plot_width = WINDOW_WIDTH - SIDEBAR_WIDTH - 2 * PLOT_PADDING
        plot_height = (WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING) // 3
        pygame.draw.rect(screen, (20,20,20), (plot_x, plot_y, plot_width, WINDOW_HEIGHT - STATUS_BAR_HEIGHT - 2 * PLOT_PADDING))
        # --- Sidebar ---
        for i, label in enumerate(TABS):
            tab_y = 50 + i * (TAB_HEIGHT + BUTTON_MARGIN)
            tab_rect = pygame.Rect(0, tab_y, SIDEBAR_WIDTH, TAB_HEIGHT)
            color = ACCENT_COLOR if i == selected_tab else BUTTON_COLOR
            pygame.draw.rect(screen, color, tab_rect)
            # Center the icon vertically and horizontally
            icon = TAB_ICONS[i]
            icon_surface = font.render(icon, True, TEXT_COLOR)
            icon_x = 10 + (SIDEBAR_WIDTH - 20 - icon_surface.get_width()) // 2
            icon_y = tab_y + (TAB_HEIGHT - icon_surface.get_height()) // 2
            screen.blit(icon_surface, (icon_x, icon_y))
        # --- Plotting ---
        if selected_tab == 3:  # SETTINGS
            window_label = font.render(f"Rolling Window (s): {window_sec}", True, TEXT_COLOR)
            screen.blit(window_label, (plot_x + 50, plot_y + 30))
        elif selected_tab == 0:  # RAW
            for i, s in enumerate(signals[:3]):
                t, v = s.get_window(window_sec)
                if len(t) < 2:
                    continue
                t = t - t[0]
                vmin, vmax = np.min(v), np.max(v)
                if vmax == vmin:
                    vmax = vmin + 1
                points = [(plot_x + j * plot_width // len(t), plot_y + i*plot_height + plot_height - int((v[j]-vmin)/(vmax-vmin)*plot_height)) for j in range(len(t))]
                if len(points) >= 2:
                    pygame.draw.lines(screen, s.color, False, points, 2)
                label = font.render(f"{s.name} t=[{t[0]:.1f},{t[-1]:.1f}]", True, TEXT_COLOR)
                screen.blit(label, (plot_x + 10, plot_y + i*plot_height + 10))
        elif selected_tab == 1:  # PROCESSED
            for i, s in enumerate(signals[3:]):
                t, v = s.get_window(window_sec)
                if len(t) < 2:
                    continue
                t = t - t[0]
                vmin, vmax = np.min(v), np.max(v)
                if vmax == vmin:
                    vmax = vmin + 1
                points = [(plot_x + j * plot_width // len(t), plot_y + i*plot_height + plot_height - int((v[j]-vmin)/(vmax-vmin)*plot_height)) for j in range(len(t))]
                if len(points) >= 2:
                    pygame.draw.lines(screen, s.color, False, points, 2)
                label = font.render(f"{s.name} t=[{t[0]:.1f},{t[-1]:.1f}]", True, TEXT_COLOR)
                screen.blit(label, (plot_x + 10, plot_y + i*plot_height + 10))
        elif selected_tab == 2:  # TWIN
            left_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            right_width = plot_width // 2 - TWIN_VIEW_SEPARATOR
            center_x = plot_x + left_width + TWIN_VIEW_SEPARATOR
            pygame.draw.line(screen, (50, 50, 50), (center_x, plot_y), (center_x, WINDOW_HEIGHT - PLOT_PADDING), 3)
            # Left: processed
            for i, s in enumerate(signals[3:]):
                t, v = s.get_window(window_sec)
                if len(t) < 2:
                    continue
                t = t - t[0]
                vmin, vmax = np.min(v), np.max(v)
                if vmax == vmin:
                    vmax = vmin + 1
                points = [(plot_x + j * left_width // len(t), plot_y + i*plot_height + plot_height - int((v[j]-vmin)/(vmax-vmin)*plot_height)) for j in range(len(t))]
                if len(points) >= 2:
                    pygame.draw.lines(screen, s.color, False, points, 2)
                label = font.render(f"{s.name}", True, TEXT_COLOR)
                screen.blit(label, (plot_x + 10, plot_y + i*plot_height + 10))
            # Right: raw
            for i, s in enumerate(signals[:3]):
                t, v = s.get_window(window_sec)
                if len(t) < 2:
                    continue
                t = t - t[0]
                vmin, vmax = np.min(v), np.max(v)
                if vmax == vmin:
                    vmax = vmin + 1
                points = [(center_x + 5 + j * right_width // len(t), plot_y + i*plot_height + plot_height - int((v[j]-vmin)/(vmax-vmin)*plot_height)) for j in range(len(t))]
                if len(points) >= 2:
                    pygame.draw.lines(screen, s.color, False, points, 2)
                label = font.render(f"{s.name}", True, TEXT_COLOR)
                screen.blit(label, (center_x + 15, plot_y + i*plot_height + 10))
        # --- Status Bar ---
        status = f"Mode: {TABS[selected_tab]} | FPS: {int(clock.get_fps())} | Runtime: {int(now)}s"
        txt = font.render(status, True, TEXT_COLOR)
        screen.blit(txt, (10, 6))
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
