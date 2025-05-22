"""
ProcessStatusBar module for the PlotUnit visualization system.

This module provides a process status bar component for the PlotUnit system,
displaying ECG process metrics such as HR, RR, SCL, and SCK.
"""

import pygame
from ..constants import *

class ProcessStatusBar:
    """
    Process status bar component for PlotUnit.
    Displays HR (heart rate), RR, SCL, and SCK metrics.
    """
    def __init__(self, surface, height, font):
        """
        Initialize the process status bar.
        Args:
            surface (pygame.Surface): The surface to draw on
            height (int): Height of the status bar
            font (pygame.font.Font): Font for text rendering
        """
        self.surface = surface
        self.width = self.surface.get_width()
        self.height = height
        self.font = font
        self.hr = None
        self.rr = None
        self.scl = None
        self.sck = None

    def update_metrics(self, hr=None, rr=None, scl=None, sck=None):
        """
        Update the metrics to be displayed.
        """
        if hr is not None:
            self.hr = hr
        if rr is not None:
            self.rr = rr
        if scl is not None:
            self.scl = scl
        if sck is not None:
            self.sck = sck

    def draw(self):
        """
        Draw the process status bar with current metrics.
        """
        left = SIDEBAR_WIDTH + ELEMENT_PADDING
        y_center = (self.height) // 2
        x = left
        # Draw status bar background
        bar_rect = pygame.Rect(0, self.surface.get_height() - self.height, self.width, self.height)
        pygame.draw.rect(self.surface, SIDEBAR_COLOR, bar_rect)
        # HR
        hr_text = f"HR: {self.hr:.1f} bpm" if self.hr is not None else "HR: --"
        hr_surface = self.font.render(hr_text, True, TEXT_COLOR)
        self.surface.blit(hr_surface, (x, self.surface.get_height() - self.height + y_center - hr_surface.get_height() // 2))
        x += hr_surface.get_width() + SECTION_MARGIN
        # RR
        rr_text = f"RR: {self.rr:.2f} s" if self.rr is not None else "RR: --"
        rr_surface = self.font.render(rr_text, True, TEXT_COLOR)
        self.surface.blit(rr_surface, (x, self.surface.get_height() - self.height + y_center - rr_surface.get_height() // 2))
        x += rr_surface.get_width() + SECTION_MARGIN
        # SCL
        scl_text = f"SCL: {self.scl:.2f}" if self.scl is not None else "SCL: --"
        scl_surface = self.font.render(scl_text, True, TEXT_COLOR)
        self.surface.blit(scl_surface, (x, self.surface.get_height() - self.height + y_center - scl_surface.get_height() // 2))
        x += scl_surface.get_width() + SECTION_MARGIN
        # SCK
        sck_text = f"SCK: {self.sck:.2f}" if self.sck is not None else "SCK: --"
        sck_surface = self.font.render(sck_text, True, TEXT_COLOR)
        self.surface.blit(sck_surface, (x, self.surface.get_height() - self.height + y_center - sck_surface.get_height() // 2))
