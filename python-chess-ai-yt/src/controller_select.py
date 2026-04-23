"""Minimal pygame pre-game screen to pick the AI controller + color.

Two screens, click-through:
  1. Which opponent? (None / v1 single-call / v2 GEPA-optimized pipeline)
  2. Which color? (White / Black)  [only shown if an AI was chosen]

Returns a dict {enable_ai, controller_cls, human_color}.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import pygame


@dataclass
class Selection:
    enable_ai: bool
    controller_cls: Optional[type]
    human_color: str  # "white" or "black"


def _draw_button(screen, rect, label, *, hovered: bool, sub: str | None = None,
                 font, sub_font) -> None:
    color = (85, 140, 220) if hovered else (55, 65, 90)
    pygame.draw.rect(screen, color, rect, border_radius=10)
    pygame.draw.rect(screen, (20, 25, 40), rect, width=2, border_radius=10)
    text = font.render(label, True, (255, 255, 255))
    if sub:
        text_rect = text.get_rect(center=(rect.centerx, rect.centery - 12))
        screen.blit(text, text_rect)
        sub_text = sub_font.render(sub, True, (210, 220, 240))
        screen.blit(sub_text, sub_text.get_rect(center=(rect.centerx, rect.centery + 16)))
    else:
        screen.blit(text, text.get_rect(center=rect.center))


def _wait_click(screen, options: list[tuple[str, str | None, object]],
                title: str) -> object:
    """Render buttons, return the payload of whichever is clicked."""
    w, h = screen.get_size()
    title_font = pygame.font.SysFont("Helvetica", 40, bold=True)
    btn_font = pygame.font.SysFont("Helvetica", 26, bold=True)
    sub_font = pygame.font.SysFont("Helvetica", 16)

    btn_w, btn_h, gap = 520, 88, 18
    total_h = len(options) * btn_h + (len(options) - 1) * gap
    y0 = (h - total_h) // 2 + 20
    rects = [pygame.Rect((w - btn_w) // 2, y0 + i * (btn_h + gap), btn_w, btn_h)
             for i in range(len(options))]

    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                for rect, (_, _, payload) in zip(rects, options):
                    if rect.collidepoint(ev.pos):
                        return payload

        screen.fill((24, 28, 40))
        title_surf = title_font.render(title, True, (255, 255, 255))
        screen.blit(title_surf, title_surf.get_rect(center=(w // 2, y0 - 80)))

        mouse = pygame.mouse.get_pos()
        for rect, (label, sub, _) in zip(rects, options):
            _draw_button(screen, rect, label,
                         hovered=rect.collidepoint(mouse),
                         sub=sub, font=btn_font, sub_font=sub_font)

        pygame.display.flip()
        clock.tick(60)


def run_select(screen) -> Selection:
    """Show the two setup screens; return the Selection."""
    # lazy imports to avoid pulling either controller on the wrong config
    def _load_v1():
        from ai_controller import AIController
        return AIController

    def _load_v2():
        from ai_controller_v2 import AIControllerV2
        return AIControllerV2

    opts_ai = [
        ("v2: GEPA-optimized pipeline",
         "3 parallel proposers + selector · OPENROUTER_API_KEY · ~-57% cp vs v1",
         ("v2", _load_v2)),
        ("v1: single-call baseline",
         "hand-written prompt · OPENAI_API_KEY · original harness",
         ("v1", _load_v1)),
        ("No AI (human vs human)",
         "local two-player",
         ("none", None)),
    ]
    picked_ai = _wait_click(screen, opts_ai, "Choose your opponent")

    if picked_ai[0] == "none":
        return Selection(enable_ai=False, controller_cls=None, human_color="white")

    try:
        controller_cls = picked_ai[1]()
    except Exception as e:
        # If loading fails (missing key, etc.), fall through to human vs human
        print(f"❌ Could not load {picked_ai[0]} controller: {e}")
        print("   Falling back to human vs human.")
        return Selection(enable_ai=False, controller_cls=None, human_color="white")

    opts_color = [
        ("Play as White", "AI plays Black", "white"),
        ("Play as Black", "AI plays White", "black"),
    ]
    human_color = _wait_click(screen, opts_color, "Choose your color")

    return Selection(enable_ai=True, controller_cls=controller_cls,
                     human_color=human_color)
