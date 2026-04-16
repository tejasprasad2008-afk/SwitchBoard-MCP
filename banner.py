"""Startup banner with gradient ASCII art.

Printed once when the MCP server starts. Skipped automatically during pytest.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.text import Text

# ── Constants ──────────────────────────────────────────────────────────

_PEACH = "#FFAB91"
_PINK = "#F48FB1"

_SUBTITLE = "v0.1.0  \u2022  MIT License  \u2022  github.com/tejasprasad2008-afk/switchboard-mcp"
_TAGLINE = "never (gonna) let your agent hit a dead end"

# ── Banner state ──────────────────────────────────────────────────────

_printed = False


def _interpolate(start_hex: str, end_hex: str, t: float) -> str:
    """Linearly interpolate between two hex colours.  *t* in [0, 1]."""
    sr = int(start_hex[1:3], 16)
    sg = int(start_hex[3:5], 16)
    sb = int(start_hex[5:7], 16)
    er = int(end_hex[1:3], 16)
    eg = int(end_hex[3:5], 16)
    eb = int(end_hex[5:7], 16)
    r = int(sr + (er - sr) * t)
    g = int(sg + (eg - sg) * t)
    b = int(sb + (eb - sb) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def print_banner() -> None:
    """Print the SwitchBoard startup banner.

    Skips entirely if ``pytest`` is loaded (so tests stay quiet).
    Prints at most once per process lifetime.
    """
    global _printed

    # ── Skip during test runs ────────────────────────────────────────
    if "pytest" in sys.modules:
        return

    if _printed:
        return
    _printed = True

    # ── Build gradient-colored ASCII art ─────────────────────────────
    try:
        import pyfiglet
        ascii_art = pyfiglet.figlet_format("SwitchBoard", font="slant", width=200)
    except Exception:
        # Fallback: plain-text header if pyfiglet is unavailable
        ascii_art = "SwitchBoard\n"

    console = Console()

    # Count non-newline, non-space characters to compute gradient stops
    visible_chars = [c for c in ascii_art if c not in ("\n", " ")]
    total = len(visible_chars)

    if total == 0:
        # Safety net — render something reasonable
        console.print("[bold]SwitchBoard[/bold]")
        return

    # Build a Rich Text object with per-character gradient color
    gradient = Text()
    char_idx = 0
    for ch in ascii_art:
        if ch in ("\n", " "):
            gradient.append(ch)
        else:
            t = char_idx / max(total - 1, 1)
            color = _interpolate(_PEACH, _PINK, t)
            gradient.append(ch, style=f"bold {color}")
            char_idx += 1

    console.print()
    console.print(gradient)
    console.print()
    console.print(f"[dim]{_SUBTITLE}[/dim]", justify="center")
    console.print(f"[dim]{_TAGLINE}[/dim]", justify="center")
    console.print()
