"""Stdlib logger used only when the plugin runs outside AstrBot.

Inside AstrBot every runtime module imports ``logger`` from ``astrbot.api`` so
the host's log level and plugin tag apply.  This fallback keeps the standalone
library and the unit tests working without AstrBot installed.
"""

from __future__ import annotations

import logging

log = logging.getLogger("zhenxin")
