"""Shared SVG icon library.

These constants were duplicated verbatim across Main_Page.py and several
pages/*.py files; centralizing them here removes ~230 lines of copy-pasted
markup and gives every page the same icon set.
"""

from utils.ui import svg_icon as _svg

ICO_COMPASS = _svg(
    '<circle cx="12" cy="12" r="10" stroke="#00ff87" stroke-width="1.8"/>'
    '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="#00ff87"/>',
    16,
)
ICO_SECTOR = _svg(
    '<rect x="3" y="3" width="7" height="9" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
    '<rect x="14" y="3" width="7" height="5" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
    '<rect x="3" y="16" width="7" height="5" rx="1" stroke="#00d2ff" stroke-width="1.8"/>'
    '<rect x="14" y="12" width="7" height="9" rx="1" stroke="#00d2ff" stroke-width="1.8"/>',
    16,
)
ICO_MARKET = _svg(
    '<path d="M3 3v18h18" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
    '<path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
    16,
)
ICO_METRICS = _svg(
    '<rect x="3" y="3" width="18" height="18" rx="3" stroke="#94a3b8" stroke-width="1.5"/>'
    '<line x1="7" y1="9"  x2="17" y2="9"  stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="7" y1="13" x2="14" y2="13" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>'
    '<line x1="7" y1="17" x2="15" y2="17" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round"/>',
    16,
)
ICO_CHART = _svg(
    '<rect x="3" y="12" width="3" height="9" rx="1" fill="#00ff87"/>'
    '<rect x="9" y="7"  width="3" height="14" rx="1" fill="#00d2ff"/>'
    '<rect x="15" y="9" width="3" height="12" rx="1" fill="#ffd600"/>',
    16,
)
ICO_STATS = _svg(
    '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" stroke="#a855f7" stroke-width="1.8"/>'
    '<line x1="4" y1="22" x2="4" y2="15" stroke="#a855f7" stroke-width="1.8"/>',
    16,
)
ICO_RULER = _svg(
    '<rect x="2" y="7" width="20" height="10" rx="2" stroke="#ffd600" stroke-width="1.8"/>'
    '<line x1="6"  y1="7" x2="6"  y2="12" stroke="#ffd600" stroke-width="1.5"/>'
    '<line x1="10" y1="7" x2="10" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
    '<line x1="14" y1="7" x2="14" y2="10" stroke="#ffd600" stroke-width="1.2"/>'
    '<line x1="18" y1="7" x2="18" y2="12" stroke="#ffd600" stroke-width="1.5"/>',
    16,
)
ICO_SHIELD = _svg(
    '<path d="M12 2l8 4v6c0 5-4 8.5-8 10C8 20.5 4 17 4 12V6l8-4z" stroke="#00ff87" stroke-width="1.8" fill="none"/>'
    '<path d="M9 12l2 2 4-4" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
    16,
)
ICO_BOX = _svg(
    '<rect x="3" y="7" width="18" height="14" rx="2" stroke="#94a3b8" stroke-width="1.8"/>'
    '<path d="M8 7V5a4 4 0 018 0v2" stroke="#94a3b8" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="12" y1="12" x2="12" y2="16" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="10" y1="14" x2="14" y2="14" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>',
    16,
)
ICO_RADAR = _svg(
    '<polygon points="12 2 22 8.5 22 19.5 12 22 2 19.5 2 8.5" stroke="#a855f7" stroke-width="1.8" fill="none"/>'
    '<polygon points="12 6 18 10 18 17 12 19 6 17 6 10" stroke="#a855f7" stroke-width="1.2" fill="none" opacity="0.6"/>'
    '<line x1="12" y1="2" x2="12" y2="22" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>'
    '<line x1="2" y1="8.5" x2="22" y2="19.5" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>'
    '<line x1="2" y1="19.5" x2="22" y2="8.5" stroke="#a855f7" stroke-width="1.2" opacity="0.6"/>',
    16,
)
ICO_INFO = _svg(
    '<circle cx="12" cy="12" r="10" stroke="#00d2ff" stroke-width="1.8"/>'
    '<line x1="12" y1="16" x2="12" y2="12" stroke="#00d2ff" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="12" y1="8" x2="12" y2="8.01" stroke="#00d2ff" stroke-width="2" stroke-linecap="round"/>',
    16,
)
ICO_NEWS = _svg(
    '<rect x="3" y="4" width="18" height="16" rx="2" stroke="#00ff87" stroke-width="1.8"/>'
    '<line x1="7" y1="8" x2="17" y2="8" stroke="#00ff87" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="7" y1="12" x2="13" y2="12" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>'
    '<line x1="7" y1="16" x2="15" y2="16" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>',
    16,
)
ICO_STAR = _svg(
    '<path d="M12 2.5l2.9 6.1 6.6.9-4.8 4.7 1.2 6.6L12 17.6l-5.9 3.2 1.2-6.6-4.8-4.7 6.6-.9z" '
    'fill="#ffd600" stroke="#ffd600" stroke-width="1" stroke-linejoin="round"/>',
    13,
)
ICO_FILTER = _svg(
    '<path d="M3 4.5h18l-6.75 8v6.5l-4.5 2v-8.5z" stroke="#00d2ff" stroke-width="1.8" '
    'stroke-linejoin="round" fill="none"/>',
    13,
)
ICO_BULB = _svg(
    '<path d="M9 18.5h6M10 21h4M12 3a6 6 0 0 0-3.2 11.1c.5.35.7.9.7 1.5v.4h5v-.4c0-.6.2-1.15.7-1.5A6 6 0 0 0 12 3z" '
    'stroke="#94a3b8" stroke-width="1.6" stroke-linejoin="round" fill="none"/>',
    13,
)
ICO_ALERT = _svg(
    '<path d="M12 3.2l9.3 16.3H2.7z" stroke="#ff3d5a" stroke-width="1.7" stroke-linejoin="round" fill="none"/>'
    '<line x1="12" y1="9.5" x2="12" y2="14" stroke="#ff3d5a" stroke-width="1.9" stroke-linecap="round"/>'
    '<circle cx="12" cy="16.8" r="1" fill="#ff3d5a"/>',
    14,
)
ICO_BOLT = _svg(
    '<path d="M13 2 4.5 13.5h5.7L11 22l8.5-11.5h-5.7z" fill="#ffd600"/>',
    13,
)
ICO_CHECK_SM = _svg(
    '<path d="M4 12.5l5 5L20 6" stroke="#00ff87" stroke-width="2.3" stroke-linecap="round" '
    'stroke-linejoin="round" fill="none"/>',
    13,
)
ICO_X_SM = _svg(
    '<line x1="5" y1="5" x2="19" y2="19" stroke="#ff3d5a" stroke-width="2.3" stroke-linecap="round"/>'
    '<line x1="19" y1="5" x2="5" y2="19" stroke="#ff3d5a" stroke-width="2.3" stroke-linecap="round"/>',
    12,
)

# ─── pages/1_Portfolio.py ──────────────────────────────────────────────────────
ICO_OK = _svg(
    '<circle cx="12" cy="12" r="9" stroke="#00ff87" stroke-width="1.8"/>'
    '<path d="M8 12l3 3 5-5" stroke="#00ff87" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
)
ICO_WARN = _svg(
    '<path d="M12 3L22 21H2L12 3Z" stroke="#ffd600" stroke-width="1.8" stroke-linejoin="round"/>'
    '<line x1="12" y1="10" x2="12" y2="14" stroke="#ffd600" stroke-width="2" stroke-linecap="round"/>'
    '<circle cx="12" cy="17.5" r="1" fill="#ffd600"/>'
)
ICO_CRIT = _svg(
    '<circle cx="12" cy="12" r="9" stroke="#ff3d5a" stroke-width="1.8"/>'
    '<line x1="9" y1="9" x2="15" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="15" y1="9" x2="9" y2="15" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>'
)
ICO_TARGET = _svg(
    '<circle cx="12" cy="12" r="9" stroke="#00d2ff" stroke-width="1.8"/>'
    '<circle cx="12" cy="12" r="5" stroke="#ffd600" stroke-width="1.5"/>'
    '<circle cx="12" cy="12" r="2" fill="#00ff87"/>',
    16,
)
ICO_SIGNAL = _svg(
    '<path d="M2 12 Q6 4 12 12 Q18 20 22 12" stroke="#00d2ff" stroke-width="2" '
    'stroke-linecap="round" fill="none"/>'
    '<circle cx="12" cy="12" r="2" fill="#ffd600"/>',
    16,
)
ICO_RISK = _svg(
    '<path d="M12 3l9 18H3L12 3z" stroke="#ff3d5a" stroke-width="1.8" stroke-linejoin="round"/>'
    '<line x1="12" y1="9" x2="12" y2="14" stroke="#ff3d5a" stroke-width="1.8" stroke-linecap="round"/>'
    '<circle cx="12" cy="17" r="1" fill="#ff3d5a"/>',
    16,
)
ICO_HEATMAP = _svg(
    '<rect x="3"  y="3"  width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>'
    '<rect x="10" y="3"  width="4" height="4" rx="1" fill="#00d2ff" opacity="0.6"/>'
    '<rect x="17" y="3"  width="4" height="4" rx="1" fill="#ffd600" opacity="0.4"/>'
    '<rect x="3"  y="10" width="4" height="4" rx="1" fill="#00d2ff" opacity="0.5"/>'
    '<rect x="10" y="10" width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>'
    '<rect x="17" y="10" width="4" height="4" rx="1" fill="#a855f7" opacity="0.5"/>'
    '<rect x="3"  y="17" width="4" height="4" rx="1" fill="#ffd600" opacity="0.3"/>'
    '<rect x="17" y="17" width="4" height="4" rx="1" fill="#00ff87" opacity="0.9"/>',
    16,
)
ICO_FRONTIER = _svg(
    '<path d="M3 20 Q8 8 14 10 Q18 12 21 4" stroke="#00ff87" stroke-width="2" stroke-linecap="round" fill="none"/>'
    '<circle cx="18" cy="6" r="2.5" fill="#ff3d5a"/>'
    '<circle cx="10" cy="17" r="2" fill="#ffd600"/>',
    16,
)
ICO_LINK = _svg(
    '<circle cx="7"  cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>'
    '<circle cx="17" cy="12" r="3" stroke="#00d2ff" stroke-width="1.8"/>'
    '<line x1="10" y1="12" x2="14" y2="12" stroke="#00d2ff" stroke-width="1.8"/>',
    16,
)
ICO_IDEA = _svg(
    '<circle cx="12" cy="10" r="6" stroke="#ffd600" stroke-width="1.8"/>'
    '<path d="M9 16.5h6M10 19h4" stroke="#ffd600" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="12" y1="4" x2="12" y2="2" stroke="#ffd600" stroke-width="1.5" stroke-linecap="round"/>',
    16,
)
ICO_UP = _svg(
    '<path d="M12 20V4M5 11l7-7 7 7" stroke="#00ff87" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    14,
)
ICO_DOWN = _svg(
    '<path d="M12 4v16M5 13l7 7 7-7" stroke="#ff3d5a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    14,
)
ICO_FLAT = _svg(
    '<line x1="4" y1="12" x2="20" y2="12" stroke="#ffd600" stroke-width="2.2" stroke-linecap="round"/>'
    '<path d="M16 8l4 4-4 4" stroke="#ffd600" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    14,
)
ICO_CAPM = _svg(
    '<line x1="3" y1="21" x2="21" y2="3" stroke="#a855f7" stroke-width="1.8" stroke-linecap="round"/>'
    '<circle cx="8" cy="16" r="2.5" fill="#00ff87"/>'
    '<circle cx="14" cy="10" r="2.5" fill="#ff3d5a"/>'
    '<circle cx="18" cy="6" r="2.5" fill="#ffd600"/>'
    '<line x1="3" y1="21" x2="21" y2="3" stroke="#a855f7" stroke-width="1.8" stroke-linecap="round" stroke-dasharray="3 2"/>',
    16,
)
ICO_STRESS = _svg(
    '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" '
    'stroke="#ff3d5a" stroke-width="1.8" fill="none"/>'
    '<line x1="12" y1="9" x2="12" y2="13" stroke="#ff3d5a" stroke-width="1.8" stroke-linecap="round"/>'
    '<line x1="12" y1="17" x2="12.01" y2="17" stroke="#ff3d5a" stroke-width="2" stroke-linecap="round"/>',
    16,
)
