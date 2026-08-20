"""Dark / light theming for Image Processing Studio.

The whole theme is driven by one injected stylesheet.  Streamlit's own
palette is decided in the browser (it is remembered in localStorage), so
it cannot be switched reliably from Python - therefore this stylesheet
is authoritative and repaints every surface the app actually uses, in
both themes.

`apply_theme()` must be called once per script run, before any content
is written.
"""

import streamlit as st

DARK = {
    "name": "dark",
    # Matches Streamlit's own dark base exactly, so the page background is
    # identical whichever stylesheet the browser ends up applying.
    "bg": "#0e1117",
    "bg_alt": "#12161c",
    "surface": "#171c24",
    "surface_hover": "#1d232c",
    "border": "#262d38",
    "border_strong": "#333c48",
    "text": "#e6eaf0",
    "text_muted": "#98a2b1",
    "accent": "#5b9bd5",
    "accent_soft": "rgba(91, 155, 213, 0.14)",
    "warn": "#e0a458",
    "grid": "#242b34",
    "shadow": "rgba(0, 0, 0, 0.35)",
}

LIGHT = {
    "name": "light",
    "bg": "#f6f7f9",
    "bg_alt": "#eef1f5",
    "surface": "#ffffff",
    "surface_hover": "#f4f6f9",
    "border": "#dfe4ea",
    "border_strong": "#c7ced7",
    "text": "#1b2028",
    "text_muted": "#5d6875",
    "accent": "#2f6fb0",
    "accent_soft": "rgba(47, 111, 176, 0.10)",
    "warn": "#b57320",
    "grid": "#e4e8ee",
    "shadow": "rgba(15, 25, 40, 0.10)",
}

PALETTES = {"dark": DARK, "light": LIGHT}


def current_palette():
    """Palette dict for whichever theme is active in session state."""
    return PALETTES.get(st.session_state.get("theme", "dark"), DARK)


def toggle_theme():
    """Flip between the two themes."""
    st.session_state["theme"] = (
        "light" if st.session_state.get("theme", "dark") == "dark" else "dark"
    )


def apply_theme():
    """Inject the stylesheet for the active theme."""
    palette = current_palette()
    st.markdown(_build_css(palette), unsafe_allow_html=True)
    return palette


def matplotlib_style():
    """Colours for Matplotlib figures so charts match the app chrome."""
    palette = current_palette()
    return {
        "face": palette["surface"],
        "text": palette["text"],
        "muted": palette["text_muted"],
        "grid": palette["grid"],
        "accent": palette["accent"],
        "border": palette["border"],
    }


def _build_css(p):
    return f"""
<style>
:root {{
    --iva-bg: {p['bg']};
    --iva-bg-alt: {p['bg_alt']};
    --iva-surface: {p['surface']};
    --iva-surface-hover: {p['surface_hover']};
    --iva-border: {p['border']};
    --iva-border-strong: {p['border_strong']};
    --iva-text: {p['text']};
    --iva-muted: {p['text_muted']};
    --iva-accent: {p['accent']};
    --iva-accent-soft: {p['accent_soft']};
    --iva-warn: {p['warn']};
    --iva-shadow: {p['shadow']};
    color-scheme: {p['name']};
}}

/* ---------------------------------------------------------------- Shell */
html, body {{
    background-color: var(--iva-bg) !important;
}}
html body .stApp.stApp.stApp,
html body [data-testid="stAppViewContainer"],
html body [data-testid="stMain"] {{
    background-color: var(--iva-bg) !important;
    color: var(--iva-text) !important;
    transition: background-color .25s ease, color .25s ease;
}}
html body [data-testid="stHeader"],
html body [data-testid="stToolbar"] {{ background: transparent !important; }}

html body [data-testid="stSidebar"],
html body [data-testid="stSidebarContent"] {{
    background-color: var(--iva-bg-alt) !important;
    border-right: 1px solid var(--iva-border);
}}

html, body, .stApp, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Typography - Streamlit sets these inline-ish, so they need force. */
html body .stApp h1, html body .stApp h2, html body .stApp h3,
html body .stApp h4, html body .stApp h5, html body .stApp h6,
html body .stApp p, html body .stApp li, html body .stApp strong,
html body .stApp .stMarkdown, html body .stApp .stMarkdown *,
html body [data-testid="stSidebar"] * {{
    color: var(--iva-text);
}}
html body .stApp h1, html body .stApp h2, html body .stApp h3 {{
    letter-spacing: -0.01em;
}}
html body .stApp a {{ color: var(--iva-accent); }}
html body .stApp hr {{ border-color: var(--iva-border); }}
html body [data-testid="stCaptionContainer"],
html body [data-testid="stCaptionContainer"] * {{
    color: var(--iva-muted) !important;
}}
html body .stApp code, html body .stApp kbd {{
    background: var(--iva-bg-alt) !important;
    color: var(--iva-accent) !important;
    border: 1px solid var(--iva-border);
    border-radius: 4px;
}}

/* --------------------------------------------------------- App header */
.iva-header {{
    border: 1px solid var(--iva-border);
    background: var(--iva-surface);
    border-radius: 10px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.5rem;
    border-left: 3px solid var(--iva-accent);
}}
/* The generic .stMarkdown typography rules above are deliberately broad, so
   every custom component states its own colour with !important. */
.iva-header h1 {{
    font-size: 1.72rem; font-weight: 650; margin: 0 0 .3rem 0; line-height: 1.2;
    color: var(--iva-text) !important;
}}
.iva-header .iva-sub {{
    font-size: .98rem; color: var(--iva-accent) !important;
    font-weight: 500; margin-bottom: .45rem;
}}
.iva-header .iva-desc {{
    font-size: .86rem; color: var(--iva-muted) !important; margin: 0; line-height: 1.55;
}}

/* ------------------------------------------------------ Section heading */
.iva-section {{ margin: .2rem 0 1.1rem 0; }}
.iva-section h2 {{
    font-size: 1.24rem; font-weight: 620; margin: 0 0 .3rem 0;
    display: flex; align-items: center; gap: .55rem; color: var(--iva-text) !important;
}}
.iva-section .iva-tag {{
    font-size: .68rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
    color: var(--iva-accent) !important; background: var(--iva-accent-soft);
    border: 1px solid var(--iva-border); border-radius: 4px; padding: .16rem .48rem;
}}
.iva-section p {{
    font-size: .87rem; color: var(--iva-muted) !important; margin: 0; line-height: 1.6;
}}

/* ----------------------------------------------------------------- Cards */
.iva-card {{
    background: var(--iva-surface);
    border: 1px solid var(--iva-border);
    border-radius: 9px;
    padding: .95rem 1.05rem;
    transition: border-color .18s ease, background-color .18s ease;
}}
.iva-card:hover {{ border-color: var(--iva-border-strong); background: var(--iva-surface-hover); }}
.iva-card .iva-label {{
    font-size: .68rem; text-transform: uppercase; letter-spacing: .08em;
    color: var(--iva-muted) !important; margin-bottom: .3rem;
}}
.iva-card .iva-value {{
    font-size: 1.18rem; font-weight: 620; color: var(--iva-text) !important;
}}
.iva-card .iva-value p {{ color: inherit !important; }}
.iva-card .iva-unit {{
    font-size: .74rem; color: var(--iva-muted) !important; margin-left: .22rem;
}}

.iva-note {{
    background: var(--iva-accent-soft);
    border: 1px solid var(--iva-border);
    border-left: 3px solid var(--iva-accent);
    border-radius: 6px; padding: .6rem .85rem;
    font-size: .82rem; color: var(--iva-muted) !important; line-height: 1.55;
    margin-bottom: .9rem;
}}
.iva-warn {{
    border: 1px solid var(--iva-warn);
    border-left: 3px solid var(--iva-warn);
    border-radius: 6px; padding: .6rem .85rem;
    font-size: .82rem; color: var(--iva-warn) !important;
    line-height: 1.5; margin-bottom: .9rem;
}}
.iva-note b, .iva-warn b, .iva-note *, .iva-warn * {{ color: inherit !important; }}

/* Kernel / formula blocks */
.iva-kernel {{
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: .82rem; line-height: 1.75; white-space: pre;
    background: var(--iva-bg-alt); border: 1px solid var(--iva-border);
    border-radius: 6px; padding: .65rem .85rem; color: var(--iva-text) !important;
    overflow-x: auto; margin: .3rem 0 .7rem 0;
}}
.iva-kernel-name {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
    color: var(--iva-muted) !important; margin-bottom: .1rem;
}}

/* Image caption strip */
.iva-imgcap {{
    font-size: .74rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
    color: var(--iva-muted) !important; margin-bottom: .35rem;
    padding-bottom: .3rem; border-bottom: 1px solid var(--iva-border);
}}

/* ---------------------------------------------------------- Data tables */
.iva-table-wrap {{ overflow-x: auto; margin: .2rem 0 1rem 0; }}
table.iva-table {{
    width: 100%; border-collapse: collapse; font-size: .82rem;
    background: var(--iva-surface); border: 1px solid var(--iva-border);
    border-radius: 8px; overflow: hidden;
}}
table.iva-table th {{
    text-align: left; font-weight: 600; font-size: .7rem;
    text-transform: uppercase; letter-spacing: .06em;
    color: var(--iva-muted) !important; background: var(--iva-bg-alt);
    padding: .5rem .7rem; border-bottom: 1px solid var(--iva-border);
    white-space: nowrap;
}}
table.iva-table td {{
    padding: .45rem .7rem; color: var(--iva-text) !important;
    border-bottom: 1px solid var(--iva-border); white-space: nowrap;
}}
table.iva-table tbody tr:last-child td {{ border-bottom: none; }}
table.iva-table tbody tr:hover td {{ background: var(--iva-surface-hover); }}

/* -------------------------------------------------------------- Widgets */
html body .stButton > button,
html body .stDownloadButton > button {{
    background-color: var(--iva-surface) !important;
    color: var(--iva-text) !important;
    border: 1px solid var(--iva-border-strong) !important;
    border-radius: 6px;
    font-size: .84rem; font-weight: 500;
    padding: .38rem .9rem;
    transition: background-color .16s ease, border-color .16s ease, color .16s ease;
}}
html body .stButton > button:hover,
html body .stDownloadButton > button:hover {{
    background-color: var(--iva-surface-hover) !important;
    border-color: var(--iva-accent) !important;
    color: var(--iva-accent) !important;
}}
html body .stButton > button:focus,
html body .stDownloadButton > button:focus {{
    box-shadow: none !important; border-color: var(--iva-accent) !important;
}}
html body .stButton > button p, html body .stDownloadButton > button p {{
    color: inherit !important;
}}

html body [data-testid="stExpander"] {{
    border: 1px solid var(--iva-border) !important;
    border-radius: 8px;
    background-color: var(--iva-surface) !important;
    margin-bottom: .7rem;
}}
html body [data-testid="stExpander"] details,
html body [data-testid="stExpander"] summary {{
    background-color: transparent !important;
    color: var(--iva-text) !important;
}}
html body [data-testid="stExpander"] summary {{ font-size: .84rem; font-weight: 550; }}
html body [data-testid="stExpander"] summary:hover,
html body [data-testid="stExpander"] summary:hover * {{ color: var(--iva-accent) !important; }}

html body [data-testid="stFileUploaderDropzone"] {{
    background-color: var(--iva-surface) !important;
    border: 1px dashed var(--iva-border-strong) !important;
    border-radius: 8px;
}}
html body [data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--iva-accent) !important; }}
html body [data-testid="stFileUploaderDropzone"] * {{ color: var(--iva-text) !important; }}
html body [data-testid="stFileUploaderDropzoneInstructions"] small {{ color: var(--iva-muted) !important; }}
html body [data-testid="stFileUploaderFile"] {{ color: var(--iva-text) !important; }}

/* Tabs */
html body .stTabs [data-baseweb="tab-list"] {{
    gap: .25rem; border-bottom: 1px solid var(--iva-border);
    background-color: transparent !important;
}}
html body .stTabs [data-baseweb="tab"] {{
    font-size: .85rem; font-weight: 520; padding: .45rem .85rem;
    border-radius: 6px 6px 0 0; background-color: transparent !important;
    color: var(--iva-muted) !important;
}}
html body .stTabs [data-baseweb="tab"] * {{ color: inherit !important; }}
html body .stTabs [data-baseweb="tab"]:hover {{ color: var(--iva-text) !important; }}
html body .stTabs [aria-selected="true"] {{ color: var(--iva-accent) !important; }}
html body .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--iva-accent) !important; }}
html body .stTabs [data-baseweb="tab-border"] {{ background-color: var(--iva-border) !important; }}

/* Inputs, selects and their popovers */
html body [data-baseweb="select"] > div,
html body [data-baseweb="input"],
html body [data-baseweb="base-input"],
html body .stTextInput input,
html body .stNumberInput input {{
    background-color: var(--iva-surface) !important;
    border-color: var(--iva-border-strong) !important;
    color: var(--iva-text) !important;
}}
html body [data-baseweb="select"] * {{ color: var(--iva-text) !important; }}
html body [data-baseweb="popover"] [role="listbox"],
html body [data-baseweb="menu"], html body [data-baseweb="popover"] ul {{
    background-color: var(--iva-surface) !important;
    border: 1px solid var(--iva-border) !important;
    box-shadow: 0 6px 18px var(--iva-shadow) !important;
}}
html body [role="option"], html body [data-baseweb="menu"] li {{
    background-color: transparent !important;
    color: var(--iva-text) !important;
}}
html body [role="option"]:hover, html body [data-baseweb="menu"] li:hover {{
    background-color: var(--iva-surface-hover) !important;
    color: var(--iva-accent) !important;
}}
html body [data-baseweb="tooltip"] div {{
    background-color: var(--iva-surface) !important;
    color: var(--iva-text) !important;
    border: 1px solid var(--iva-border) !important;
}}

/* Sliders */
html body [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background-color: var(--iva-accent) !important;
}}
html body [data-testid="stSliderThumbValue"],
html body [data-testid="stSliderTickBarMin"],
html body [data-testid="stSliderTickBarMax"] {{
    color: var(--iva-muted) !important;
}}
html body [data-testid="stSliderTickBar"] {{ background: transparent !important; }}

/* Radio / checkbox */
html body [data-testid="stRadio"] label,
html body [data-testid="stCheckbox"] label {{ color: var(--iva-text) !important; }}
html body [data-testid="stRadio"] label p,
html body [data-testid="stCheckbox"] label p {{
    color: var(--iva-text) !important; font-size: .84rem;
}}

/* Widget labels */
html body [data-testid="stWidgetLabel"] p,
html body [data-testid="stWidgetLabel"] label {{
    font-size: .8rem !important; font-weight: 500;
    color: var(--iva-muted) !important;
}}

/* Alerts */
html body [data-testid="stAlert"], html body .stAlert {{
    background-color: var(--iva-surface) !important;
    border: 1px solid var(--iva-border) !important;
    border-radius: 6px;
}}
html body [data-testid="stAlert"] * {{ color: var(--iva-text) !important; }}

/* Images */
html body [data-testid="stImage"] img {{
    border-radius: 6px;
    border: 1px solid var(--iva-border);
}}
html body [data-testid="stImageCaption"] {{ color: var(--iva-muted) !important; font-size: .76rem; }}

/* Chrome we do not need */
html body [data-testid="stSidebarNav"] {{ display: none; }}
html body footer, html body #MainMenu {{ visibility: hidden; }}
</style>
"""
