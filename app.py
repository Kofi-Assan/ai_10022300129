# Name: Kofi Assan | Index: 10022300129 | CS4241-Introduction to Artificial Intelligence
"""
Streamlit UI: query, retrieved chunks, scores, final prompt, answer.
Run from project root: streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import logging
import random
import secrets
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag.pipeline import (  # noqa: E402
    PipelineLog,
    apply_feedback_boost,
    call_llm,
    run_llm_only,
)
from rag.prompts import build_context_block, build_rag_prompt, select_context  # noqa: E402
from rag.retrieval import (  # noqa: E402
    hybrid_retrieve,
    pure_vector_topk,
    retrieve_with_optional_expansion,
)
from rag.store import FaissStore  # noqa: E402

logging.basicConfig(level=logging.INFO)
INDEX_DIR = ROOT / "data" / "index"
_FONTS_DIR = ROOT / "assets" / "fonts"

# Orbitron loads in the browser (geometric HUD). Name "Prosty Extended Bold" only works if installed
# or if you add a licensed .woff2 as assets/fonts/ProstyExtended-Bold.woff2 (embedded via @font-face).
_WEB_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;800;900&display=swap');"
)


@lru_cache(maxsize=1)
def _prosty_font_face_css() -> str:
    for name in ("ProstyExtended-Bold.woff2", "Prosty-Extended-Bold.woff2"):
        path = _FONTS_DIR / name
        if not path.is_file():
            continue
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return (
            "@font-face{font-family:'Prosty Extended Bold';font-weight:700;font-style:normal;"
            "font-display:swap;src:url(data:font/woff2;base64,"
            + b64
            + ") format('woff2');}"
        )
    return ""


_UI_FONT_FAMILY = "'Prosty Extended Bold', Orbitron, system-ui, sans-serif"

# SVG: grid scroll; draggable wave peaks (JS); sea motion via translate only (no ball-synced pulse)
_LIME = "#DFFF00"
_BALL_DUR_A = 3.2  # seconds — primary ball along path
_BALL_DUR_B = 4.65  # seconds — second ball (slower lap)
_BALL_B_BEGIN = 0.85  # seconds — stagger second ball so both are visible
# Background (ghost / distant) waves — smaller, dimmer balls
_BG_BALL_DUR_1 = 2.95
_BG_BALL_DUR_2 = 3.45
_BG_BALL_DUR_3 = 4.1
_BG_BALL_B2 = 0.5
_BG_BALL_B3 = 1.05


def _hud_k_nearest_edges(positions: list[tuple[float, float]], k: int) -> set[tuple[int, int]]:
    """Undirected edges from each node to its k nearest neighbors."""
    n = len(positions)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        xi, yi = positions[i]
        dists: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            xj, yj = positions[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dists.append((d, j))
        dists.sort(key=lambda t: t[0])
        for _, j in dists[:k]:
            edges.add((min(i, j), max(i, j)))
    return edges


def _hud_random_edges(
    rng: random.Random,
    positions: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Varied topologies so layouts look different each run (caller should use SystemRandom)."""
    n = len(positions)
    edges: set[tuple[int, int]] = set()
    mode = rng.choice(("knn", "knn_plus", "random_mesh", "ring", "hub"))
    if mode == "knn":
        edges |= _hud_k_nearest_edges(positions, rng.choice((2, 3)))
    elif mode == "knn_plus":
        edges |= _hud_k_nearest_edges(positions, rng.choice((2, 3)))
        for _ in range(rng.randint(6, 20)):
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                edges.add((min(i, j), max(i, j)))
    elif mode == "random_mesh":
        m = rng.randint(max(n + 4, 16), n * 4 + 8)
        for _ in range(m):
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                edges.add((min(i, j), max(i, j)))
    elif mode == "ring":
        order = list(range(n))
        rng.shuffle(order)
        for a in range(n):
            i, j = order[a], order[(a + 1) % n]
            edges.add((min(i, j), max(i, j)))
    else:  # hub
        hub = rng.randrange(n)
        for i in range(n):
            if i != hub:
                edges.add((min(hub, i), max(hub, i)))
        for _ in range(rng.randint(2, 12)):
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def _hud_pulse_overlay_html(num_patterns: int = 5) -> str:
    """Full-viewport HUD: opacity fades while scale increases; loops through different patterns sequentially."""
    rng = random.SystemRandom()
    vb = 1000.0
    hc = vb / 2.0
    pulse_s = 2.35
    
    fade_cycle_s = round(rng.uniform(8.5, 12.0), 2)
    T = fade_cycle_s * num_patterns
    
    parts = [
        f'<div class="acity-hud-pulse" aria-hidden="true" data-hud-rev="{secrets.token_hex(8)}">',
        f'<svg width="100%" height="100%" viewBox="0 0 {vb:g} {vb:g}" '
        'preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" '
        'style="display:block;width:100%;height:100%;">'
    ]
    
    t1 = f"{0.05 / num_patterns:.4f}"
    t2 = f"{0.66 / num_patterns:.4f}"
    t3 = f"{0.70 / num_patterns:.4f}"
    
    for k in range(num_patterns):
        # Enforce at most 6 nodes
        n_nodes = rng.randint(3, 6)
        margin_lo, margin_hi = rng.uniform(28, 55), rng.uniform(905, 972)
        nodes = [(rng.uniform(margin_lo, margin_hi), rng.uniform(margin_lo, margin_hi), rng.uniform(0, 2.4)) for _ in range(n_nodes)]
        positions = [(x, y) for x, y, _ in nodes]
        edges = _hud_random_edges(rng, positions)
        
        max_scale = round(rng.uniform(1.22, 1.55), 3)
        er, eg, eb = rng.randint(110, 200), rng.randint(20, 85), rng.randint(20, 85)
        edge_stroke = f"#{er:02x}{eg:02x}{eb:02x}"
        nr = min(255, er + rng.randint(35, 100))
        ng = max(0, eg - rng.randint(0, 25))
        nb = max(0, eb - rng.randint(0, 25))
        node_fill = f"#{nr:02x}{ng:02x}{nb:02x}"
        
        begin_s = k * fade_cycle_s
        
        parts.append('<g opacity="0">')
        parts.append(
            f'<animate attributeName="opacity" dur="{T}s" begin="{begin_s}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="0;{t1};{t2};{t3};1" '
            'values="0;0.15;0.95;0;0"/>'
        )
        parts.append(f'<g transform="translate({hc},{hc})"><g>')
        parts.append(
            f'<animateTransform attributeName="transform" type="scale" additive="replace" '
            f'dur="{T}s" begin="{begin_s}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="0;{t1};{t2};{t3};1" '
            f'values="1;1;{max_scale};{max_scale};1"/>'
        )
        parts.append(f'<g transform="translate({-hc},{-hc})">')
        parts.append(f'<g stroke="{edge_stroke}" stroke-width="1.8" stroke-linecap="round" opacity="0.85" fill="none">')
        for i, j in edges:
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
        parts.append("</g>")
        
        ring_stroke = "#ff6644"
        for (cx, cy, begin_node) in nodes:
            pulse_begin = begin_s + begin_node
            parts.append(
                f'<g transform="translate({cx:.1f},{cy:.1f})">'
                f'<circle r="3" fill="{node_fill}" stroke="#1a0505" stroke-width="0.5" opacity="0.92"/>'
                f'<circle r="3" fill="none" stroke="{ring_stroke}" stroke-width="1.1" vector-effect="non-scaling-stroke" opacity="0.55">'
                f'<animate attributeName="r" values="3;42" dur="{pulse_s}s" begin="{pulse_begin:.3f}s" '
                f'repeatCount="indefinite" calcMode="spline" '
                f'keyTimes="0;1" keySplines="0.2 0.8 0.2 1"/>'
                f'<animate attributeName="opacity" values="0.85;0" dur="{pulse_s}s" begin="{pulse_begin:.3f}s" '
                f'repeatCount="indefinite" calcMode="spline" '
                f'keyTimes="0;1" keySplines="0.1 0.9 0.2 1"/>'
                "</circle></g>"
            )
        parts.append("</g></g></g></g>")

    parts.append("</svg></div>")
    return "".join(parts)


_WAVE_PEAK_DRAG_JS = """
<script>
(function () {
  var svg = document.getElementById("waveSvg");
  var grp = document.getElementById("wavePeakGroup");
  if (!svg || !grp) return;
  var KEY = "acity_wave_peak_sy";
  var MIN = 0.5;
  var MAX = 1.95;
  var SENS = 0.0045;
  var CX = 600;
  var CY = 100;

  function applyScale(sy) {
    sy = Math.max(MIN, Math.min(MAX, sy));
    grp.setAttribute(
      "transform",
      "translate(" + CX + "," + CY + ") scale(1," + sy + ") translate(" + -CX + "," + -CY + ")"
    );
    try { sessionStorage.setItem(KEY, String(sy)); } catch (e) {}
  }

  var currentSy = 1;
  try {
    var sv = sessionStorage.getItem(KEY);
    if (sv) {
      var n = parseFloat(sv);
      if (!isNaN(n)) currentSy = n;
    }
  } catch (e) {}
  applyScale(currentSy);

  var drag = false;
  var startY = 0;
  var startSy = 1;

  function down(e) {
    if (e.button != null && e.button !== 0) return;
    drag = true;
    startY = e.clientY;
    startSy = currentSy;
    try { svg.setPointerCapture(e.pointerId); } catch (x) {}
    e.preventDefault();
    svg.style.cursor = "grabbing";
  }
  function move(e) {
    if (!drag) return;
    var dy = startY - e.clientY;
    currentSy = startSy + dy * SENS;
    applyScale(currentSy);
  }
  function up(e) {
    if (!drag) return;
    drag = false;
    svg.style.cursor = "ns-resize";
    try { svg.releasePointerCapture(e.pointerId); } catch (x) {}
  }

  svg.style.cursor = "ns-resize";
  svg.addEventListener("pointerdown", down);
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", up);
})();
</script>
"""

_WAVE_BANNER_HTML = (
    f"""
<style>
{_WEB_FONT_IMPORT}
{_prosty_font_face_css()}
html,body{{margin:0;padding:0;background:#000;height:100%;font-family:{_UI_FONT_FAMILY};}}
</style>
<div id="waveBannerShell" style="font-family:{_UI_FONT_FAMILY};width:100%;
  margin:0;padding:0;box-sizing:border-box;background:#000000;">
  <svg id="waveSvg" viewBox="0 0 1200 200" xmlns="http://www.w3.org/2000/svg"
       preserveAspectRatio="xMidYMid meet" role="img"
       aria-label="Sea wave banner. Drag vertically on the waves to raise or lower peaks."
       style="width:100%;height:180px;display:block;background:#000000;touch-action:none;">
    <defs>
      <pattern id="gridScroll" width="48" height="48" patternUnits="userSpaceOnUse">
        <path d="M 48 0 L 0 0 0 48" fill="none" stroke="#1c1c1c" stroke-width="0.55"/>
      </pattern>
      <path id="ghostWaveA" d="M 0,58 C 160,18 320,118 480,58 S 800,18 960,58 S 1120,118 1200,58"/>
      <path id="ghostWaveB" d="M 0,82 C 180,128 360,32 540,82 S 900,128 1080,82 S 1140,38 1200,82"/>
      <path id="ghostWaveD" d="M 0,22 C 140,92 280,-8 420,68 S 700,-12 840,68 S 1020,22 1200,68"/>
    </defs>
    <rect width="1200" height="200" fill="#000000"/>
    <rect width="1200" height="200" fill="url(#gridScroll)" opacity="0.78">
      <animateTransform attributeName="transform" type="translate" additive="replace"
        values="0 0; 48 0" dur="18s" repeatCount="indefinite" calcMode="linear"/>
    </rect>
    <g id="wavePeakGroup" transform="translate(600,100) scale(1,1) translate(-600,-100)">
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -10; 0 0; 0 9; 0 0"
          keyTimes="0;0.25;0.5;0.75;1" dur="5.5s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1;0.42 0 0.58 1"/>
        <use href="#ghostWaveA" fill="none" stroke="#3d4a2a" stroke-width="1.25" opacity="0.45"
             stroke-linecap="round"/>
        <circle r="3.6" fill="{_LIME}" stroke="#0a0a0a" stroke-width="0.85">
          <animateMotion dur="{_BG_BALL_DUR_1}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ghostWaveA"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;0.40;0.40;0" keyTimes="0;0.06;0.90;1"
            dur="{_BG_BALL_DUR_1}s" repeatCount="indefinite"/>
        </circle>
      </g>
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 5; 0 -12; 0 4; 0 11; 0 5"
          keyTimes="0;0.25;0.5;0.75;1" dur="4.2s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.45 0 0.55 1;0.45 0 0.55 1;0.45 0 0.55 1;0.45 0 0.55 1"/>
        <use href="#ghostWaveB" fill="none" stroke="#2a3320" stroke-width="1" opacity="0.35"
             stroke-linecap="round"/>
        <circle r="3.2" fill="{_LIME}" stroke="#0a0a0a" stroke-width="0.75">
          <animateMotion dur="{_BG_BALL_DUR_2}s" begin="{_BG_BALL_B2}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ghostWaveB"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;0.34;0.34;0" keyTimes="0;0.08;0.91;1"
            dur="{_BG_BALL_DUR_2}s" begin="{_BG_BALL_B2}s" repeatCount="indefinite"/>
        </circle>
      </g>
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -7; 0 6; 0 0"
          keyTimes="0;0.33;0.66;1" dur="4.8s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.45 0 0.55 1;0.45 0 0.55 1;0.45 0 0.55 1"/>
        <use href="#ghostWaveD" fill="none" stroke="#2f3824" stroke-width="1.2" opacity="0.4"
             stroke-linecap="round"/>
        <circle r="2.8" fill="{_LIME}" stroke="#0a0a0a" stroke-width="0.65">
          <animateMotion dur="{_BG_BALL_DUR_3}s" begin="{_BG_BALL_B3}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ghostWaveD"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;0.28;0.28;0" keyTimes="0;0.07;0.90;1"
            dur="{_BG_BALL_DUR_3}s" begin="{_BG_BALL_B3}s" repeatCount="indefinite"/>
        </circle>
      </g>
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -14; 0 0; 0 12; 0 0"
          keyTimes="0;0.25;0.5;0.75;1" dur="4s" repeatCount="indefinite" calcMode="spline"
          keySplines="0.4 0 0.6 1;0.4 0 0.6 1;0.4 0 0.6 1;0.4 0 0.6 1"/>
        <path id="ragWavePath"
              d="M 4,70 C 140,8 280,132 420,70 S 700,8 840,70 S 1020,132 1196,70"
              fill="none" stroke="{_LIME}" stroke-width="2.25" stroke-linecap="round"/>
        <circle r="7" fill="{_LIME}" stroke="#000000" stroke-width="1.5">
          <animateMotion dur="{_BALL_DUR_A}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ragWavePath"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;1;1;0" keyTimes="0;0.055;0.90;1"
            dur="{_BALL_DUR_A}s" repeatCount="indefinite"/>
        </circle>
        <circle r="5.5" fill="{_LIME}" stroke="#000000" stroke-width="1.2">
          <animateMotion dur="{_BALL_DUR_B}s" begin="{_BALL_B_BEGIN}s" repeatCount="indefinite" rotate="auto">
            <mpath href="#ragWavePath"/>
          </animateMotion>
          <animate attributeName="opacity" calcMode="linear"
            values="0;1;1;0" keyTimes="0;0.16;0.91;1"
            dur="{_BALL_DUR_B}s" begin="{_BALL_B_BEGIN}s" repeatCount="indefinite"/>
        </circle>
      </g>
    </g>
  </svg>
</div>
"""
    + _WAVE_PEAK_DRAG_JS
)

# Full-page decoration: "--" lines scrolling upward forever (two identical halves → seamless loop)
_DASH_MARQUEE_ROWS = 72
_DASH_MARQUEE_STACK = "<br/>".join(["--"] * _DASH_MARQUEE_ROWS)
_FLOAT_DASH_HTML = (
    f"""
<style>
  @keyframes acity_dash_up {{
    0% {{ transform: translateY(0); }}
    100% {{ transform: translateY(-50%); }}
  }}
  .acity-dash-bg {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
  }}
  .acity-dash-bg__rail {{
    position: absolute;
    top: 0;
    display: flex;
    flex-direction: column;
    width: 2.75rem;
    opacity: 0.2;
    font-size: 13px;
    line-height: 2rem;
    font-weight: 700;
    font-family: {_UI_FONT_FAMILY};
    color: rgba(255, 255, 255, 0.72);
    text-align: center;
    animation: acity_dash_up 24s linear infinite;
    will-change: transform;
  }}
  .acity-dash-bg__rail--left {{
    left: clamp(8px, 2.8vw, 56px);
    right: auto;
  }}
  .acity-dash-bg__rail--right {{
    right: clamp(8px, 2.8vw, 56px);
    left: auto;
  }}
  .acity-dash-bg__half {{ flex: 0 0 auto; }}
</style>
<div class="acity-dash-bg" aria-hidden="true">
  <div class="acity-dash-bg__rail acity-dash-bg__rail--left">
    <div class="acity-dash-bg__half">"""
    + _DASH_MARQUEE_STACK
    + """</div>
    <div class="acity-dash-bg__half">"""
    + _DASH_MARQUEE_STACK
    + """</div>
  </div>
  <div class="acity-dash-bg__rail acity-dash-bg__rail--right">
    <div class="acity-dash-bg__half">"""
    + _DASH_MARQUEE_STACK
    + """</div>
    <div class="acity-dash-bg__half">"""
    + _DASH_MARQUEE_STACK
    + """</div>
  </div>
</div>
"""
)

st.set_page_config(page_title="ACity RAG (CS4241)", layout="wide")

st.markdown(
    """
    <style>
      """
    + _WEB_FONT_IMPORT
    + _prosty_font_face_css()
    + """
      .stApp {
        background-color: #000000 !important;
        font-family: """
    + _UI_FONT_FAMILY
    + """ !important;
        letter-spacing: 0.06em;
      }
      div.acity-hud-pulse {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 0 !important;
        pointer-events: none !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
      }
      .stApp h1 {
        font-family: """
    + _UI_FONT_FAMILY
    + """ !important;
        font-weight: 800 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
      }
      .stApp h2, .stApp h3 {
        font-family: """
    + _UI_FONT_FAMILY
    + """ !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
      }
      .stApp [data-testid="stHeader"] {
        font-family: """
    + _UI_FONT_FAMILY
    + """ !important;
      }
      .stApp pre, .stApp code,
      .stApp [data-testid="stCodeBlock"],
      .stApp .stMarkdown pre {
        font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important;
        letter-spacing: normal !important;
        text-transform: none !important;
      }
      .stApp [data-testid="stHeader"],
      .stApp section[data-testid="stSidebar"],
      .stApp [data-testid="stAppViewContainer"],
      .stApp .main {
        position: relative;
        z-index: 1;
      }
      section[data-testid="stSidebar"] { background-color: #0a0a0a !important; }
      div[data-testid="stToolbar"] { background-color: transparent !important; }
      /* High-visibility primary — Run RAG */
      .stApp button[kind="primary"],
      .stApp button[data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #f4ff66 0%, #DFFF00 42%, #b8cf00 100%) !important;
        color: #0a0a0a !important;
        border: 2px solid #f8ff99 !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.7rem 2rem !important;
        min-height: 3.1rem !important;
        box-shadow:
          0 0 32px rgba(223, 255, 0, 0.65),
          0 0 60px rgba(223, 255, 0, 0.25),
          0 8px 24px rgba(0, 0, 0, 0.55) !important;
        text-shadow: none !important;
      }
      .stApp button[kind="primary"]:hover,
      .stApp button[data-testid="baseButton-primary"]:hover {
        filter: brightness(1.12) saturate(1.05) !important;
        box-shadow:
          0 0 40px rgba(223, 255, 0, 0.85),
          0 0 80px rgba(223, 255, 0, 0.35),
          0 10px 28px rgba(0, 0, 0, 0.55) !important;
      }
      .stApp button[kind="primary"]:focus-visible,
      .stApp button[data-testid="baseButton-primary"]:focus-visible {
        outline: 3px solid #f8ff99 !important;
        outline-offset: 3px !important;
      }
      /* Make sidebar scrollable */
      .stSidebar > div:first-child {
        overflow-y: auto !important;
        max-height: 100vh !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(_hud_pulse_overlay_html(num_patterns=5), unsafe_allow_html=True)
st.markdown(_FLOAT_DASH_HTML, unsafe_allow_html=True)

# File-based chat history persistence
CHAT_HISTORY_FILE = ROOT / "chat_history.json"

def load_chat_history_from_file():
    """Load chat history from JSON file"""
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading chat history: {e}")
    return []

def save_chat_history_to_file(history):
    """Save chat history to JSON file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error saving chat history: {e}")

# Initialize chat history from file
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history_from_file()

# JavaScript for localStorage operations (as backup)
CHAT_STORAGE_JS = """
<script>
(function() {
    // Save chat history to localStorage
    window.saveChatHistory = function(history) {
        try {
            localStorage.setItem('acity_chat_history', JSON.stringify(history));
            console.log('Saved chat history to localStorage:', history.length, 'entries');
        } catch (e) {
            console.error('Failed to save chat history:', e);
        }
    };
    
    // Load chat history from localStorage
    window.loadChatHistory = function() {
        try {
            const stored = localStorage.getItem('acity_chat_history');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('Failed to load chat history:', e);
            return [];
        }
    };
    
    // Clear chat history from localStorage
    window.clearChatHistory = function() {
        try {
            localStorage.removeItem('acity_chat_history');
        } catch (e) {
            console.error('Failed to clear chat history:', e);
        }
    };
})();
</script>
"""

st.html(CHAT_STORAGE_JS)
st.title("Academic City RAG Assistant")
st.caption("Kofi Assan · 10022300129 · CS4241 — manual RAG (no LangChain/LlamaIndex)")
components.html(_WAVE_BANNER_HTML, height=180, scrolling=False)

if not (INDEX_DIR / "index.faiss").is_file():
    st.error(
        "Index not found. In a terminal at the project root, run:\n\n"
        "`python scripts/download_data.py` then `python scripts/build_index.py`"
    )
    st.stop()

@st.cache_resource
def load_store():
    return FaissStore.load(INDEX_DIR)


store = load_store()

with st.sidebar:
    st.subheader("Chat History")
    
    # Chat history display
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"{chat['timestamp'][:19].replace('T', ' ')} - {chat['query'][:50]}{'...' if len(chat['query']) > 50 else ''}", expanded=False):
                st.markdown(f"**Query:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                if chat.get('retrieval_info'):
                    st.markdown(f"**Sources:** {', '.join(chat['retrieval_info'].get('sources', []))}")
    else:
        st.caption("No chat history yet")
    
    # Clear history button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        save_chat_history_to_file([])
        clear_js = """
        <script>
        window.clearChatHistory();
        </script>
        """
        st.html(clear_js)
        st.rerun()
    
    st.divider()
    
    st.subheader("Retrieval")
    use_hybrid = st.toggle("Hybrid (vector + BM25)", value=True)
    use_expansion = st.toggle("Query expansion (Part B)", value=False, disabled=not use_hybrid)
    top_k = st.slider("Top-k", 3, 20, 8)
    prompt_style = st.selectbox("Prompt style", ["strict", "concise"])
    st.subheader("Part G — feedback boost")
    st.caption("Prefer answers grounded in:")
    boost_election = st.checkbox("Election CSV", value=False)
    boost_budget = st.checkbox("Budget PDF", value=True)
    compare_llm_only = st.checkbox("Also run LLM-only (Part E)", value=False)

query = st.text_input("Your question", placeholder="Ask about elections or the 2025 budget…")

if st.button("Run RAG", type="primary") and query.strip():
    boost_sources = set()
    if boost_election:
        boost_sources.add("ghana_elections")
    if boost_budget:
        boost_sources.add("budget_2025")

    plog = PipelineLog()
    if use_hybrid:
        if use_expansion:
            hits, expanded = retrieve_with_optional_expansion(
                store, query, k=top_k, use_expansion=True
            )
        else:
            hits = hybrid_retrieve(store, query, k=top_k)
            expanded = query
    else:
        hits = pure_vector_topk(store, query, k=top_k)
        expanded = query
    if boost_sources:
        hits = apply_feedback_boost(hits, boost_sources)

    plog.add("query", {"text": query})
    plog.add(
        "retrieval",
        {
            "mode": "hybrid" if use_hybrid else "vector_only",
            "query_expansion": bool(use_expansion) if use_hybrid else False,
            "expanded_query": expanded,
            "feedback_boost": sorted(boost_sources),
            "hits": [
                {
                    "source": h.chunk.source,
                    "vector_score": round(h.vector_score, 4),
                    "bm25_score": round(h.bm25_score, 4),
                    "fused_score": round(h.fused_score, 4),
                    "text_preview": h.chunk.text[:300]
                    + ("…" if len(h.chunk.text) > 300 else ""),
                }
                for h in hits
            ],
        },
    )

    selected = select_context(hits, max_chars=6000)
    context_block = build_context_block(selected)
    final_prompt = build_rag_prompt(expanded, context_block, style=prompt_style)
    plog.add(
        "context_selection",
        {
            "num_chunks": len(selected),
            "sources": list({h.chunk.source for h in selected}),
        },
    )
    plog.add(
        "prompt",
        {"style": prompt_style, "final_prompt": final_prompt, "prompt_chars": len(final_prompt)},
    )

    with st.spinner("Calling LLM…"):
        answer = call_llm(final_prompt)
    plog.add("generation", {"answer_preview": answer[:500]})

    st.subheader("Answer")
    st.write(answer)
    
    # Save to chat history
    chat_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer,
        'retrieval_info': {
            'mode': "hybrid" if use_hybrid else "vector_only",
            'sources': list({h.chunk.source for h in hits}),
            'num_chunks': len(hits)
        }
    }
    st.session_state.chat_history.append(chat_entry)
    
    # Save to file (primary method)
    save_chat_history_to_file(st.session_state.chat_history)
    
    # Also save to localStorage (backup method)
    save_history_js = f"""
    <script>
    (function() {{
        const history = window.loadChatHistory();
        history.push({json.dumps(chat_entry)});
        window.saveChatHistory(history);
        console.log('Chat saved to file and localStorage');
    }})();
    </script>
    """
    st.html(save_history_js)

    with st.expander("Retrieved chunks & scores"):
        for h in hits:
            st.markdown(
                f"**{h.chunk.source}** · vec={h.vector_score:.4f} · bm25={h.bm25_score:.4f} · fused={h.fused_score:.4f}"
            )
            st.text(h.chunk.text[:1200] + ("…" if len(h.chunk.text) > 1200 else ""))

    with st.expander("Final prompt sent to LLM"):
        st.code(final_prompt, language="text")

    with st.expander("Pipeline log (JSON-like)"):
        st.json(plog.stages)

    if compare_llm_only:
        with st.spinner("LLM-only baseline…"):
            base_ans, base_log = run_llm_only(query)
        st.subheader("Baseline: LLM without retrieval")
        st.write(base_ans)
        with st.expander("LLM-only prompt"):
            st.code(
                next(
                    (s["final_prompt"] for s in base_log.stages if s["stage"] == "prompt"),
                    "",
                ),
                language="text",
            )

elif query.strip():
    st.info("Click **Run RAG** to query the index.")

