"""Sphinx configuration for the trust case study documentation (Read the Docs)."""
import shutil
from pathlib import Path

project = "Can I Trust My Anomaly Detection System?"
author = "Muhammad Rashid, Elvio Amparore, Enrico Ferrari, Damiano Verda"
copyright = "2024–2026, " + author
release = "1.0.0"

extensions = [
    "myst_parser",          # the existing Markdown guides in docs/
    "sphinx_copybutton",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = [
    "_build",
    "demos/*/app/**",       # static demo pages, copied verbatim below
    "demos/*/compute/**",   # scripts that generate the demo data
]

html_theme = "furo"
html_title = "Trust case study"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/rashidrao-pk/anomaly_detection_trust_case_study/",
    "source_branch": "main",
    "source_directory": "docs/",
}
myst_heading_anchors = 3

# Interactive demos are self-contained pages (HTML, JS and data). Sphinx does
# not process them; they are copied next to the page that embeds them, so
# demos/<name>/index.html can load them from app/index.html.
DEMOS = Path(__file__).parent / "demos"


def copy_demo_apps(app, exception):
    if exception or app.builder.format != "html":
        return
    for src in DEMOS.glob("*/app"):
        dst = Path(app.outdir) / src.relative_to(DEMOS.parent)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def setup(app):
    app.connect("build-finished", copy_demo_apps)
