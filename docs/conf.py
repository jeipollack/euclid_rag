from pathlib import Path
import sys

repo_path = Path(__file__).parents[1]
docs_path = repo_path / "docs"
src_path = repo_path / "python"
static_path = src_path / "euclid" / "static"

# At top of conf.py, after imports:
with open(docs_path / "_rst_epilog.rst", encoding="utf-8") as f:
    rst_epilog = f.read()

# Make sure Sphinx can import your package under python/euclid/rag
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "euclid" / "rag"))
# ——————————————————————————————
# Project information
# ——————————————————————————————
project = "euclid_rag"
author = "Euclid Collaboration"
release = "0.1.0"  # bump to your current version

# ——————————————————————————————
# Sphinx extensions
# ——————————————————————————————
extensions = [
    "sphinx.ext.autodoc",  # for Python docstring extraction
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # to link out to Python, NumPy, etc.
    "myst_parser",  # if you want Markdown support,
    "sphinx_automodapi.automodapi",  # for module-level docstrings
]

# ——————————————————————————————
# Templates and static files
# ——————————————————————————————
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_rst_epilog.rst"]


html_static_path = [str(static_path)]  # create this folder (even if empty)
html_css_files = []  # add any custom.css here

# ——————————————————————————————
# HTML output settings
# ——————————————————————————————
html_theme = "pydata_sphinx_theme"
html_title = project
html_logo = str(static_path / "logo.png")
html_favicon = None


html_theme_options = {
    "external_links": [],
    "icon_links": [],
    "pygment_light_style": "friendly",
    "pygment_dark_style": "monokai",
    "logo": {
        "text": project,
        "alt_text": project,
        "image_light": str(static_path / "logo.png"),
        "image_dark": str(static_path / "logo_dark.png"),
        "image_class": "logo",
        "image_style": "default",
    },
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
    },
    "show_toc_level": 3,
    "navigation_with_keys": True,
}

html_sidebars = {"**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"]}


# Disable the “Edit on GitHub” button entirely
html_context = {"show_github_edit_link": False, "default_mode": "auto"}

# ——————————————————————————————
# Intersphinx mappings
# ——————————————————————————————
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
