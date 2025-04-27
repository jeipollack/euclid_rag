import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# At top of conf.py, after imports:
rst_epilog = open(
    os.path.join(os.path.dirname(__file__), "_rst_epilog.rst"),
    encoding="utf-8",
).read()

# Make sure Sphinx can import your package under python/euclid/rag
sys.path.insert(0, os.path.abspath("../python"))

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


html_static_path = ["_static"]  # create this folder (even if empty)
html_css_files = []  # add any custom.css here
# html_js_files = ["custom.js"]  # add any custom.js here

# ——————————————————————————————
# HTML output settings
# ——————————————————————————————
html_theme = "pydata_sphinx_theme"
html_title = project
html_short_title = project
html_logo = None
html_favicon = None

html_theme_options = {
    "external_links": [],
    "icon_links": [],
    "pygment_light_style": "friendly",
    "pygment_dark_style": "monokai",
    "logo": {
        "text": project,
        "alt_text": project,
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
        "image_class": "logo",
        "image_style": "default",
    },
}


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
