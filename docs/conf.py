import os
import sys

# Add your project root to path for autodoc
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

# Project information
project = "rush"
copyright = "2026, QDX"
author = "Sean L"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "exess_params",
]

# MyST extensions needed for ::: directives and definition lists in Markdown.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
]

# Support for Markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "shibuya"
html_static_path = ["_sphinx_static"]
html_logo = "_sphinx_static/logo.svg"
html_css_files = ["custom.css"]

# Shibuya theme options
html_theme_options = {
    "accent_color": "iris",
    "toctree_includehidden": False,
    "color_mode": "dark",
    "logo_target": "https://qdx.co",
    "nav_links": [
        {"title": "Use EXESS", "url": "https://exess.qdx.co"},
        {"title": "Use Rush", "url": "https://qdx.co/rush"},
    ],
}

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Remove module prefixes from type names
python_use_unqualified_type_names = True

# Move type hints to parameter descriptions for cleaner signatures
autodoc_typehints = "description"

autodoc_preserve_defaults = True

# Use short form for type names (list instead of typing.List)
autodoc_typehints_format = "short"

# Control member ordering - group by type (functions, classes, then data)
autodoc_member_order = "groupwise"


# Skip individual enum members to prevent documentation clutter
def skip_enum_members(app, what, name, obj, skip, options):
    """
    Skip individual enum members in documentation.

    This prevents long enums from cluttering the docs and sidebar.
    Enum classes still appear with their docstrings.
    """
    import enum

    # Check if the object itself is an enum member
    if isinstance(obj, enum.Enum):
        return True

    return skip


def setup(app):
    """Connect the autodoc-skip-member event."""
    app.connect("autodoc-skip-member", skip_enum_members)
