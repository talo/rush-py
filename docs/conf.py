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
html_static_path = ["_static"]

# Shibuya theme options
html_theme_options = {
    "accent_color": "iris",
    "toctree_includehidden": False,
}

# Use a slimmer sidebar for the internal EXESS docs section.
html_sidebars = {
    "exess/**": ["sidebars/localtoc.html"],
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


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def add_exess_nav(app, pagename, templatename, context, doctree):
    if not pagename.startswith("exess/"):
        return

    from copy import deepcopy

    from sphinx import addnodes
    from sphinx.environment.adapters.toctree import TocTree

    env = app.builder.env
    exess_index = "exess/index"
    if exess_index not in env.all_docs:
        return

    exess_doctree = env.get_doctree(exess_index)
    toctrees = list(exess_doctree.findall(addnodes.toctree))
    if not toctrees:
        return

    toc_tree = TocTree(env)
    collapse = _to_bool(context.get("theme_toctree_collapse"), default=False)
    titles_only = _to_bool(context.get("theme_toctree_titles_only"), default=False)
    includehidden = _to_bool(context.get("theme_toctree_includehidden"), default=False)
    maxdepth = context.get("theme_toctree_maxdepth")
    try:
        maxdepth = int(maxdepth) if maxdepth not in (None, "") else 0
    except (TypeError, ValueError):
        maxdepth = 0

    root = None
    for toctree in toctrees:
        resolved = toc_tree.resolve(
            pagename,
            app.builder,
            deepcopy(toctree),
            prune=True,
            maxdepth=maxdepth,
            titles_only=titles_only,
            collapse=collapse,
            includehidden=includehidden,
        )
        if resolved is None:
            continue
        if root is None:
            root = resolved
        else:
            root.extend(resolved.children)

    if root is None:
        return

    context["exess_globaltoc"] = app.builder.render_partial(root)["fragment"]


def setup(app):
    """Connect the autodoc-skip-member event."""
    app.connect("autodoc-skip-member", skip_enum_members)
    app.connect("html-page-context", add_exess_nav)
