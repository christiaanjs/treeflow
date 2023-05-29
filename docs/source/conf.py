# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from silence_tensorflow import silence_tensorflow

silence_tensorflow()

project = "TreeFlow"
copyright = "2023, Christiaan Swanepoel"
author = "Christiaan Swanepoel"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinxcontrib.napoleon",
    "myst_parser",
    "sphinx_click",
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
}
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
