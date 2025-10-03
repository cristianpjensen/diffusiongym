"""Sphinx configuration file for Flow Gym documentation."""

import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Flow Gym"
copyright = "2025, Cristian Perez Jensen"
author = "Cristian Perez Jensen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress warnings
suppress_warnings = ["ref.ref", "autosummary"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = []

sys.path.insert(0, os.path.abspath(".."))

autosummary_generate = True
autosummary_format_signature = ""
add_module_names = False

# Autodoc settings
autodoc_default_options = {
    "inherited-members": False,
}
