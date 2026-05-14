# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
from datetime import datetime

PATH2DOC = pathlib.Path(__file__).parent.resolve()
PATH2ROOT = PATH2DOC.parent.parent
PATH2SRC = PATH2ROOT.joinpath("swimnetworks-gitlab")

try:
    sys.path.insert(0, PATH2DOC.as_posix())
    sys.path.insert(0, PATH2ROOT.as_posix())
    sys.path.insert(0, PATH2SRC.as_posix())

except ImportError:
    raise ImportError(
        f"The path to the swimnetworks root folder ({PATH2ROOT=}) is incorrect. "
        f"Check in conf.py file"
    )

project = 'swimnetworks'
copyright = f'2023-{datetime.now().year}, the swimnetworks contributors"'
author = 'Bolager, Erik; Burak, Iryna; Cukarska, Ana; Datar, Chinmay; Dietrich, Felix; Rahma, Atamert; Sun, Qing'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx.ext.todo",
    "sphinx.ext.inheritance_diagram",
    # 'napoleon' supports NumPy and Google style documentation (no external Sphinx module
    #  required)
    #  -> https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    # numpydoc docstring guide
    #  -> https://numpydoc.readthedocs.io/en/latest/format.html
    "sphinx.ext.napoleon",
    # Provides automatic generation of API documentation pages for Python package
    # modules. https://sphinx-automodapi.readthedocs.io/en/latest/
    "sphinx_automodapi.automodapi",

    "sphinx.ext.viewcode",
    # Generate automatic links to the documentation of Python objects in other projects.
    # see options below
    # https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
    "sphinx.ext.intersphinx",
    # https://nbsphinx.readthedocs.io/en/0.8.5/
    # provides a source parser for Jupyter notebooks (*.ipynb files)

]

autosummary_generate = True
autodoc_mock_imports = ["flask", "mongoengine"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

# include private members (like _membername)
napoleon_include_private_with_doc = False

# include special members (like __membername__)
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False

# use the :ivar: role for instance variables
# shows the "Attributes" section
napoleon_use_ivar = True

# True -> :param: role for each function parameter.
# False -> use a single :parameters: role for all the parameters.
napoleon_use_param = True
napoleon_use_keyword = True

# True to use the :rtype: role for the return type. False to output the return type inline
# with the description.
napoleon_use_rtype = True

# ----------------------------------------------------------------------------------------
# sphinx_automodapi.automodapi (see full list of available options:
# Full config explanations here:
# https://sphinx-automodapi.readthedocs.io/en/latest/

# Do not include inherited members by default
automodsumm_inherited_members = False

# ----------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
# generate automatic links to the documentation of objects in other projects.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["README.rst", "setup.py"]

language = 'english'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
