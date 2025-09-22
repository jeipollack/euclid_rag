##################
Development guide
##################

This page provides procedures and guidelines for developing and contributing to **euclid_rag**.

Scope of contributions
======================

**euclid_rag** is an open-source package. You may contribute to `euclid_rag <https://github.com/jeipollack/euclid_rag>`_ itself, or fork it for your own purposes.

Before contributing, please read our `contributing guidelines <https://github.com/jeipollack/euclid_rag/blob/main/CONTRIBUTING.md>`_ and `code of conduct <https://github.com/jeipollack/euclid_rag/blob/main/CODE_OF_CONDUCT.md>`_.

If you plan to submit improvements or changes, it's a good idea to propose them in a `GitHub issue`_ before investing time in a pull request.

.. _GitHub issue: https://github.com/jeipollack/euclid_rag/issues

.. _dev-environment:

Setting up a local development environment
==========================================

To develop ``euclid_rag``, create a virtual environment with your method of choice (e.g. ``venv``, ``virtualenvwrapper``), then clone and install:

.. code-block:: sh

   git clone https://github.com/jeipollack/euclid_rag.git
   cd euclid_rag
   make init

This ``init`` step does three things:

1. Installs ``euclid_rag`` in editable mode with the "dev" extra that includes test and documentation dependencies.
2. Installs ``pre-commit`` and ``tox``.
3. Installs the pre-commit hooks.

You should have Docker installed and configured to run the test suite in full, if needed.

.. _pre-commit-hooks:

Pre-commit hooks
================

The pre-commit hooks, installed via the ``make init`` command, ensure files are valid and formatted.

Some hooks automatically reformat code:

``ruff``
    Formats Python code and applies safe lint fixes.

``blacken-docs``
    Formats Python code within reStructuredText documentation and docstrings.

If a hook fails, your Git commit will be aborted. Fix the issues, stage the changes, and retry your commit.

.. _dev-run-tests:

Running tests
=============

To run the full test suite, use ``tox``, which mirrors how CI runs tests:

.. code-block:: sh

   tox run

To see available environments:

.. code-block:: sh

   tox list

To run specific tests via ``pytest``, use:

.. code-block:: sh

   tox run -e py -- tests/example_test.py

.. _dev-build-docs:

Building documentation
======================

Documentation is built with Sphinx_:

.. _Sphinx: https://www.sphinx-doc.org/en/master/

.. code-block:: sh

   tox run -e docs

Output will be in the ``docs/_build/html`` directory.

Updating pre-commit
===================

To update versions of hooks:

.. code-block:: sh

   pre-commit autoupdate

This is useful at the start of a new development cycle.

.. _dev-change-log:

Updating the change log
=======================

**euclid_rag** uses `scriv`_ to manage its change log.

To create a new entry:

.. code-block:: sh

   scriv create --edit

This generates a fragment in ``changelog.d/``. Remove unused sections and summarize your changes.

Use these section headers:

.. rst-class:: compact

- **Backward-incompatible changes**
- **New features**
- **Bug fixes**
- **Other changes**

Keep bullet points on one line each to format well on GitHub.

.. _style-guide:

Style guide
===========

Code
----

- Follow :pep:`8`. Use ``black`` and ``isort`` via pre-commit.
- Use :pep:`484` type annotations.
- Write tests using ``pytest``.

Documentation
-------------

- Follow the `Google Developer Style Guide`_ for user-facing docs.
- Use ``numpydoc``-formatted docstrings.
- Write one sentence per line in reStructuredText.

.. _Google Developer Style Guide: https://developers.google.com/style/
