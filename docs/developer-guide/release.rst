#################
Release procedure
#################

This page provides an overview of how `euclid_rag <https://github.com/jeipollack/euclid_rag>`_ releases are made.
This information is primarily useful for maintainers.

Releases are largely automated through GitHub Actions (see the `ci.yaml`_ workflow file for details).
When a semantic version tag is pushed to GitHub, ``euclid_rag`` is automatically `released to PyPI`_ (coming soon) with that version.
Documentation is built and published for each version (see `Euclid RAG Documentation`_.).

.. _`Euclid RAG Documentation`: https://example.com/to-be-added
.. _`released to PyPI`: https://pypi.org/
.. _`ci.yaml`: https://github.com/jeipollack/euclid_rag/blob/main/.github/workflows/ci.yaml

.. _regular-release:

Regular releases
================

Releases are made from the ``main`` branch after changes are merged.
You can release a new major version (``X.0.0``), a new minor version (``X.Y.0``), or a new patch version (``X.Y.Z``).
To patch an earlier version, see :ref:`backport-release`.

Release tags use semantic versioning according to the :pep:`440` spec.

1. Change log and documentation
-------------------------------

Changelog messages are collected using scriv_.
See :ref:`dev-change-log` in the *Developer Guide* for more.

Before release, gather the changelog fragments in :file:`changelog.d` by running:

.. code-block:: bash

   scriv collect --version X.Y.Z

This creates a full changelog entry in :file:`CHANGELOG.md` and removes the fragments.

Review, edit, and commit the updated changelog.
Create a PR, get it merged, and then proceed to tagging the release.

2. GitHub release and tag
-------------------------

Use [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) to publish the new version.

- Tag from the correct branch (usually ``main``).
- Use semantic versioning like ``X.Y.Z`` (no ``v`` prefix).
- Set the release name to the version string.
- Use the changelog content in the release notes.

Tags **must** follow :pep:`440`, since version metadata is derived using setuptools_scm_.

Once tagged, the GitHub Actions workflow will publish the package to PyPI and update the documentation site.

.. _setuptools_scm: https://github.com/pypa/setuptools-scm

.. _backport-release:

Backport releases
=================

To patch older major/minor versions, use a **release branch** named after the ``X.Y`` version.

Creating a release branch
-------------------------

If a release branch does not yet exist:

.. code-block:: bash

   git checkout X.Y.Z
   git checkout -b X.Y
   git push -u origin X.Y

Developing on a release branch
------------------------------

Use the release branch for all patch-level updates.
Backport changes from ``main`` using ``git cherry-pick`` if needed.

Releasing from a release branch
-------------------------------

Follow the same process as a regular release, but tag from the release branch instead of ``main``.
``ci.yaml`` will still handle publishing to PyPI and updating the documentation.

