# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Define and retrieve the version string for the package."""

import pkgutil

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]

__path__ = pkgutil.extend_path(__path__, __name__)

import importlib.resources
import pathlib

with importlib.resources.path("euclid", "static") as static_path:
    STATIC_DIR = pathlib.Path(static_path)
