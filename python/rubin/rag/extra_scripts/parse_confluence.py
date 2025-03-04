#
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

"""Load Confluence documents, process metadata, and upload them to Weaviate
with optional chunking.
"""

from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document
from universal_weaviate_uploader import push_docs_to_weaviate

# Instantiate ConfluenceLoader
loader = ConfluenceLoader("https://confluence.lsstcorp.org")

# Load documents with specified parameters
docs = loader.load(
    space_key="DM",
    include_archived_content=False,
    include_restricted_content=False,
    include_attachments=False,
    max_pages=10000,
    include_comments=True,
    keep_markdown_format=True,
    keep_newlines=True,
)

# Convert the loaded documents to a list
docs_list = list(docs)

new_documents = []
for doc in docs_list:
    metadata = doc.metadata.copy()
    metadata["pageid"] = metadata.pop("id")
    metadata["source_key"] = "confluence"
    new_doc = Document(page_content=doc.page_content, metadata=metadata)
    new_documents.append(new_doc)

# Push documents to Weaviate
push_docs_to_weaviate(new_documents, do_chunk=True)
