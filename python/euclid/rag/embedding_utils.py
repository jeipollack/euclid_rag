# This program is licensed under the GNU Lesser General Public License
# (LGPL) v3.0, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""embedding_utils.py.

Utility functions for embedding documents and queries using
transformer-based models.

This module provides a base class for embedding models
and a specific implementation for the E5 model from
HuggingFace.
"""

import torch
from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer


# Device detection helper
def get_device() -> str:
    """Return the best available device: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class BaseEmbedder(Embeddings):
    """Abstract base for embedding models."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


# Embedding 'generator' using E5 from HuggingFace
class E5MpsEmbedder(BaseEmbedder):
    """E5 embedding model running on GPU, MPS or CPU."""

    def __init__(
        self, model_name: str = "intfloat/e5-small-v2", batch_size: int = 16
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = get_device()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vector representations."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            toks = self.tokenizer(
                texts[i : i + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                # Extract CLS (summary) token -> transformer specific
                out.extend(
                    self.model(**toks)
                    .last_hidden_state[:, 0, :]
                    .cpu()
                    .numpy()
                    .tolist()
                )
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed([text])[0]
