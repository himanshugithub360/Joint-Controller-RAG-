"""E5 query encoder, matching Search-R1's retrieval server exactly but runnable on CPU.

Adapted from Search-R1 search_r1/search/retrieval_server.py (Encoder.encode and pooling,
commit 598e61bd, Apache-2.0). Changes: no hard-coded .cuda(), batching added. Same
behaviour: "query: " prefix, mean pooling over the attention mask, L2 normalisation,
max_length 256, float32 output.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5QueryEncoder:
    def __init__(self, model_name: str = "intfloat/e5-base-v2", max_length: int = 256,
                 device: str | None = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, queries: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(queries), self.batch_size):
            batch = [f"query: {q}" for q in queries[i : i + self.batch_size]]
            inputs = self.tokenizer(batch, max_length=self.max_length, padding=True,
                                    truncation=True, return_tensors="pt").to(self.device)
            hidden = self.model(**inputs, return_dict=True).last_hidden_state
            emb = torch.nn.functional.normalize(mean_pool(hidden, inputs["attention_mask"]), dim=-1)
            out.append(emb.cpu().numpy().astype(np.float32, order="C"))
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)
