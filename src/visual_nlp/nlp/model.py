"""NLP encoder plus selected task heads. JSON assembly belongs to the pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

from visual_nlp.nlp.embeddings import DocumentEmbeddings
from visual_nlp.nlp.heads import DocumentHead, FieldHead, MLMHead, RelationHead
from visual_nlp.transformer.encoder import TransformerEncoder
from visual_nlp.utils.config import Config


class NLPModel(nn.Module):
    def __init__(
        self,
        embeddings: DocumentEmbeddings,
        encoder: TransformerEncoder,
        mlm_head: MLMHead,
        field_head: FieldHead | None,
        relation_head: RelationHead | None,
        document_head: DocumentHead | None,
        vocab_size: int,
        mask_id: int,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.mlm_head = mlm_head
        self.field_head = field_head
        self.relation_head = relation_head
        self.document_head = document_head
        self.vocab_size = vocab_size
        self.mask_id = mask_id

    @classmethod
    def from_config(cls, config: Config, vocab_size: int, mask_id: int, pad_id: int = 0) -> "NLPModel":
        d_model = int(config.model.d_model)
        embeddings = DocumentEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=int(config.model.max_length),
            dropout=float(config.model.get("dropout", 0.1)),
            use_layout=bool(config.embeddings.get("use_layout", True)),
            use_page=bool(config.embeddings.get("use_page", True)),
            pad_id=pad_id,
        )
        encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=int(config.model.n_heads),
            n_layers=int(config.model.n_layers),
            d_ff=int(config.model.d_ff),
            dropout=float(config.model.get("dropout", 0.1)),
            activation=str(config.model.get("activation", "gelu")),
        )
        field_labels = list(config.finetune.get("field_labels", []))
        doc_classes = list(config.finetune.get("document_classes", ["unknown"]))
        tasks = config.finetune.get("tasks", {})
        return cls(
            embeddings,
            encoder,
            MLMHead(d_model, vocab_size),
            FieldHead(d_model, len(field_labels)) if tasks.get("field_extraction", True) else None,
            RelationHead(d_model, len(field_labels)) if tasks.get("relation_extraction", False) else None,
            DocumentHead(d_model, len(doc_classes)) if tasks.get("document_classification", False) else None,
            vocab_size,
            mask_id,
        )

    def encode(self, batch: dict, use_layout: bool | None = None) -> torch.Tensor:
        embedded = self.embeddings(
            batch["input_ids"],
            token_boxes=batch.get("token_boxes"),
            page_ids=batch.get("page_ids"),
            use_layout=use_layout,
        )
        return self.encoder(embedded, padding_mask=batch.get("padding_mask"))

    def forward(self, batch: dict, mode: str = "pretrain") -> dict[str, torch.Tensor]:
        use_layout = False if mode == "pretrain" else None
        hidden = self.encode(batch, use_layout=use_layout)
        outputs: dict[str, torch.Tensor] = {"hidden": hidden}
        if mode == "pretrain":
            outputs["mlm_logits"] = self.mlm_head(hidden)
            return outputs
        if self.field_head is not None:
            outputs["field_logits"] = self.field_head(hidden)
        if self.relation_head is not None:
            outputs["relation_logits"] = self.relation_head(hidden, batch["padding_mask"])
        if self.document_head is not None:
            outputs["document_logits"] = self.document_head(hidden, batch["padding_mask"])
        return outputs
