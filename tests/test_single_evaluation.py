#!/usr/bin/env python3
"""
Smoke-test evaluate_specific_model with heavy dependencies mocked out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from merginguriel import evaluate_specific_model as eval_module
from merginguriel import naming_config as naming_config_module


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            batch = 1
        else:
            batch = len(texts)
        return {"input_ids": torch.zeros((batch, 4), dtype=torch.long)}


class DummyModel:
    def __init__(self):
        self.num_labels = 2
        self.config = SimpleNamespace(
            id2label={0: "intent_a", 1: "intent_b"},
            label2id={"intent_a": 0, "intent_b": 1},
            model_type="xlm-roberta",
            architectures=["XLMRobertaForSequenceClassification"],
        )

    def __call__(self, **inputs):
        batch = inputs["input_ids"].shape[0]
        logits = torch.zeros((batch, self.num_labels))
        return SimpleNamespace(logits=logits)


class DummyDataset:
    def __init__(self):
        self._items = [
            {"utt": "hello world", "intent": 0},
            {"utt": "book flight", "intent": 1},
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            subset = self._items[idx]
            return {
                "utt": [item["utt"] for item in subset],
                "intent": [item["intent"] for item in subset],
            }
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)


@pytest.mark.usefixtures()
def test_evaluate_specific_model_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_module.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: DummyTokenizer()))
    monkeypatch.setattr(eval_module.AutoModelForSequenceClassification, "from_pretrained", classmethod(lambda cls, *a, **k: DummyModel()))
    monkeypatch.setattr(eval_module, "load_dataset", lambda *a, **k: DummyDataset())
    monkeypatch.setattr(naming_config_module.naming_manager, "detect_model_family_from_path", lambda *a, **k: "xlm-roberta-base")
    monkeypatch.setattr(eval_module.torch.cuda, "is_available", lambda: False)

    results = eval_module.evaluate_specific_model(
        model_name="dummy/model",
        locale="en-US",
        eval_folder=str(tmp_path),
    )

    assert results["evaluation_info"]["locale"] == "en-US"
    assert "accuracy" in results["performance"]
