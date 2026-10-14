"""Ordinal class anchors r_1..r_C from a frozen text encoder applied to the rank prompts, followed by a trainable
linear projection into the shared 1024-d space."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from solid.model import EMB

HF_TEXT = {
    'roberta': 'roberta-base',
    'pubmedbert': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
}


class CONCHText(nn.Module):
    """Text tower of CONCH (https://huggingface.co/MahmoodLab/conch, gated: accept the terms and `huggingface-cli login`)."""

    width = 512

    def __init__(self):
        super().__init__()
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        model, _ = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/conch')
        del model.visual
        self.model, self.tokenizer = model, get_tokenizer()

    def forward(self, texts, device):
        from conch.open_clip_custom import tokenize
        return self.model.encode_text(tokenize(texts=texts, tokenizer=self.tokenizer).to(device), normalize=False)


class HFText(nn.Module):
    """BERT-style encoder pooled as CLS + mean over all 77 positions."""

    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(HF_TEXT[name])
        self.model = AutoModel.from_pretrained(HF_TEXT[name])
        self.width = self.model.config.hidden_size

    def forward(self, texts, device):
        t = self.tok(texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(device)
        h = self.model(**t).last_hidden_state
        return h[:, 0] + h.mean(1)


class TextAnchors(nn.Module):
    """prompts: one string per class, in rank order. The text encoder is frozen and its outputs are cached; only the
    projection is trained. `center` removes the component shared by all prompts so the anchors start apart."""

    def __init__(self, prompts, encoder='conch', center=True):
        super().__init__()
        self.prompts, self.center = list(prompts), center
        self.enc = CONCHText() if encoder == 'conch' else HFText(encoder)
        for p in self.enc.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.enc.width, EMB)
        if encoder != 'conch':
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self._cache = None

    def forward(self, device):
        if self._cache is None:
            with torch.no_grad():
                self.enc.eval()
                self._cache = self.enc(self.prompts, device).float()
        h = self._cache
        if self.center:
            h = h - h.mean(0, keepdim=True)
        return F.normalize(self.proj(h), dim=-1)
