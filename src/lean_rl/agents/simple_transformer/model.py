import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
    ):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)
        output = self.transformer(src_embedded, tgt_embedded)
        return self.fc_out(output)
