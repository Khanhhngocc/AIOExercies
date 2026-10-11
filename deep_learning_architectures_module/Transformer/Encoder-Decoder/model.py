import torch
import torch.nn as nn
import math

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device='cpu'):
        super().__init__()
        self.device = device
        self.word_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim
        )

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        output1 = self.word_emb(x)
        output2 = self.pos_emb(positions)
        output = output1 + output2

        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=ff_dim,
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=ff_dim,
                out_features=embed_dim,
                bias=True
            )
        )

        self.layernorm_1 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )

        self.layernorm_2 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(query + attn_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        out_2 = self.layernorm_2(out_1 + ffn_output)

        return out_2


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            src_vocab_size, embed_dim, max_length, device
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    ff_dim,
                    dropout
                ) for i in range(num_layers)
            ]
        )

    def forward(self, x):
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output, output, output)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=ff_dim,
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=ff_dim,
                out_features=embed_dim,
                bias=True
            )
        )

        self.layernorm_1 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )
        self.layernorm_2 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )
        self.layernorm_3 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6
        )
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.attn(x, x, x, attn_mask=tgt_mask)
        attn_output = self.dropout_1(attn_output)
        out_1 = self.layernorm_1(x + attn_output)
        attn_output, _ = self.cross_attn(out_1, enc_output, enc_output, attn_mask=src_mask)

        attn_output = self.dropout_2(attn_output)
        out_2 = self.layernorm_2(out_1 + attn_output)

        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_3(ffn_output)
        out_3 = self.layernorm_3(out_2 + ffn_output)

        return out_3


class TransformerDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.embdding = TokenAndPositionEmbedding(
            tgt_vocab_size,
            embed_dim,
            max_length,
            device
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim, num_heads, ff_dim, dropout
                ) for i in range(num_layers)
            ]
        )

    def forward(self, x, enc_ouput, src_mask, tgt_mask):
        output = self.embdding(x)
        for layer in self.layers:
            output = layer(output, enc_ouput, src_mask, tgt_mask)

        return output

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim, max_length, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            src_vocab_size,
            emb_dim,
            max_length,
            num_layers,
            num_heads,
            ff_dim,
            dropout,
            device
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            emb_dim,
            max_length,
            num_layers,
            num_heads,
            ff_dim,
            dropout,
            device
        )
        self.fc = nn.Linear(emb_dim, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        src_mask = torch.zeros(
            (src_seq_len, src_seq_len),
            device=self.device
        ).type(torch.bool)

        tgt_mask = (torch.triu(torch.ones(
            (tgt_seq_len, tgt_seq_len),
            device=self.device
        )) == 1).transpose(0, 1)

        tgt_mask = tgt_mask.float().masked_fill_(
            tgt_mask == 0, float('-inf')
        ).masked_fill_(tgt_mask == 1, float(0.0))

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        enc_output = self.encoder(src)
        dec_outout = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_outout)
        return output


batch_size = 128
src_vocab_size = 1000
tgt_vocab_size = 2000
embed_dim = 200
max_length = 1
num_layers = 2
num_heads = 4
ff_dim = 256

model = Transformer(
    src_vocab_size, tgt_vocab_size,
    embed_dim, max_length, num_layers, num_heads, ff_dim
)

src = torch.randint(
    high=2,
    size=(batch_size, max_length),
    dtype=torch.int64
)

tgt = torch.randint(
    high=2,
    size=(batch_size, max_length),
    dtype=torch.int64
)

prediction = model(src, tgt)
prediction.shape