import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer
from models_pvt_simclr import PVTSimCLR


class MMST_ViT(nn.Module):
    def __init__(
        self,
        out_dim=1,
        num_grid=4,
        num_short_term_seq=6,
        num_long_term_seq=12,
        num_year=5,
        pvt_backbone=None,
        context_dim=40,
        dim=None,
        batch_size=4,
        depth=2,
        heads=4,
        pool='cls',
        dim_head=64,
        dropout=0.2,
        emb_dropout=0.1,
        scale_dim=4,
        num_regencies=512,
        regency_emb_dim=64,
        aux_cont_dim=6,
        use_regency_embedding=True,
    ):
        super().__init__()

        assert pool in {'cls', 'mean'}

        self.batch_size = batch_size
        self.num_grid = num_grid
        self.num_short_term_seq = num_short_term_seq
        self.num_long_term_seq = num_long_term_seq
        self.num_year = num_year
        self.context_dim = context_dim
        self.pool = pool
        self.use_regency_embedding = use_regency_embedding

        if pvt_backbone is None:
            pvt_backbone = PVTSimCLR(
                base_model='pvt_tiny',
                out_dim=512,
                context_dim=context_dim,
                pretrained=True,
            )

        self.pvt_backbone = pvt_backbone

        backbone_dim = None
        if hasattr(self.pvt_backbone, "proj") and hasattr(self.pvt_backbone.proj, "out_features"):
            backbone_dim = self.pvt_backbone.proj.out_features
        elif hasattr(self.pvt_backbone, "out_dim"):
            backbone_dim = self.pvt_backbone.out_dim
        if backbone_dim is None:
            backbone_dim = 512

        if dim is None:
            dim = backbone_dim

        self.backbone_dim = backbone_dim
        self.dim = dim

        self.feature_proj = nn.Identity() if self.backbone_dim == self.dim else nn.Linear(self.backbone_dim, self.dim)

        self.proj_context = nn.LazyLinear(num_short_term_seq * dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.norm1 = nn.LayerNorm(dim)

        if self.use_regency_embedding:
            self.regency_embedding = nn.Embedding(num_regencies, regency_emb_dim)
        else:
            self.regency_embedding = None

        self.aux_mlp = nn.Sequential(
            nn.LayerNorm(aux_cont_dim),
            nn.Linear(aux_cont_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fusion_dim = dim + 128 + (regency_emb_dim if self.use_regency_embedding else 0)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def _prepare_short_context(self, ys, b, t, g):
        if ys is None:
            raise ValueError("ys cannot be None")

        if ys.dim() == 5:
            pass
        elif ys.dim() == 4:
            ys = ys.unsqueeze(2)
        elif ys.dim() == 3:
            ys = ys.unsqueeze(2).unsqueeze(2)
        elif ys.dim() == 2:
            ys = ys.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported ys shape: {tuple(ys.shape)}")

        if ys.shape[0] != b:
            raise ValueError(f"ys batch mismatch: expected {b}, got {ys.shape[0]}")
        if ys.shape[-1] != self.context_dim:
            raise ValueError(f"ys last dim must be {self.context_dim}, got {ys.shape[-1]}")

        if ys.shape[1] != t:
            if ys.shape[1] == 1:
                ys = ys.repeat(1, t, 1, 1, 1)
            else:
                rep = (t + ys.shape[1] - 1) // ys.shape[1]
                ys = ys.repeat(1, rep, 1, 1, 1)[:, :t]

        if ys.shape[2] != g:
            if ys.shape[2] == 1:
                ys = ys.repeat(1, 1, g, 1, 1)
            else:
                rep = (g + ys.shape[2] - 1) // ys.shape[2]
                ys = ys.repeat(1, 1, rep, 1, 1)[:, :, :g]

        return ys

    def _prepare_long_context(self, yl, b, t, device):
        if yl is None:
            yl = torch.zeros(b, self.num_year, self.num_long_term_seq, self.context_dim, device=device)

        if yl.dim() == 5:
            if yl.shape[2] == 1:
                yl = yl.squeeze(2)
            else:
                yl = rearrange(yl, 'b y m n d -> b y (m n) d')
        elif yl.dim() == 4:
            pass
        elif yl.dim() == 3:
            yl = yl.unsqueeze(1)
        elif yl.dim() == 2:
            yl = yl.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported yl shape: {tuple(yl.shape)}")

        if yl.shape[0] != b:
            raise ValueError(f"yl batch mismatch: expected {b}, got {yl.shape[0]}")
        if yl.shape[-1] != self.context_dim:
            raise ValueError(f"yl last dim must be {self.context_dim}, got {yl.shape[-1]}")

        yl = rearrange(yl, 'b ... d -> b (...) d')
        yl = rearrange(yl, 'b n d -> b (n d)')
        yl = self.proj_context(yl)
        yl = rearrange(yl, 'b (t d) -> b t d', t=t, d=self.dim)

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        yl = torch.cat((cls_temporal_tokens, yl), dim=1)
        yl = self.norm1(yl)
        return yl

    def forward_features(self, x, ys):
        b, t, g, _, _, _ = x.shape
        ys = self._prepare_short_context(ys, b, t, g)

        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        ys = rearrange(ys, 'b t g n d -> (b t g) n d')

        total = x.shape[0]
        chunks = total // self.batch_size if total % self.batch_size == 0 else total // self.batch_size + 1

        outputs = []
        for i in range(chunks):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, total)

            x_tmp = x[start:end]
            ys_tmp = ys[start:end]

            x_hat_tmp = self.pvt_backbone(x_tmp, context=ys_tmp)
            if x_hat_tmp.dim() == 3:
                x_hat_tmp = x_hat_tmp[:, 0]

            x_hat_tmp = self.feature_proj(x_hat_tmp)
            outputs.append(x_hat_tmp)

        return torch.cat(outputs, dim=0)

    def forward(self, x, ys=None, yl=None, regency_idx=None, aux_cont=None):
        b, t, g, _, _, _ = x.shape

        x = self.forward_features(x, ys)
        x = rearrange(x, '(b t g) d -> b t g d', b=b, t=t, g=g)

        cls_space_tokens = repeat(self.space_token, '() g d -> b t g d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x = x + self.pos_embedding[:, :t, :(g + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t g d -> (b t) g d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) d -> b t d', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        yl = self._prepare_long_context(yl, b, t, x.device)
        x = self.temporal_transformer(x, yl)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        if aux_cont is None:
            aux_feat = torch.zeros(b, 128, device=x.device)
        else:
            aux_feat = self.aux_mlp(aux_cont)

        if self.use_regency_embedding:
            if regency_idx is None:
                regency_feat = torch.zeros(b, self.regency_embedding.embedding_dim, device=x.device)
            else:
                regency_feat = self.regency_embedding(regency_idx)
            fused = torch.cat([x, regency_feat, aux_feat], dim=-1)
        else:
            fused = torch.cat([x, aux_feat], dim=-1)

        return self.mlp_head(fused)


if __name__ == "__main__":
    x = torch.randn((1, 6, 4, 3, 224, 224))
    ys = torch.randn((1, 2, 40))
    yl = torch.randn((1, 1, 40))

    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=40, pretrained=False)

    model = MMST_ViT(
        out_dim=1,
        pvt_backbone=pvt,
        dim=512,
        context_dim=40,
        num_grid=4,
        use_regency_embedding=True,
    )

    z = model(x, ys=ys, yl=yl, regency_idx=torch.tensor([0]), aux_cont=torch.randn(1, 6))
    print(z)
    print(z.shape)