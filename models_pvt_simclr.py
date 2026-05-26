import models_pvt
import torch
from attention import MultiModalTransformer
from torch import nn


class PVTSimCLR(nn.Module):
    def __init__(
        self,
        base_model='pvt_tiny',
        out_dim=512,
        context_dim=10,
        num_head=8,
        mm_depth=2,
        dropout=0.0,
        pretrained=True,
        gated_ff=True,
    ):
        super(PVTSimCLR, self).__init__()

        self.backbone = models_pvt.__dict__[base_model](pretrained=pretrained)
        num_ftrs = self.backbone.head.in_features

        self.proj = nn.Linear(num_ftrs, out_dim)
        self.proj_context = nn.Linear(context_dim, out_dim)

        dim_head = out_dim // num_head

        self.mm_transformer = MultiModalTransformer(
            out_dim,
            mm_depth,
            num_head,
            dim_head,
            context_dim=out_dim,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(context_dim)

    def forward(self, x, context=None):
        h = self.backbone.forward_features(x)

        if h.dim() == 2:
            h = h.unsqueeze(1)

        x = self.proj(h)

        if context is not None:
            context = self.norm1(context)

            if context.dim() == 2:
                context = context.unsqueeze(1)

            context = self.proj_context(context)
            x = self.mm_transformer(x, context=context)
        else:
            x = self.mm_transformer(x)

        return x[:, 0]


if __name__ == '__main__':
    model = PVTSimCLR(
        base_model='pvt_tiny',
        out_dim=512,
        context_dim=10,
        pretrained=False
    )

    x = torch.randn(8, 3, 224, 224)
    context = torch.randn(8, 4, 10)

    y = model(x, context=context)
    print(y.shape)