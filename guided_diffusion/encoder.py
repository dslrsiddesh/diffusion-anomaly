import torch as th
import torch.nn as nn
from guided_diffusion.unet import ResBlock, AttentionBlock, Downsample, Upsample, conv_nd, normalization

class AEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 128,
        channel_mult: tuple = (1, 1, 2, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        dims: int = 2,
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = False,
        num_heads: int = 1,
        use_new_attention_order: bool = False,
    ):
        print("-------------------- Encoder initiated in encoder.py --------------------")
        super().__init__()
        self.model_channels = model_channels
        self.time_embed_dim = model_channels * 4
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_head_channels = -1
        self.use_new_attention_order = use_new_attention_order
        
        # Create a dummy embedding
        self.dummy_emb = nn.Parameter(th.zeros(1, self.time_embed_dim), requires_grad=False)
        
        self.initial = nn.Sequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            nn.SiLU()
        )
        self.resblocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(
                        ch,
                        emb_channels=self.time_embed_dim,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = mult * model_channels
            self.resblocks.append(level_blocks)
            if level != len(channel_mult) - 1:
                self.downsamples.append(
                    Downsample(ch, conv_resample, dims=dims, out_channels=ch)
                )
        self.middle = nn.ModuleList([
            ResBlock(ch, emb_channels=self.time_embed_dim, dropout=dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=self.num_head_channels, use_new_attention_order=use_new_attention_order),
            ResBlock(ch, emb_channels=self.time_embed_dim, dropout=dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        ])
        
    def forward(self, x):
        features = {}
        # print(f"Input shape: {x.shape}")
        
        h = self.initial(x)
        # print(f"After initial shape: {h.shape}")
        
        # Create a batch of dummy embeddings
        emb = self.dummy_emb.expand(x.shape[0], -1)
        
        # Pass through ResBlocks and downsampling layers
        for i, level_blocks in enumerate(self.resblocks):
            # print(f"Level {i} starting shape: {h.shape}")
            for j, block in enumerate(level_blocks):
                h = block(h, emb=emb)
                # print(f"Level {i}, block {j}, shape after ResBlock: {h.shape}")
            res = 256 // (2 ** i)
            features[res] = h
            # print(f"Saving feature[{res}] with shape: {h.shape}")
            if i != len(self.resblocks) - 1:
                h = self.downsamples[i](h)
                # print(f"After downsample {i}, shape: {h.shape}")
        
        # Handle the middle layers
        # print(f"Before middle blocks shape: {h.shape}")
        for i, layer in enumerate(self.middle):
            if isinstance(layer, ResBlock):
                h = layer(h, emb=emb)
            else:
                h = layer(h)
            # print(f"After middle block {i}, shape: {h.shape}")
        
        features["bn"] = h  # Add the bottleneck features
        # print(f"Bottleneck feature[8] shape: {h.shape}")
        
        # print("All feature shapes:")
        for k in features.keys():
            print(f"feature[{k}]: {features[k].shape}")
            
        return features