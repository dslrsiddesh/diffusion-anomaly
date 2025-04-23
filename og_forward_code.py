def forward(self, x, timesteps, y=None, encoder_features=None):
    """
    Apply the model to an input batch.

    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """
    print("---------------------------UNetModel forward method called---------------------------")
    if encoder_features is not None:
        print("------ Received encoder features in UNetModel forward------")

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"

    hs = []
    emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

    if self.num_classes is not None:
        assert y.shape == (x.shape[0],)
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for module in self.input_blocks:
        h = module(h, emb)
        hs.append(h)
    h = self.middle_block(h, emb)
    for module in self.output_blocks:
        h = th.cat([h, hs.pop()], dim=1)
        h = module(h, emb)
    h = h.type(x.dtype)
    return self.out(h)