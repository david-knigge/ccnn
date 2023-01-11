import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, no_channels):
        super().__init__()
        self.ln = torch.nn.LayerNorm(no_channels)

    def forward(self, x):
        y = self.ln(x.transpose(1, -1)).transpose(-1, 1)
        return y


class GraphBatchNorm(torch.nn.Module):
    def __init__(self, no_channels):
        super().__init__()
        self.bn = self.module = torch.nn.BatchNorm1d(no_channels)
        self.no_channels = no_channels

    def forward(self, data):
        """

        :param x: Input feature vector of size [batch, num_nodes, num_channels]. Batchnorm1d expects the
            spatial and channel dimensions in reverse ordering. Why the fuck does this work in PyG's implementation???
        """
        batch_size, num_nodes = data.x.shape[0], data.x.shape[1]
        data.x = self.bn(data.x.view(batch_size * num_nodes, -1)).view(
            batch_size, num_nodes, -1
        )
        return data

# class LayerNorm(nn.Module):
#     r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         # if self.data_format == "channels_last":
#         #     return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         # elif self.data_format == "channels_first":
#         #     u = x.mean(1, keepdim=True)
#         #     s = (x - u).pow(2).mean(1, keepdim=True)
#         #     x = (x - u) / torch.sqrt(s + self.eps)
#         #     x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         #     return x
#         x = x.permute(0, 2, 1)
#         out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         out = out.permute(0, 2, 1)
#         return out
