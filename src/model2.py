"""
Reference code: 
https://github.com/huggingface/pytorch-image-models/tree/main

https://github.com/pengzhiliang/Conformer?tab=readme-ov-file 
"""


import torch
import torchvision
import torch.nn as nn
from torchvision.ops.misc import MLP, Conv2dNormActivation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
#ViT
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from processing import drop_path, trunc_normal_


class MLPBlock(MLP):
    """A class representing a Transformer MLP block."""

    # Class variable to track version
    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        """
        Initialize the MLPBlock.

        Args:
            in_dim (int): Input dimension.
            mlp_dim (int): Dimension of the MLP.
            dropout (float): Dropout probability.

        """
        # Call the constructor of the superclass (MLP)
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],  # MLP layer dimensions
            activation_layer=nn.GELU,  # Activation function
            inplace=None,
            dropout=dropout,  # Dropout probability
        )

        # Initialize weights and biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Initialize weights using Xavier initialization
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # Initialize biases with small random values

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Load the model state from the state dictionary.

        Args:
            state_dict: The state dictionary containing model parameters.
            prefix: Prefix for the state dictionary keys.
            local_metadata: Local metadata about the state dictionary.
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict function.
            missing_keys: A list to which it will append the missing keys found in state_dict that are expected by this module.
            unexpected_keys: A list to which it will append the unexpected keys found in state_dict that are not expected by this module.
            error_msgs: A list of error messages associated with loading the state_dict.

        """
        # Get the version from local metadata
        version = local_metadata.get("version", None)

        # Check if version is None or less than 2
        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    # Replace old keys with new keys
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        # Call the superclass's _load_from_state_dict method
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )



class Encoder(nn.Module):
    """A class representing the Transformer Model Encoder for sequence-to-sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        pos_embedding: torch.Tensor,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Initialize the Encoder.

        Args:
            seq_length (int): Length of the input sequence.
            num_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension.
            mlp_dim (int): Dimension of the MLP.
            dropout (float): Dropout probability.
            pos_embedding (torch.Tensor): Positional embeddings.
            attention_dropout (float): Attention dropout probability.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. Defaults to partial(nn.LayerNorm, eps=1e-6).

        """
        super().__init__()
        # Initialize positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.empty(1, 66, hidden_dim).normal_(std=pos_embedding)
        )  # from BERT
        # self.pos_embedding = nn.Parameter(torch.randn((1 // seq_length) **2, hidden_dim))
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Define encoder layers
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        # Layer normalization
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        """
        Perform forward pass.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # Check input dimensions
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        # Add positional embeddings to input
        input += self.pos_embedding
        # Apply dropout, encoder layers, and layer normalization
        return self.ln(self.layers(self.dropout(input)))


class EncoderBlock(nn.Module):
    """A class representing the Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Initialize the EncoderBlock.

        Args:
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension.
            mlp_dim (int): Dimension of the MLP.
            dropout (float): Dropout probability.
            attention_dropout (float): Attention dropout probability.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. Defaults to partial(nn.LayerNorm, eps=1e-6).

        """
        super().__init__()
        self.num_heads = num_heads

        # Layer normalization for attention block
        self.ln_1 = norm_layer(hidden_dim)
        # Multi-head self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for MLP block
        self.ln_2 = norm_layer(hidden_dim)
        # MLP block
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        """
        Perform forward pass.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # Check input dimensions
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        # Apply layer normalization for attention block
        x = self.ln_1(input)
        # Apply multi-head self-attention mechanism
        x, _ = self.self_attention(x, x, x, need_weights=False)
        # Apply dropout
        x = self.dropout(x)
        # Residual connection
        x = x + input

        # Apply layer normalization for MLP block
        y = self.ln_2(x)
        # Apply MLP block
        y = self.mlp(y)
        # Residual connection
        return x + y


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        stations: int,
        past_timesteps: int,
        future_timesteps: int,
        num_vars: int,
        pos_embedding: torch.Tensor,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 128,
        mlp_dim: int = 768,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.future_timesteps = future_timesteps
        self.past_timesteps = past_timesteps
        self.stations = stations
        self.timesteps = future_timesteps + past_timesteps
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.num_vars = num_vars

        self.mlp = torchvision.ops.MLP(
            768, [hidden_dim], None, torch.nn.GELU, dropout=dropout
        )

        seq_length = stations * (future_timesteps + past_timesteps)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            pos_embedding,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if hasattr(self.heads, "pre_logits") and isinstance(
            self.heads.pre_logits, nn.Linear
        ):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in)
            )
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        # n = batch size
        # h = number of stations
        # w = number of time steps
        # c = number of features
        print("Error", x.shape)
        # n, h, w, c = x.shape
        # torch._assert(
        #     h == self.stations,
        #     f"Wrong image height! Expected {self.stations} but got {h}!",
        # )
        # torch._assert(
        #     w == self.timesteps,
        #     f"Wrong image width! Expected {self.timesteps} but got {w}!",
        # )

        # # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))

        # x = x.reshape(n, h * w, self.num_vars)
        x = self.mlp(x)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        # Classifier "token" is the future prediction - we will probably just want to select just some of these variables.
        # x \in (batch, stations * timesteps + 1, num_classes = 1)
        x = x[
            :, -(self.stations * self.future_timesteps) :, :
        ]  # this shape is (batch, stations, num_classes = 1)


        x = self.heads(x)  # is a linear transformation from hidden_dim to 1
        x = x.reshape(n, self.stations, self.future_timesteps, self.num_classes)

        return (
            x.squeeze()
        )  # logically we are saying return one value for the each future timestep for each station (interpreted as error)


class ConvBlock(nn.Module):
    """A class representing a convolutional block."""

    def __init__(
        self,
        inplanes,
        outplanes,
        stride=1,
        res_conv=False,
        act_layer=nn.ReLU,
        groups=1,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
        drop_block=None,
        drop_path=None,
    ):
        """
        Initialize the ConvBlock.

        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            res_conv (bool, optional): Whether to use a residual convolution. Defaults to False.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.ReLU.
            groups (int, optional): Number of groups for the convolutional layers. Defaults to 1.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. Defaults to partial(nn.BatchNorm2d, eps=1e-6).
            drop_block (object, optional): Dropout block. Defaults to None.
            drop_path (object, optional): Dropout path. Defaults to None.

        """
        super(ConvBlock, self).__init__()

        # Calculate expansion and intermediate planes
        expansion = 4
        med_planes = outplanes // expansion

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            med_planes,
            med_planes,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        # Residual connection convolutional layer if needed
        if res_conv:
            self.residual_conv = nn.Conv2d(
                inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        """Initialize the last batch normalization layer to zero."""
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        """
        Perform forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            x_t (torch.Tensor, optional): Tensor from previous time step. Defaults to None.
            return_x_2 (bool, optional): Whether to return intermediate tensor. Defaults to True.

        Returns:
            torch.Tensor: Output tensor.

        """
        residual = x

        print("convBLOCK1", x.shape)
        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        print("convBLOCK2", x.shape)
        # Second convolutional layer
        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)
        print("convBLOCK3", x.shape, x2.shape)
        # Third convolutional layer
        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        # Dropout path
        if self.drop_path is not None:
            x = self.drop_path(x)

        # Residual connection
        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x



class FCUDown(nn.Module):
    """A class representing a down-sampling unit for converting CNN feature maps to Transformer patch embeddings."""

    def __init__(
        self,
        inplanes,
        outplanes,
        dw_stride,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Initialize the FCUDown.

        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            dw_stride (int): Down-sampling stride.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. Defaults to partial(nn.LayerNorm, eps=1e-6).

        """
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        # Projection convolutional layer
        self.conv_project = nn.Conv2d(
            inplanes, outplanes, kernel_size=1, stride=1, padding=0
        )
        # Average pooling layer
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        """
        Perform forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            x_t (torch.Tensor): Tensor from previous time step.

        Returns:
            torch.Tensor: Output tensor.

        """
        print("FCUDOWN", x.shape)
        x = self.conv_project(x)  # [N, C, H, W]
        print("FCUDOWN0", x.shape)
        # Sample pooling and flattening
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        print("FCUDOWN1", x.shape)
        # Concatenate with tensor from previous time step
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        print("FCUDOWN2", x.shape)

        return x



class FCUUp(nn.Module):
    """A class representing an up-sampling unit for converting Transformer patch embeddings to CNN feature maps."""

    def __init__(
        self,
        inplanes,
        outplanes,
        up_stride,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
    ):
        """
        Initialize the FCUUp.

        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            up_stride (int): Up-sampling stride.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.ReLU.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. Defaults to partial(nn.BatchNorm2d, eps=1e-6).

        """
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        print("inplanes", inplanes)
        print("outplanes", outplanes)
        self.conv_project = nn.Conv2d(
            inplanes, outplanes, kernel_size=1, stride=1, padding=0
        )
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        """
        Perform forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            H (int): Height of the output tensor.
            W (int): Width of the output tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        #this is a difference between original model and mine ***
        B,C,_ = x.shape
        # Reshape the tensor
        print("new_shape", x.shape)
        print("H,W", H, W)
        # x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        # x_r = x.transpose(1, 2).reshape(B, C, H, W)
        # print("xr", x_r.shape)
        # Apply convolution, batch normalization, and activation

        print(self.conv_project(x))

        x_r = self.act(self.bn(self.conv_project(x)))

        # Upsample the tensor
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """A special case for ConvBlock with down-sampling."""

    def __init__(
        self,
        inplanes,
        act_layer=nn.ReLU,
        groups=1,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
        drop_block=None,
        drop_path=None,
    ):
        """
        Initialize the Med_ConvBlock.

        Args:
            inplanes (int): Number of input channels.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.ReLU.
            groups (int, optional): Number of groups for the convolution operation. Defaults to 1.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. 
                Defaults to partial(nn.BatchNorm2d, eps=1e-6).
            drop_block (nn.Module, optional): Block dropout layer. Defaults to None.
            drop_path (nn.Module, optional): Drop path layer. Defaults to None.

        """
        super(Med_ConvBlock, self).__init__()

        # Define the expansion factor for the intermediate planes
        expansion = 4
        med_planes = inplanes // expansion  # Calculate the number of intermediate channels

        # First convolutional layer with kernel size 1x1
        self.conv1 = nn.Conv2d(
            inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = norm_layer(med_planes)  # Batch normalization layer
        self.act1 = act_layer(inplace=True)  # Activation layer

        # Second convolutional layer with kernel size 3x3
        self.conv2 = nn.Conv2d(
            med_planes,
            med_planes,
            kernel_size=3,
            stride=1,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(med_planes)  # Batch normalization layer
        self.act2 = act_layer(inplace=True)  # Activation layer

        # Third convolutional layer with kernel size 1x1
        self.conv3 = nn.Conv2d(
            med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = norm_layer(inplanes)  # Batch normalization layer
        self.act3 = act_layer(inplace=True)  # Activation layer

        # Dropout layers
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        """Initialize the weights of the last batch normalization layer to zeros."""
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        """
        Perform forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional layers and activations.

        """
        residual = x  # Store the input tensor for residual connection

        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)  # Batch normalization
        if self.drop_block is not None:
            x = self.drop_block(x)  # Apply dropout if specified
        x = self.act1(x)  # Apply activation function

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)  # Batch normalization
        if self.drop_block is not None:
            x = self.drop_block(x)  # Apply dropout if specified
        x = self.act2(x)  # Apply activation function

        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)  # Batch normalization
        if self.drop_block is not None:
            x = self.drop_block(x)  # Apply dropout if specified

        if self.drop_path is not None:
            x = self.drop_path(x)  # Apply drop path if specified

        x += residual  # Add the residual connection
        x = self.act3(x)  # Apply activation function

        return x

class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keeping feature maps for the CNN block
    and patch embeddings for the transformer encoder block.
    """

    def __init__(
        self,
        inplanes,
        outplanes,
        res_conv,
        stride,
        dw_stride,
        embed_dim,
        stations,
        past_timesteps,
        forecast_hour,
        in_chans,
        pos_embedding,
        num_layers,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        last_fusion=False,
        num_med_block=0,
        groups=1,
    ):
        """
        Initialize the ConvTransBlock.

        Args:
            inplanes (int): Number of input channels.
            outplanes (int): Number of output channels.
            res_conv (bool): Whether to use residual convolution in the fusion block.
            stride (int): Stride of the convolution operation.
            dw_stride (int): Stride of the down-sampling operation.
            embed_dim (int): Dimension of the embedding.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of the MLP dimension to the embedding dimension. 
                Defaults to 4.0.
            qkv_bias (bool, optional): Whether to include bias in the qkv calculation. Defaults to False.
            qk_scale (float, optional): Scale factor for qk calculation. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Dropout rate for attention weights. Defaults to 0.0.
            drop_path_rate (float, optional): Dropout rate for drop path. Defaults to 0.0.
            last_fusion (bool, optional): Whether it is the last fusion block. Defaults to False.
            num_med_block (int, optional): Number of medium blocks. Defaults to 0.
            groups (int, optional): Number of groups for the convolution operation. Defaults to 1.

        """
        super(ConvTransBlock, self).__init__()

        expansion = 4
        # CNN block
        print("CNN Block")
        print("inplanes", inplanes)
        print("outplanes", outplanes)
        print("res_conv", res_conv)
        print("stride", stride)
        print("groups", groups)
        self.cnn_block = ConvBlock(
            inplanes=inplanes,
            outplanes=outplanes,
            res_conv=res_conv,
            stride=stride,
            groups=groups,
        )

        # Fusion block
        if last_fusion:
            self.fusion_block = ConvBlock(
                inplanes=outplanes,
                outplanes=outplanes,
                stride=2,
                res_conv=True,
                groups=groups,
            )
        else:
            self.fusion_block = ConvBlock(
                inplanes=outplanes, outplanes=outplanes, groups=groups
            )

        # Medium blocks
        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                # Initialize medium convolution blocks
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        # Squeeze block
        self.squeeze_block = FCUDown(
            inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride
        )

        print("embed_dim", embed_dim)
        print("outplanes", outplanes)
        print("expand", expansion)
        print("DW", dw_stride)
        # Expand block
        self.expand_block = FCUUp(
            inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride
        )

        # Transformer block
        self.trans_block = VisionTransformer(
            stations=stations,
            past_timesteps=past_timesteps,
            future_timesteps=forecast_hour,
            num_vars=in_chans,
            pos_embedding=pos_embedding,
            num_layers=num_heads,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            x_t (torch.Tensor): Transformer patch embeddings.

        Returns:
            torch.Tensor: Output tensor.

        """
        # CNN block forward pass
        x, x2 = self.cnn_block(x)

        print("Here is where the shapes come from")
        print(x.shape)
        print(x2.shape)

        _, _, H, W = x2.shape

        print("x_st_0", x2.shape)
        print("x_st_0", x2.shape)
        # Squeeze block forward pass
        x_st = self.squeeze_block(x2, x_t)

        print("x_st", x_st.shape)
        print("x_t", x_t.shape)
        x_st = x_st.permute(2,1,0)
        x_t = x_t.permute(2,1,0)
        print("x_st", x_st.shape)
        print("x_t", x_t.shape)



        # Transformer block forward pass
        x_t = self.trans_block(x_st + x_t)
        print("x_t", x_t.shape)

        # Medium convolution blocks forward pass
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        # Expand block forward pass
        print("DW", self.dw_stride)
        print("H, W", H, W)
        print("xt", x_t.shape)
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        print("xtr", x_t_r.shape)

        # Fusion block forward pass
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Module):
    """
    Conformer model architecture for vision tasks.
    """

    def __init__(
        self,
        patch_size,
        in_chans,
        stations,
        past_timesteps,
        forecast_hour,
        pos_embedding,
        num_layers,
        num_classes=1,
        base_channel=768, #batch_size
        channel_ratio=2,
        num_med_block=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        attention_dropout=0.0,
        drop_path_rate=0.0,
    ):
        """
        Initialize the Conformer model.

        Args:
            patch_size (int, optional): Patch size. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            base_channel (int, optional): Base number of channels. Defaults to 64.
            channel_ratio (int, optional): Channel ratio. Defaults to 4.
            num_med_block (int, optional): Number of medium blocks. Defaults to 0.
            embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
            depth (int, optional): Depth of the model. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of the MLP dimension to the embedding dimension. 
                Defaults to 4.0.
            qkv_bias (bool, optional): Whether to include bias in the qkv calculation. Defaults to False.
            qk_scale (float, optional): Scale factor for qk calculation. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Dropout rate for attention weights. Defaults to 0.0.
            drop_path_rate (float, optional): Dropout rate for drop path. Defaults to 0.0.
        """

        super().__init__()

        # Initialize parameters
        self.num_classes = num_classes
        self.num_features = in_chans
        assert depth % 3 == 0

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: Get the feature maps by convolution block
        self.conv1 = nn.Conv2d(
            in_chans, base_channel, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channel)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 1st stage
        stage_1_channel = int(base_channel * channel_ratio)
        print("stage1", stage_1_channel)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(
            inplanes=base_channel, outplanes=stage_1_channel, res_conv=True, stride=1
        )
        self.trans_patch_conv = nn.Conv2d(
            base_channel,
            embed_dim,
            kernel_size=trans_dw_stride,
            stride=trans_dw_stride,
            padding=0,
        )
        self.trans_1 = EncoderBlock(
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=768,
            dropout=0.0,
            attention_dropout=0.0
        )

        # 2~4 stages
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    stage_1_channel,
                    stage_1_channel,
                    False,
                    1,
                    dw_stride=trans_dw_stride,
                    embed_dim=embed_dim,
                    stations=stations,
                    past_timesteps=past_timesteps,
                    forecast_hour=forecast_hour,
                    in_chans=in_chans,
                    pos_embedding=pos_embedding,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                ),
            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        print("stage2", stage_2_channel)
        print("base", base_channel)
        print("channel_ratio", channel_ratio)
        # 5~8 stages
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            print(i)
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    inplanes = in_channel,
                    outplanes= stage_2_channel,
                    res_conv= res_conv,
                    stride = s,
                    dw_stride=trans_dw_stride // 2,
                    embed_dim=embed_dim,
                    stations=stations,
                    past_timesteps = past_timesteps,
                    forecast_hour=forecast_hour,
                    pos_embedding=pos_embedding,
                    num_layers=num_layers,
                    in_chans=in_chans,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=drop_path_rate,
                    num_med_block=num_med_block,
                ),
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stages
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    inplanes=in_channel,
                    outplanes= stage_3_channel,
                    res_conv = res_conv,
                    stride = s,
                    dw_stride=trans_dw_stride // 4,
                    embed_dim=embed_dim,
                    stations=stations,
                    past_timesteps=past_timesteps,
                    forecast_hour=forecast_hour,
                    pos_embedding=pos_embedding,
                    num_layers=num_layers,
                    in_chans = in_chans,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=drop_path_rate,
                    num_med_block=num_med_block,
                    last_fusion=last_fusion,
                ),
            )
        self.fin_stage = fin_stage

        trunc_normal_.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize model weights.

        Args:
            m (nn.Module): Module to initialize weights for.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Specify parameters to exclude from weight decay during optimization.
        """
        return {"cls_token"}

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list: List of output tensors (classification outputs).
        """
        B = x.shape[0]
        F = x.shape[3]
        stat_num = x.shape[1]
        seq = x.shape[2]
        cls_tokens = self.cls_token.expand(B, -1, -1)


        print("shape_00", x.shape)
        # changing input to shape [batch, features, stations, seq_length]
        x = x.permute(0,3,1,2)
        # Create a new tensor with the desired shape filled with zeros

        # Stem stage
        print("shape_0", x.shape)
        print(x)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        print("shape", x_base.shape)

        # 1st stage
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        print("0", x_t.shape)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        print("1", x_t.shape)
        x_t = self.trans_1(x_t)
        print("2", x_t.shape)

        # 2nd to final stages
        print("fin_stage", self.fin_stage)
        for i in range(2, self.fin_stage):
            print("FINAL", i)
            x, x_t = eval("self.conv_trans_" + str(i))(x, x_t)

        # Convolutional classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        # Transformer classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        return [conv_cls, tran_cls]
