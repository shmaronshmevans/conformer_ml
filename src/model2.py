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

# ViT
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from processing import drop_path, trunc_normal_

import numpy as np
import gc


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features
        hidden_features=None,  # Number of hidden features in the MLP
        out_features=None,  # Number of output features
        act_layer=nn.GELU,  # Activation function used in the MLP
        drop=0.0,  # Dropout probability
    ):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features in the MLP. Defaults to None (in_features).
            out_features (int, optional): Number of output features. Defaults to None (in_features).
            act_layer (torch.nn.Module, optional): Activation function used between the layers. Defaults to nn.GELU.
            drop (float, optional): Dropout probability. Defaults to 0.0.

        """
        super().__init__()

        # If out_features or hidden_features are not specified, default to in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define the first fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Activation function used between the layers
        self.act = act_layer()

        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Dropout layer with specified dropout probability
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).

        """
        # Pass input through the first fully connected layer
        x = self.fc1(x)

        # Apply activation function
        x = self.act(x)

        # Apply dropout
        x = self.drop(x)

        # Pass through the second fully connected layer
        x = self.fc2(x)

        # Apply dropout again
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """
        Multi-head self attention mechanism module.

        Args:
            dim (int): Dimensionality of the input features.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): Whether to include bias in the query, key, and value linear layers. Defaults to False.
            qk_scale ([type], optional): Scale factor for query and key. Defaults to None.
            attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            proj_drop (float, optional): Dropout probability for the output tensor. Defaults to 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        # Calculate dimensionality of each attention head
        head_dim = dim // num_heads

        # Set scale factor for query and key
        self.scale = qk_scale or head_dim**-0.5

        # Linear transformation layers for query, key, and value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Linear transformation layer for projecting output tensor
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, input_dim).
        """
        # Extract batch size (B), sequence length (N), and input dimension (C)
        B, N, C = x.shape

        # Linear transformation for query, key, and value, and reshape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into query, key, and value

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply softmax to get attention weights and apply dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values (attention mechanism)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Project output tensor and apply dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        """
        Transformer block consisting of multi-head self-attention mechanism and MLP.

        Args:
            dim (int): Dimensionality of the input features.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of hidden dimension size to input dimension size in MLP. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to include bias in the query, key, and value linear layers. Defaults to False.
            qk_scale ([type], optional): Scale factor for query and key. Defaults to None.
            drop (float, optional): Dropout probability for MLP and attention mechanism. Defaults to 0.0.
            attn_drop (float, optional): Dropout probability for attention weights. Defaults to 0.0.
            drop_path (float, optional): Dropout probability for stochastic depth (drop path). Defaults to 0.0.
            act_layer (torch.nn.Module, optional): Activation function used in the MLP. Defaults to nn.GELU.
            norm_layer (Callable[..., torch.nn.Module], optional): Normalization layer. Defaults to partial(nn.LayerNorm, eps=1e-6).
        """
        super().__init__()

        # Layer normalization for the input features
        self.norm1 = norm_layer(dim)

        # Multi-head self-attention mechanism
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer normalization for the output of attention mechanism
        self.norm2 = norm_layer(dim)

        # MLP module
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, input_dim).
        """
        # Pass input through layer normalization, attention mechanism, and apply drop path
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # Pass output through layer normalization, MLP, and apply drop path
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


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

        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        # Second convolutional layer
        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)
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
        stations,
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
        self.stations = stations
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
        x = self.conv_project(x)  # [N, C, H, W]
        # Sample pooling and flattening
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        # Concatenate with tensor from previous time step
        x = torch.cat([x_t[:, : self.stations][:, None, :].squeeze(1), x], dim=1)

        return x


class FCUUp(nn.Module):
    """A class representing an up-sampling unit for converting Transformer patch embeddings to CNN feature maps."""

    def __init__(
        self,
        inplanes,
        outplanes,
        up_stride,
        stations,
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
        self.stations = stations
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
        # this is a difference between original model and mine ***
        B, _, C = x.shape

        x_r = x[:, self.stations :].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
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
        med_planes = (
            inplanes // expansion
        )  # Calculate the number of intermediate channels

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
            inplanes=outplanes // expansion,
            outplanes=embed_dim,
            dw_stride=dw_stride,
            stations=stations,
        )

        # Expand block
        self.expand_block = FCUUp(
            inplanes=embed_dim,
            outplanes=outplanes // expansion,
            up_stride=dw_stride,
            stations=stations,
        )

        # Transformer block
        self.trans_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=dropout,
            attn_drop=attention_dropout,
            drop_path=drop_path_rate,
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

        _, _, H, W = x2.shape

        # Squeeze block forward pass
        x_st = self.squeeze_block(x2, x_t)
        x_t = adjust_tensor_shape(x_t, x_st.shape[1], 1)

        # Transformer block forward pass
        x_t = self.trans_block(x_st + x_t)

        # Medium convolution blocks forward pass
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        # Expand block forward pass
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)

        # Fusion block forward pass
        # pad x_t_r to be compatible with x
        x_t_r = adjust_tensor_shape(x_t_r, x.shape[2], 3)
        x_t_r = adjust_tensor_shape(x_t_r, x.shape[2], 2)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


# def adjust_tensor_shape(x_t, shape, dimension):
#     """
#     Adjusts the shape of tensor x_t to ensure that its first dimension equals 12.

#     Args:
#     - x: torch.Tensor, input tensor
#     - x_t: torch.Tensor, target tensor

#     Returns:
#     - Adjusted tensor x_t
#     """
#     current_dim = x_t.size(1)
#     if current_dim < 12:
#         # Calculate the amount needed to pad x_t's first dimension
#         pad_size = 12 - current_dim
#         # Pad x_t's first dimension to make it equal to 12
#         x_t = torch.nn.functional.pad(x_t, (0, 0, 0, pad_size))
#     return x_t


def adjust_tensor_shape(x_t, shape, dimension):
    """
    Adjusts the shape of tensor x_t along the specified dimension.

    Args:
    - x_t: torch.Tensor, input tensor
    - shape: int, target shape along the specified dimension
    - dimension: int, dimension along which to adjust the shape

    Returns:
    - Adjusted tensor x_t
    """
    current_dim = x_t.size(dimension)
    if current_dim < shape:
        # Calculate the amount needed to pad x_t along the specified dimension
        pad_size = shape - current_dim

        # Initialize padding for all dimensions as zero
        padding = [0] * (2 * x_t.dim())

        # Adjust padding for the specified dimension (padding before and after the dimension)
        padding_index = 2 * (x_t.dim() - dimension - 1)
        padding[padding_index] = pad_size  # Pad after the dimension
        # padding[padding_index + 1] = pad_size  # Uncomment to pad both sides

        # Apply padding to the tensor
        x_t = torch.nn.functional.pad(x_t, padding)

    return x_t


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
        num_classes=0,  # set to 0 to do regression task
        base_channel=768,
        channel_ratio=4,
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
        self.stations = stations
        assert depth % 3 == 0

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(stations, 1, embed_dim))
        self.trans_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # Stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * 48), stations)

        # Stem stage: Get the feature maps by convolution block
        self.conv1 = nn.Conv2d(
            in_chans, base_channel, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channel)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 1st stage
        stage_1_channel = int(base_channel * channel_ratio)

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
        self.trans_1 = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=dropout,
            attn_drop=attention_dropout,
            drop_path=self.trans_dpr[0],
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
        # 5~8 stages
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    inplanes=in_channel,
                    outplanes=stage_2_channel,
                    res_conv=res_conv,
                    stride=s,
                    dw_stride=trans_dw_stride // 2,
                    embed_dim=embed_dim,
                    stations=stations,
                    past_timesteps=past_timesteps,
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
                    outplanes=stage_3_channel,
                    res_conv=res_conv,
                    stride=s,
                    dw_stride=trans_dw_stride // 4,
                    embed_dim=embed_dim,
                    stations=stations,
                    past_timesteps=past_timesteps,
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

    def forward(self, x, x_t=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (classification output) for each station.
        """
        # B = x.shape[0]
        # t = x.shape[1]
        # h, w = x.shape[2], x.shape[3]
        # features = x.shape[4]
        B, t, h, w, features = x.shape

        cls_tokens = self.cls_token.expand(-1, B, -1).transpose(
            0, 1
        )  # [batch_size, stations, features]

        # Stem stage
        x = x.view(B, features * t, h, w)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # x_base_t = self.maxpool(self.act1(self.bn1(self.conv1(x_t0))))

        # 1st stage
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        # 2nd to final stages
        for i in range(2, self.fin_stage):
            # Access the layer by name
            layer = getattr(self, f"conv_trans_{i}")
            # Call the layer with the current `x` and `x_t`
            x, x_t = layer(x, x_t)
            gc.collect()

        # Convolutional classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)
        gc.collect()

        # Transformer classification
        tran_cls_ls = []
        for s in np.arange(1, self.stations + 1):
            s = s + 1
            x_t = self.trans_norm(x_t)
            tran_cls = self.trans_cls_head(x_t[:, -s, :])
            tran_cls_ls.append(tran_cls[:, 0])
            gc.collect()
        tran_cls_ls = torch.stack(tran_cls_ls).permute(1, 0)
        tran_cls_ls = torch.flip(tran_cls_ls, [1])

        return conv_cls, tran_cls_ls
