from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StardistConfig(BaseModel):
    
    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    
    decoders: tuple[Literal["stardist", "sem"], ...]
    """Names of the decoder branches of this network."""
    
    extra_convs: dict[str, dict[str, int]]
    """"The extra conv blocks before segmentation heads of the architecture.
    In the form: names of the decoder branches (has to match `decoders`) mapped to
    dicts of output name - number of output channels.
    Example: {"stardist": {"conv1": 64, "conv2": 32}, "sem": {"conv1": 64}}"""
    
    heads : dict[str, dict[str, int]]
    """Segmentation heads of the architecture, i.e., names of the decoder branches (has
    to match `decoders`) mapped to dicts of output name - number of output classes.
    Example: {"stardist": {"stardist": 2}, "sem": {"sem": 5}}"""
    
    # TODO: necessary ?
    inst_key : str = "dist"
    """The key for the model output that will be used in the instance
    segmentation post-processing pipeline as the binary segmentation result."""
    
    depth : int =  Field(ge=1, le=5)
    """The depth of the encoder, i.e., the number of returned feature maps from
    the encoder. Maximum depth is 5."""
    
    out_channels : tuple[int, ...] = (256, 128, 64, 32)
    """Out channels for each decoder stage."""
    
    style_channels : int = 256
    """Number of style vector channels. If None, style vectors are ignored."""
    
    enc_name : str = "resnet50"
    """Name of the encoder. See timm docs for more info."""
    
    # TODO: remove ?    
    enc_pretrain : bool = False
    """Whether to use imagenet pretrained weights in the encoder."""
    
    enc_freeze : bool = False
    """Freeze encoder weights for training."""
    
    enc_out_indices : Optional[tuple[int, ...]] = None
    """Indices of the encoder output features. If None, indices is set to
    `range(len(depth))`."""
    
    # TODO: necessary ?
    upsampling : Literal[
        "fixed-unpool", "nearest", "bilinear", "bicubic", "conv_transpose" 
    ] = "fixed-unpool"
    """The upsampling method."""
    
    # TODO: necessary ?
    long_skip : Optional[Literal["unet", "unetpp", "unet3p", "unet3p-lite"]] = "unet"
    """Long skip method to be used."""
    
    # TODO: necessary ?
    merge_policy : Literal["sum", "cat"] = "sum"
    """The long skip merge policy."""
    
    # TODO: necessary ?
    short_skip : Literal["residual", "dense", "basic"] = "basic"
    "The name of the short skip method."
    
    # TODO: necessary ?
    block_type : Literal["basic", "mbconv", "fmbconv" "dws", "bottleneck"] = "basic"
    """The type of the convolution block type."""
    
    # TODO: necessary ?
    normalization : Optional[Literal["bn", "bcn", "gn", "in", "ln"]] = "bn"
    """Normalization method."""
    
    # TODO: necessary ?
    activation : Literal[
        "mish", "swish", "relu", "relu6", "rrelu", "selu",
        "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
        "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
    ] = "relu"
    """Activation function."""
    
    # TODO: necessary ? 
    convolution : Literal["conv", "wsconv", "scaled_wsconv"] = "conv"
    """The convolution method."""
    
    # TODO: check this
    preactivate : bool = True
    """If True, normalization will be applied before convolution."""
    
    # TODO: check this
    attention : Optional[Literal["se", "scse", "gc", "eca"]] = None
    """Attention method."""
    
    # TODO: necessary ?
    preattend : bool = False
    """If True, Attention is applied at the beginning of forward pass."""
    
    # TODO: check this
    add_stem_skip : bool = False
    """If True, a stem conv block is added to the model whose output is used
    as a long skip input at the final decoder layer that is the highest
    resolution layer and the same resolution as the input image."""
    
    # TODO: necessary ?
    out_size : Optional[int] = None
    """If specified, the output size of the model will be (out_size, out_size).
    I.e. the outputs will be interpolated to this size."""
    
    # TODO: necessary ?
    skip_params : Optional[dict]
    """Extra keyword arguments for the skip-connection modules. These depend
    on the skip module. Refer to specific skip modules for more info, i.e.,
    `UnetSkip`, `UnetppSkip`, `Unet3pSkip`."""
    
    # TODO: necessary ?
    encoder_params : Optional[dict]
    """Extra keyword arguments for the encoder. These depend on the encoder.
    Refer to specific encoders for more info."""