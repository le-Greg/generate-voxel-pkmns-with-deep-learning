# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def upfirdn3d(input, kernel, up=1, down=1, pad=(0, 0)):
    # In the original LSGM there is a cuda implementation, but i did not convert it to handle 3D data
    # I don't know what the speed/memory impact is
    return upfirdn3d_native(
            input, kernel, up, up, up, down, down, down, pad[0], pad[1], pad[0], pad[1], pad[0], pad[1]
        )


def upfirdn3d_native(
    input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1
):
    _, channel, in_h, in_w, in_d = input.shape
    input = input.reshape(-1, in_h, in_w, in_d, 1)

    _, in_h, in_w, in_d, minor = input.shape
    kernel_h, kernel_w, kernel_d = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, in_d, 1, minor)
    out = F.pad(out, [0, 0, 0, up_z - 1, 0, 0, 0, up_y - 1, 0, 0, 0, up_x - 1])
    out = out.view(-1, in_h * up_x, in_w * up_y, in_d * up_z, minor)

    out = F.pad(
        out, [0, 0, max(pad_z0, 0), max(pad_z1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_x0, 0), max(pad_x1, 0)]
    )
    out = out[
          :,
          max(-pad_x0, 0): out.shape[1] - max(-pad_x1, 0),
          max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
          max(-pad_z0, 0): out.shape[3] - max(-pad_z1, 0),
          :,
          ]

    out = out.permute(0, 4, 1, 2, 3)
    out = out.reshape(
        [-1, 1, in_h * up_x + pad_x0 + pad_x1, in_w * up_y + pad_y0 + pad_y1, in_d * up_z + pad_z0 + pad_z1]
    )
    w = torch.flip(kernel, [0, 1, 2]).view(1, 1, kernel_h, kernel_w, kernel_d)
    out = F.conv3d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_x + pad_x0 + pad_x1 - kernel_h + 1,
        in_w * up_y + pad_y0 + pad_y1 - kernel_w + 1,
        in_d * up_z + pad_z0 + pad_z1 - kernel_d + 1,
    )
    out = out.permute(0, 2, 3, 4, 1)
    out = out[:, ::down_x, ::down_y, ::down_z, :]

    out_h = (in_h * up_x + pad_x0 + pad_x1 - kernel_h) // down_x + 1
    out_w = (in_w * up_y + pad_y0 + pad_y1 - kernel_w) // down_y + 1
    out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d) // down_z + 1

    return out.view(-1, channel, out_h, out_w, out_d)


# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
    """Get/create weight tensor for a convolution or fully-connected layer."""

    return module.param(weight_var, kernel_init, shape)


class Conv3d(nn.Module):
    """Conv3d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def __init__(self, in_ch, out_ch, kernel, up=False, down=False,
                 resample_kernel=(1, 3, 3, 1),
                 use_bias=True,
                 kernel_init=None):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_3d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_3d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv3d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1, 1)

        return x


def naive_upsample_3d(x, factor=2):
    _N, C, H, W, D = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1, D, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor, D * factor))


def naive_downsample_3d(x, factor=2):
    _N, C, H, W, D = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor, D // factor, factor))
    return torch.mean(x, dim=(3, 5, 7))


def upsample_conv_3d(x, w, k=None, factor=2, gain=1):
    """Fused `upsample_2d()` followed by `tf.nn.conv2d()`.

       Padding is performed only once at the beginning, not between the
       operations.
       The fused op is considerably more efficient than performing the same
       calculation
       using standard TensorFlow ops. It supports gradients of arbitrary order.
       Args:
         x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
           C]`.
         w:            Weight tensor of the shape `[filterH, filterW, inChannels,
           outChannels]`. Grouped convolution can be performed by `inChannels =
           x.shape[0] // numGroups`.
         k:            FIR filter of the shape `[firH, firW]` or `[firN]`
           (separable). The default is `[1] * factor`, which corresponds to
           nearest-neighbor upsampling.
         factor:       Integer upsampling factor (default: 2).
         gain:         Scaling factor for signal magnitude (default: 1.0).

       Returns:
         Tensor of the shape `[N, C, H * factor, W * factor]` or
         `[N, H * factor, W * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert len(w.shape) == 5, w.shape
    outC, inC, convH, convW, convD = w.shape

    assert convW == convH == convD

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 3))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor, factor)

    # Determine data dimensions.
    # stride = [1, 1, factor, factor]
    output_shape = ((_shape(x, 2) - 1) * factor + convH,
                    (_shape(x, 3) - 1) * factor + convW,
                    (_shape(x, 4) - 1) * factor + convD)
    output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                      output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW,
                      output_shape[2] - (_shape(x, 4) - 1) * stride[2] - convD)
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    # Transpose weights.
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW, convD))
    w = w.flip([-1, -2, -3]).permute(0, 2, 1, 3, 4, 5)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW, convD))

    x = F.conv_transpose3d(x, w, stride=stride, output_padding=output_padding, padding=0)
    ## Original TF code.
    # x = tf.nn.conv2d_transpose(
    #     x,
    #     w,
    #     output_shape=output_shape,
    #     strides=stride,
    #     padding='VALID',
    #     data_format=data_format)
    ## JAX equivalent

    return upfirdn3d(x, torch.tensor(k, device=x.device),
                     pad=((p + 1) // 2 + factor - 1, p // 2 + 1))


def conv_downsample_3d(x, w, k=None, factor=2, gain=1):
    """Fused `tf.nn.conv2d()` followed by `downsample_2d()`.

      Padding is performed only once at the beginning, not between the operations.
      The fused op is considerably more efficient than performing the same
      calculation
      using standard TensorFlow ops. It supports gradients of arbitrary order.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          w:            Weight tensor of the shape `[filterH, filterW, inChannels,
            outChannels]`. Grouped convolution can be performed by `inChannels =
            x.shape[0] // numGroups`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).

      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor]` or
          `[N, H // factor, W // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW, convD = w.shape
    assert convW == convH == convD
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = (factor, factor, factor)
    x = upfirdn3d(x, torch.tensor(k, device=x.device),
                  pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.einsum('i,j,k', k, k, k)
    k /= np.sum(k)
    assert k.ndim == 3
    assert k.shape[0] == k.shape[1] == k.shape[2]
    return k


def _shape(x, dim):
    return x.shape[dim]


def upsample_3d(x, k=None, factor=2, gain=1):
    r"""Upsample a batch of 2D images with the given filter.

      Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
      and upsamples each image with the given filter. The filter is normalized so
      that
      if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with
      zeros so that its shape is a multiple of the upsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            nearest-neighbor upsampling.
          factor:       Integer upsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).

      Returns:
          Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 3))
    p = k.shape[0] - factor
    return upfirdn3d(x, torch.tensor(k, device=x.device),
                     up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_3d(x, k=None, factor=2, gain=1):
    r"""Downsample a batch of 2D images with the given filter.

      Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
      and downsamples each image with the given filter. The filter is normalized
      so that
      if the input pixels are constant, they will be scaled by the specified
      `gain`.
      Pixels outside the image are assumed to be zero, and the filter is padded
      with
      zeros so that its shape is a multiple of the downsampling factor.
      Args:
          x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
          k:            FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to
            average pooling.
          factor:       Integer downsampling factor (default: 2).
          gain:         Scaling factor for signal magnitude (default: 1.0).

      Returns:
          Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn3d(x, torch.tensor(k, device=x.device),
                     down=factor, pad=((p + 1) // 2, p // 2))
