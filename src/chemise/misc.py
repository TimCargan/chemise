import functools
import operator
from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jaxtyping import Array, Float, Num


def gaussian_1d_kernel(size: int, sigma: int) -> Float[Array, "W"]:
    """
    Generate a 1d gaussian kernel
    :param size: size of kernel
    :param sigma:
    :return:
    """
    x = jnp.arange(-size // 2 + 1, size // 2 + 1)
    x = jnp.power(x, 2)
    gk = nn.softmax(-x / (2.0 * (sigma ** 2)))
    return gk


def gaussian_nd_kernel(size: Sequence[int], sigma: int) -> Float[Array, "..."]:
    """
    Generates an nd gaussian kernel of a given size
    :param size: Sequence, size of kernel (d0, d1 ..., dn)
    :param sigma:
    :return:
    """
    nd = len(size) - 1
    expand_dims = np.arange(nd) - nd
    gks = [jnp.expand_dims(gaussian_1d_kernel(s, sigma), axis=expand_dims + i) for i, s in enumerate(size)]
    gaussian_kernel = functools.reduce(operator.mul, gks)
    return gaussian_kernel

def gaussian_2d_kernel(size: Tuple[int, int], sigma: int) -> Float[Array, "H W"]:
    """
    Generates a 2d gaussian kernel of a given size
    :param size: tuple, size of kernel (H, W)
    :param sigma:
    :return:
    """
    return gaussian_nd_kernel(size, sigma)


def gaussian_blur(img: Num[Array, "... H W C"], kernel_size: tuple[int, int] = (7, 7), sigma=2) -> Num[Array, "... H W C"]:
    """
    Apply a gaussian blur
    :param img: ndarray of rank 3+ to blur
    :param kernel_size: tuple, size of kernel (H, W)
    :param sigma:
    :return:
    """
    gk = gaussian_2d_kernel(kernel_size, sigma=sigma)
    gk = jnp.expand_dims(gk, axis=[0, 1])
    shape = img.shape
    gk = jnp.tile(gk, (shape[-1], shape[-1], 1, 1))  # rhs = OIHW conv kernel tensor

    img = jnp.transpose(img, [0, 3, 1, 2])  # lhs = NCHW image tensor
    blur = jax.lax.conv(img, gk, window_strides=(1, 1),  padding='SAME')
    res = jnp.transpose(blur, [0, 2, 3, 1])
    return res


@functools.partial(jax.vmap, in_axes=(3, None, None), out_axes=(3))
def v_gaussian_blur(img: Num[Array, "... H W C"], kernel_size: tuple[int, int] = (7, 7), sigma=2):
    return gaussian_blur(jnp.expand_dims(img, axis=-1), kernel_size, sigma)[..., 0]


def nd_tile(x: Num[Array, "... N"], /, size:Sequence[int]) -> Num[Array, "... N"]:
    """
    add extra N dims of a given size tiling the given value
     e.g. to make a square out of a scaler
    :param x: [..., N] data to tile
    :param size: Size of tiled dims, (d0, d1, ..., dn)
    :return: x as [..., d0, d1, ..., dn, N]
    """
    no_tile = [1] * len(x.shape[:-1])  # All but the last dim shouldn't be tiled
    extra_axis = np.arange(len(size)) - len(size)
    x = jnp.expand_dims(x, axis=extra_axis)
    x = jnp.tile(x, (*no_tile, *size, 1))
    return x