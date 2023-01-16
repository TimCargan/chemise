import functools
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

Array = Any


class DynamicScaleResult(NamedTuple):
    dynamic_scale: 'DynamicScale'
    finite: Array
    aux: Any
    grad: Any


class DynamicScale(struct.PyTreeNode):
    """Dynamic loss scaling for mixed precision gradients.

  For many models gradient computations in float16 will result in numerical
  issues because small/large gradients being flushed to zero/infinity.
  Dynamic loss scaling is an algorithm that aims to find the largest scalar
  multiple for which the gradient does not overflow. This way the risk of
  underflow is minimized.

  the `value_and_grad` method mimicks `jax.value_and_grad`. Beside the loss
  and gradients it also ouputs and updated `DynamicScale` instance with the
  current loss scale factor. This method also returns a boolean value indicating
  whether the gradients are finite.

  Example::

    from flax.training.dynamic_scale import DynamicScale

    def loss_fn(p):
      return jnp.asarray(p, jnp.float16) ** 2
    p = jnp.array(1., jnp.float32)

    dyn_scale = DynamicScale(growth_interval=10)
    compute_grad = jax.jit(lambda ds, p: ds.value_and_grad(loss_fn)(p))
    for _ in range(100):
      dyn_scale, is_fin, loss, grad = compute_grad(dyn_scale, p)
      p += jnp.where(is_fin, 0.01 * grad, 0.)
      print(loss)

  Jax currently cannot execute conditionals efficiently on GPUs therefore we
  selectifly ignore the gradient update using `jax.numpy.where` in case of
  non-finite gradients.

  Attributes:
    growth_factor: how much to grow the scalar after a period of finite
      gradients (default: 2.).
    backoff_factor: how much to shrink the scalar after a non-finite gradient
      (default: 0.5).
    growth_interval: after how many steps of finite gradients the scale should
      be increased (default: 2000).
    fin_steps: indicates how many gradient steps in a row have been finite.
    scale: the current scale by which the loss is multiplied.
  """
    growth_factor: float = struct.field(pytree_node=False, default=2.0)
    backoff_factor: float = struct.field(pytree_node=False, default=0.5)
    growth_interval: int = struct.field(pytree_node=False, default=2000)
    fin_steps: Array = 0
    scale: Array = 65536.0

    def value_and_grad(self, fun: Callable[..., Any],
                       argnums: Union[int, Sequence[int]] = 0,
                       has_aux: bool = False,
                       axis_name: Optional[str] = None,
                       ) -> Callable[..., DynamicScaleResult]:
        """Wrapper around `jax.value_and_grad`.

    Args:
      fun: Function to be differentiated. Its arguments at positions specified
        by ``argnums`` should be arrays, scalars, or standard Python containers.
        It should return a scalar (which includes arrays with shape ``()``
        but not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where
        the first element is considered the output of the mathematical function
        to be differentiated and the second element is auxiliary data.
        Default False.
      axis_name: If an axis is given the gradients will be averaged across
        replicas (default: None).
    Returns:
      A function that takes the same arguments as `fun` and
      returns a DynamicScaleResult
    """

        @functools.wraps(fun)
        def loss_wrapper(*args):
            aux = fun(*args)
            if has_aux:
                return (self.scale * aux[0], aux[1])
            else:
                return self.scale * aux

        grad_fn = jax.value_and_grad(loss_wrapper, argnums, has_aux)

        def grad_fn_wrapper(*args):
            aux, grad = grad_fn(*args)
            aux = (aux[0] / self.scale, aux[1]) if has_aux else aux / self.scale

            grad = jax.tree_util.tree_map(
                lambda g: jnp.asarray(g, jnp.float32) / self.scale, grad)
            if axis_name is not None:
                grad = lax.pmean(grad, axis_name)

            finite = jnp.array(True)
            for g in jax.tree_util.tree_leaves(grad):
                finite &= jnp.all(lax.is_finite(g))

            grow = self.fin_steps == self.growth_interval
            fin_scale = jnp.where(
                grow & finite,
                jnp.minimum(self.scale * self.growth_factor, jnp.finfo(jnp.float32).max),
                self.scale)
            inf_scale = jnp.maximum(self.scale * self.backoff_factor, jnp.finfo(jnp.float32).min)
            new_scale = jnp.where(finite, fin_scale, inf_scale)
            new_fin_steps = jnp.where(grow | (~finite), 0, self.fin_steps + 1)

            new_self = self.replace(fin_steps=new_fin_steps, scale=new_scale)
            return DynamicScaleResult(new_self, finite, aux, grad)

        return grad_fn_wrapper
