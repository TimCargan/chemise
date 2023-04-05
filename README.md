# Chemises 

> Chemises - a lightweight undergarment made of linen

This is a library of helper functions and basic module layers for uses with  the `flax.linen` library.
We aim to provide the standard boilerplate needed to train models quickly and easily. 
Whilst being highly extendable to fit more complex training pattens.

For now, we rely on `tensorlfow.data` as the input data.



## Set Up
It is recommended to install jax first, see their documentation [here](https://github.com/google/jax/#installation).

```shell
pip install chemise
```

Alternatively to install from source, clone the repo and install it. 

```shell
git clone <repo>
cd <repo>
pip install -e .
```

### Project Layout
- `/src` - all the lib code, everything needed to created and run the models, data pipelines etc.