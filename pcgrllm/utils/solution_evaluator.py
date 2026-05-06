import jax.numpy as jnp

def create_fixed_size_onehot(imp_tiles, size=3, value_range=(4, 6)):
    onehot_array = jnp.zeros(size, dtype=jnp.int32)
    imp_indices = imp_tiles - value_range[0]
    valid_mask = (imp_indices >= 0) & (imp_indices < size)
    valid_indices = jnp.where(valid_mask, imp_indices, -1)
    onehot_array = onehot_array.at[valid_indices].set(1)
    return onehot_array
