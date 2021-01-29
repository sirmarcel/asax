from jax_md import space
import jax.numpy as np
from jax import value_and_grad, jit


def get_displacement(atoms):

    if not all(atoms.get_pbc()):
        displacement, _ = space.free()
    else:
        cell = atoms.get_cell().array
        inverse = space._small_inverse(cell)

        displacement_in_scaled_coordinates, _ = space.periodic_general(cell)

        # periodic_general works in scaled coordinates, but ase works in real space,
        # so we define a custom displacement that maps automatically by multiplying
        # the inverse unit cell matrix with the positions
        def displacement(
            Ra: space.Array, Rb: space.Array, **unused_kwargs
        ) -> space.Array:
            """Displacement that maps from real-space into scaled coordinates"""
            return displacement_in_scaled_coordinates(
                space.transform(inverse, Ra), space.transform(inverse, Rb)
            )

    return displacement


def get_potential(displacement, get_energy):
    energy = get_energy(displacement)
    return jit(value_and_grad(energy))


def get_potential_with_stress(displacement, get_energy):

    ones = np.eye(N=3, M=3, dtype=np.double)

    def energy_under_strain(R: space.Array, strain: space.Array) -> space.Array:
        def displacement_under_strain(
            Ra: space.Array, Rb: space.Array, **unused_kwargs
        ) -> space.Array:
            transform = ones + strain
            return _transform(transform, displacement(Ra, Rb))

        energy = get_energy(displacement_under_strain)

        return energy(R)

    def potential(R: space.Array) -> space.Array:
        zeros = np.zeros((3, 3), dtype=np.double)

        return value_and_grad(energy_under_strain, argnums=(0, 1))(R, zeros)

    return jit(potential)


def _transform(T: space.Array, v: space.Array) -> space.Array:
    """Apply a linear transformation, T, to a collection of vectors, v.
    Transform is written such that it acts as the identity during gradient
    backpropagation.
    Args:
      T: Transformation; ndarray(shape=[spatial_dim, spatial_dim]).
      v: Collection of vectors; ndarray(shape=[..., spatial_dim]).
    Returns:
      Transformed vectors; ndarray(shape=[..., spatial_dim]).
    """
    space._check_transform_shapes(T, v)
    return np.dot(v, T)
