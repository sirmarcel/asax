def get_displacement(atoms):
    from jax_md import space

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


def get_potential(energy):
    from jax import value_and_grad, jit

    return jit(value_and_grad(energy))
