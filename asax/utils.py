def get_displacement(atoms):
    from jax_md import space

    if not all(atoms.get_pbc()):
        displacement_fn, _ = space.free()
    else:
        from ase.geometry.cell import crystal_structure_from_cell
        assert crystal_structure_from_cell(cell) == "cubic"

        cell = atoms.get_cell()

        box = cell.lengths()[0]
        displacement_fn, _ = space.periodic(side=box, wrapped=True)

    return displacement_fn


def get_potential(energy):
    from jax import value_and_grad, jit

    return jit(value_and_grad(energy))
