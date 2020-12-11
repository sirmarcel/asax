def atoms_to_space(atoms):
    from jax_md import space

    if not all(atoms.get_pbc()):
        displacement_fn, shift_fn = space.free()
        box = None
    else:
        from ase.geometry.cell import crystal_structure_from_cell

        cell = atoms.get_cell()

        assert crystal_structure_from_cell(cell) == "cubic"
        box = cell.lengths()[0]
        displacement_fn, shift_fn = space.periodic(side=box, wrapped=True)

    return displacement_fn, shift_fn, box
