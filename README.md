# 1️⃣🎷 asax

`ase + jax-md = asax`

`asax` exposes [`jax-md`](https://github.com/google/jax-md/) energy functions as [`ase`](https://gitlab.com/ase/ase) calculators.

## Facts

Needs `ase`, `jax-md`, `numpy`. `nose` to run tests. (`cd tests; nosetests`).

Tests currently fail unless `ase.calculators.lj` is patched to remove the energy shift. (Pending `ase` changes.)

Codestyle `black`, roughly Google style otherwise.

`poetry install` in this folder will install an editable version.

## Design

`jax-md` is fundamentally built around functional programming: In the end, you get a function that maps `positions -> energies`. Implicitly in this, the `positions` are mapped to displacements (otherwise, the energy function wouldn't be tranlationally invariant). This function is built up with the `setup()` method of the calculator sub-classes implemented here. Since it implicitly depends on the boundary conditions, this re-building needs to be repeated every time the unit cell changes. This logic is taken care of in `asax.Calculator`.

## Shortcomings

We only treat full periodic boundary conditions, i.e. 3D systems, or no boundary conditions.
