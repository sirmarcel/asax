# 1️⃣🎷 asax
## `ase + jax-md = asax`

## Facts

Needs `ase`, `jax-md`, `numpy`. `nose` to run tests. (`cd tests; nosetests`).

Tests currently fail unless `ase.calculators.lj` is patched to remove the energy shift. (Pending `ase` changes.)

Codestyle `black`, roughly Google style otherwise.

`poetry install` in this folder will install an editable version.
