from typing import Callable, Tuple, Any, Union, List
import warnings
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones as aLJ
from jax import grad
from jax_md import space, energy, quantity, util
from jax_md.energy import DisplacementFn
from jax_md.simulate import NVEState
from jax_md.space import ShiftFn
from jaxlib.xla_extension import DeviceArray
from numpy import ndarray

EnergyFn = Callable[[space.Array, energy.NeighborList], space.Array]
PotentialFn = Callable[[space.Array], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, None, None]]
PotentialProperties = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def strained_neighbor_list_potential(energy_fn, neighbors, box: jnp.ndarray) -> PotentialFn:
    def potential(R: space.Array) -> PotentialProperties:
        # 1) Set the box under strain using a symmetrized deformation tensor
        # 2) Override the box in the energy function
        # 3) Derive forces, stress and stresses as gradients of the deformed energy function
        deformation = jnp.zeros_like(box)

        # a function to symmetrize the deformation tensor and apply it to the box
        transform_box_fn = lambda deformation: space.transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, box)

        # atomwise and total energy functions that act on the transformed box. same for force, stress and stresses.
        deformation_energy_fn = lambda deformation, R, *args, **kwargs: energy_fn(R, box=transform_box_fn(deformation),
                                                                                  neighbor=neighbors)
        total_energy_fn = lambda deformation, R, *args, **kwargs: jnp.sum(deformation_energy_fn(deformation, R))
        force_fn = lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=1)(deformation, R) * -1

        stress_fn = lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=0)(deformation,
                                                                                             R) / jnp.linalg.det(box)
        stress = stress_fn(deformation, R, neighbor=neighbors)

        total_energy = total_energy_fn(deformation, R, neighbor=neighbors)
        atomwise_energies = deformation_energy_fn(deformation, R, neighbor=neighbors)
        forces = force_fn(deformation, R, neighbor=neighbors)

        return total_energy, atomwise_energies, forces, stress

    return potential


def unstrained_neighbor_list_potential(energy_fn, neighbors) -> PotentialFn:
    def potential(R: space.Array) -> PotentialProperties:
        total_energy_fn = lambda R, *args, **kwargs: jnp.sum(energy_fn(R, *args, **kwargs))
        forces_fn = quantity.force(total_energy_fn)

        total_energy = total_energy_fn(R, neighbor=neighbors)
        atomwise_energies = energy_fn(R, neighbor=neighbors)
        forces = forces_fn(R, neighbor=neighbors)
        stress, stresses = None, None
        return total_energy, atomwise_energies, forces, stress

    return potential


def get_initial_nve_state(atoms: Atoms) -> NVEState:
    R = atoms.get_positions()
    V = atoms.get_velocities()  # å/ ase fs
    forces = atoms.get_forces()
    masses = atoms.get_masses()[0]
    return NVEState(R, V, forces, masses)


def block_and_dispatch(properties: Tuple[DeviceArray, ...]):
    for p in properties:
        if p is None:
            continue

        p.block_until_ready()

    return [None if p is None else np.array(p) for p in properties]


def initialize_cubic_argon(multiplier: List[int] = 5, temperature_K: int = 30):
    atoms = bulk("Ar", cubic=True) * [multiplier, multiplier, multiplier]
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    
    atoms.calc = aLJ(sigma=2.0, epsilon=1.5, rc=10.0, ro=6.0)  # TODO: Remove later
    return atoms
