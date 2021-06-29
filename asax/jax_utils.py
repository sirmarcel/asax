from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import grad
from jax_md import space, energy, quantity
from jax_md.energy import NeighborList
from jaxlib.xla_extension import DeviceArray

EnergyFn = Callable[[space.Array, energy.NeighborList], space.Array]
PotentialFn = Callable[
    [space.Array], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, None, None]
]
PotentialProperties = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def strained_neighbor_list_potential(energy_fn, box: jnp.ndarray) -> PotentialFn:

    def potential(R: space.Array, neighbor: NeighborList) -> PotentialProperties:
        # 1) Set the box under strain using a symmetrized deformation tensor
        # 2) Override the box in the energy function
        # 3) Derive forces, stress and stresses as gradients of the deformed energy function
        deformation = jnp.zeros_like(box)

        # a function to symmetrize the deformation tensor and apply it to the box
        transform_box_fn = lambda deformation: space.transform(
            jnp.eye(3) + (deformation + deformation.T) * 0.5, box
        )

        # atomwise and total energy functions that act on the transformed box. same for force, stress and stresses.
        deformation_energy_fn = lambda deformation, R, *args, **kwargs: energy_fn(
            R, box=transform_box_fn(deformation), neighbor=neighbor
        )
        total_energy_fn = lambda deformation, R, *args, **kwargs: jnp.sum(
            deformation_energy_fn(deformation, R)
        )
        force_fn = (
            lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=1)(
                deformation, R
            )
            * -1
        )

        stress_fn = lambda deformation, R, *args, **kwargs: grad(
            total_energy_fn, argnums=0
        )(deformation, R) / jnp.linalg.det(box)
        stress = stress_fn(deformation, R, neighbor=neighbor)

        total_energy = total_energy_fn(deformation, R, neighbor=neighbor)
        atomwise_energies = deformation_energy_fn(deformation, R, neighbor=neighbor)
        forces = force_fn(deformation, R, neighbor=neighbor)

        return total_energy, atomwise_energies, forces, stress

    return potential


def unstrained_neighbor_list_potential(energy_fn) -> PotentialFn:

    def potential(R: space.Array, neighbor: NeighborList) -> PotentialProperties:
        total_energy_fn = lambda R, *args, **kwargs: jnp.sum(
            energy_fn(R, *args, **kwargs)
        )
        forces_fn = quantity.force(total_energy_fn)

        total_energy = total_energy_fn(R, neighbor=neighbor)
        atomwise_energies = energy_fn(R, neighbor=neighbor)
        forces = forces_fn(R, neighbor=neighbor)
        return total_energy, atomwise_energies, forces, None

    return potential


def block_and_dispatch(properties: Tuple[DeviceArray, ...]):
    for p in properties:
        if p is None:
            continue

        p.block_until_ready()

    return [None if p is None else np.array(p) for p in properties]