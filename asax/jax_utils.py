from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import grad
from jax_md import space, energy, quantity
from jax_md.energy import NeighborList
from jax_md.quantity import EnergyFn
from jaxlib.xla_extension import DeviceArray

PotentialFn = Callable[
    [space.Array], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, None, None]
]
PotentialProperties = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def strained_neighbor_list_potential(
    energy_fn: EnergyFn, box: jnp.ndarray, dtype: str
) -> PotentialFn:
    deformation = jnp.zeros_like(box, dtype=dtype)

    # 1) Set the box under strain using a symmetrized deformation tensor
    # 2) Override the box in the energy function
    # 3) Derive forces, stress and stresses as gradients of the deformed energy function

    def potential(
        R: space.Array, neighbor: NeighborList, *args, **kwargs
    ) -> PotentialProperties:
        # a function to symmetrize the deformation tensor and apply it to the box
        transform_box_fn = lambda deformation: space.transform(
            jnp.eye(3, dtype=dtype) + (deformation + deformation.T) * 0.5, box
        )

        # atomwise and total energy functions that act on the transformed box.
        strained_energy_fn = (
            lambda R, deformation, neighbor, *args, **kwargs: energy_fn(
                R, *args, **kwargs, box=transform_box_fn(deformation), neighbor=neighbor
            )
        )

        total_strained_energy_fn = (
            lambda R, deformation, neighbor, *args, **kwargs: jnp.sum(
                strained_energy_fn(R, deformation, *args, **kwargs, neighbor=neighbor)
            )
        )

        # same for force ...
        force_fn = (
            lambda R, deformation, neighbor, *args, **kwargs: grad(
                total_strained_energy_fn, argnums=0
            )(R, deformation, *args, **kwargs, neighbor=neighbor)
            * -1
        )

        # ... and stress
        box_volume = jnp.linalg.det(box)
        stress_fn = (
            lambda R, deformation, neighbor, *args, **kwargs: grad(
                total_strained_energy_fn, argnums=1
            )(R, deformation, neighbor, *args, **kwargs)
            / box_volume
        )

        total_energy = total_strained_energy_fn(
            R, deformation, neighbor, *args, **kwargs
        )
        atomwise_energies = strained_energy_fn(
            R, deformation, neighbor, *args, **kwargs
        )
        forces = force_fn(R, deformation, neighbor, *args, **kwargs)
        stress = stress_fn(R, deformation, neighbor, *args, **kwargs)

        return total_energy, atomwise_energies, forces, stress

    return potential


def unstrained_neighbor_list_potential(energy_fn: EnergyFn) -> PotentialFn:
    def potential(
        R: space.Array, neighbor: NeighborList, *args, **kwargs
    ) -> PotentialProperties:
        # this simply wraps the passed energy_fn to achieve a single consistent signature for all following lambdas.
        atomwise_energy_fn = lambda R, neighbor, *args, **kwargs: energy_fn(
            R, *args, **kwargs, neighbor=neighbor
        )

        total_energy_fn = lambda R, neighbor, *args, **kwargs: jnp.sum(
            energy_fn(R, *args, **kwargs, neighbor=neighbor)
        )
        force_fn = quantity.force(total_energy_fn)

        atomwise_energies = atomwise_energy_fn(R, neighbor, *args, **kwargs)
        total_energy = total_energy_fn(R, neighbor, *args, **kwargs)
        forces = force_fn(R, neighbor, *args, **kwargs)
        return total_energy, atomwise_energies, forces, None

    return potential


def block_and_dispatch(properties: Tuple[DeviceArray, ...]):
    for p in properties:
        if p is None:
            continue

        p.block_until_ready()

    return [None if p is None else np.array(p) for p in properties]
