from typing import Callable, Tuple
import warnings
import jax.numpy as jnp
from jax import grad
from jax_md import space, quantity, energy

EnergyFn = Callable[[space.Array, energy.NeighborList], space.Array]
PotentialFn = Callable[[space.Array], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, None, None]]
PotentialProperties = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]

def get_displacement(atoms):
    if not all(atoms.get_pbc()):
        displacement, _ = space.free()
        warnings.warn("Atoms object without periodic boundary conditions passed!")
        return displacement

    cell = atoms.get_cell().array
    inverse_cell = space.inverse(cell)
    displacement_in_scaled_coordinates, _ = space.periodic_general(cell)

    # **kwargs are now used to feed through the box information
    def displacement(Ra: space.Array, Rb: space.Array, **kwargs) -> space.Array:
        Ra_scaled = space.transform(inverse_cell, Ra)
        Rb_scaled = space.transform(inverse_cell, Rb)
        return displacement_in_scaled_coordinates(Ra_scaled, Rb_scaled, **kwargs)

    return displacement


def strained_lj_nl(energy_fn, neighbors, box: jnp.ndarray) -> PotentialFn:
    def potential(R: space.Array) -> PotentialProperties:
        # 1) Set the box under strain using a symmetrized deformation tensor
        # 2) Override the box in the energy function
        # 3) Derive forces, stress and stresses as gradients of the deformed energy function
        deformation = jnp.zeros_like(box)

        # a function to symmetrize the deformation tensor and apply it to the box
        transform_box_fn = lambda deformation: space.transform(jnp.eye(3) + (deformation + deformation.T) * 0.5, box) 
        
        # atomwise and total energy functions that act on the transformed box. same for force, stress and stresses.
        deformation_energy_fn = lambda deformation, R, *args, **kwargs: energy_fn(R, box=transform_box_fn(deformation), neighbor=neighbors)
        total_energy_fn = lambda deformation, R, *args, **kwargs: jnp.sum(deformation_energy_fn(deformation, R))            
        force_fn = lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=1)(deformation, R) * -1
        
        stress_fn = lambda deformation, R, *args, **kwargs: grad(total_energy_fn, argnums=0)(deformation, R) / jnp.linalg.det(box)
        stress = stress_fn(deformation, R, neighbor=neighbors)  
        
        total_energy = total_energy_fn(deformation, R, neighbor=neighbors)
        atomwise_energies = deformation_energy_fn(deformation, R, neighbor=neighbors)
        forces = force_fn(deformation, R, neighbor=neighbors)
        
        return total_energy, atomwise_energies, forces, stress

    return potential


def lj_nl(energy_fn, neighbors, box: jnp.ndarray):
    def potential(R: space.Array) -> PotentialProperties:
        total_energy_fn = lambda R, *args, **kwargs: jnp.sum(energy_fn(R, *args, **kwargs))
        forces_fn = quantity.force(total_energy_fn)

        total_energy = total_energy_fn(R, neighbor=neighbors)
        atomwise_energies = energy_fn(R, neighbor=neighbors)
        forces = forces_fn(R, neighbor=neighbors)
        return total_energy, atomwise_energies, forces, None

    return potential