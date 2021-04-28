from .calculator import Calculator
import numpy as np
from asax import jax_utils
from jax import jit
import jax.numpy as jnp
from jax_md import energy, partition
from ase.constraints import full_3x3_to_voigt_6_stress


class LennardJones(Calculator):
    """Lennard-Jones Potential"""
    implemented_properties = ["energy", "forces"]

    _energy_fn: jax_utils.EnergyFn = None
    _neighbor_fn: energy.NeighborFn = None
    _neighbors: partition.NeighborList = None

    def __init__(self, stress=False, epsilon=1.0, sigma=1.0, rc=None, ro=None, **kwargs):
        """
        Paramters:
            sigma: The potential minimum is at  2**(1/6) * sigma, default 1.0
            epsilon: The potential depth, default 1.0
            rc: Cut-off for the NeighborList is set to 3 * sigma if None, the default.
            ro: Onset of the cutoff function. Set to 0.8*rc if None, the default.
            x64: Determine if double precision is used. Default to True.

        """
        super().__init__(**kwargs)
        self.stress = stress
        if self.stress:
            self.implemented_properties.append("stress")

        self.epsilon = epsilon
        self.sigma = sigma

        if rc is None:
            rc = 3 * self.sigma
        self.rc = rc

        if ro is None:
            ro = 0.8 * self.rc
        self.ro = ro

    def get_potential(self):
        # box as vanilla np.array causes strange indexing errors with neighbor lists now and then
        box = jnp.array(self.box)
        normalized_ro = self.ro / self.sigma
        normalized_rc = self.rc / self.sigma

        if self._neighbors is None:
            self._neighbor_fn, self._energy_fn = energy.lennard_jones_neighbor_list(self.displacement, box,
                                                                                    sigma=jnp.array(self.sigma),
                                                                                    epsilon=jnp.array(self.epsilon),
                                                                                    r_onset=normalized_ro,
                                                                                    r_cutoff=normalized_rc,
                                                                                    per_particle=True)
            self._neighbors = self._neighbor_fn(self.R)

        if self.stress:
            return jit(jax_utils.strained_lj_nl(self._energy_fn, self._neighbors, box))
        return jit(jax_utils.lj_nl(self._energy_fn, self._neighbors, box))

    def compute_properties(self):
        potential_energy, potential_energies, forces, stress = self.potential(self.R)
        self.results["energy"] = float(potential_energy.block_until_ready())
        self.results["energies"] = np.float64(potential_energies.block_until_ready())
        self.results["forces"] = np.float64(forces.block_until_ready())
        if stress:
            stress_dispatched = np.float64(stress.block_until_ready())
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress_dispatched)
