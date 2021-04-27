from jax import jit
from asax import jax_utils
from .calculator import Calculator


class LennardJones(Calculator):
    """Lennard-Jones Potential"""
    implemented_properties = ["energy", "forces"]

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
        if self.stress: self.implemented_properties.append("stress")

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

        if self.neighbors is None:
            self.neighbor_fn, self.energy_fn = energy.lennard_jones_neighbor_list(self.displacement, box, sigma=self.sigma, epsilon=self.epsilon, r_onset=normalized_ro, r_cutoff=normalized_rc, per_particle=True)
            self.neighbors = self.neighbor_fn(self.R)

        if self.stress:
            return jit(jax_utils.strained_lj_nl(self.energy_fn, self.neighbors, box))
        return jit(jax_utils.lj_nl(self.displacement, self.sigma, self.epsilon, normalized_ro, normalized_rc))

    
    def compute_properties(self):
        energy, energies, forces, stress = self.potential()

        self.results["energy"] = energy
        # TODO: energies
        self.results["forces"] = forces
        if stress: self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
