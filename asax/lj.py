from .calculator import Calculator

from .utils import atoms_to_space


class LennardJones(Calculator):

    implemented_properties = [
        "energy",
        "forces",
        # "stress",
        # "stresses",
        # "energies",
    ]

    def __init__(self, epsilon=1.0, sigma=1.0, rc=None, ro=None, **kwargs):
        """
        Paramters:
            sigma: The potential minimum is at  2**(1/6) * sigma, default 1.0
            epsilon: The potential depth, default 1.0
            rc: Cut-off for the NeighborList is set to 3 * sigma if None. Default None
            ro: float, None
              Onset of the cutoff function. Set to 0.8*rc if None.
              Default None
            x64: bool, False
              Determine if double precision is used.

        """

        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.sigma = sigma

        if rc is None:
            rc = 3 * self.sigma
        self.rc = rc

        if ro is None:
            ro = 0.8 * self.rc
        self.ro = ro

    def setup(self):
        displacement, _, box = atoms_to_space(self.atoms)
        self.energy, self.forces = get_lj(
            displacement, self.epsilon, self.sigma, self.ro, self.rc, box, self.x64
        )


def get_lj(displacement, epsilon, sigma, ro, rc, box, x64):
    from jax_md import energy
    from jax_md.quantity import force
    from jax import jit
    from jax.config import config

    config.update("jax_enable_x64", x64)

    energy_fn = energy.lennard_jones_pair(
        displacement,
        sigma=sigma,
        epsilon=epsilon,
        r_onset=ro / sigma,
        r_cutoff=rc / sigma,
    )

    force_fn = force(energy_fn)

    return jit(energy_fn), jit(force_fn)
