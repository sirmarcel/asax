from ase import units, Atoms
from ase.md import VelocityVerlet
import asax
from ase.build import bulk
import time
from ase.calculators.lj import LennardJones as aseLJ

sigma = 2.0
epsilon = 1.5
rc = 10.0
ro = 6.0


def run_nve(steps: int, timestep):
    start = time.monotonic()

    atoms = bulk("Ar", cubic=True) * [5, 5, 5]
    atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)
    atoms.calc = asax.lj.LennardJones(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro)

    dyn = VelocityVerlet(atoms, timestep=timestep)
    dyn.run(steps)
    elapsed_seconds = round(time.monotonic() - start, 2)
    mean_step_time_ms = round((elapsed_seconds / steps) * 1000, 2)

    print("{} steps".format(steps))
    print("{} seconds total".format(elapsed_seconds))
    print("{} ms/step".format(mean_step_time_ms))
    return steps, elapsed_seconds, mean_step_time_ms


run_nve(10000, timestep=5.0 * units.fs)