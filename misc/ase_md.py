from ase import units
from ase.md import VelocityVerlet
from ase.build import bulk
import time
from ase.calculators.lj import LennardJones as aseLJ

sigma = 2.0
epsilon = 1.5
rc = 10.0
ro = 6.0

atoms = bulk("Ar", cubic=True) * [5, 5, 5]
atoms.set_cell(1.05 * atoms.get_cell(), scale_atoms=True)
atoms.calc = aseLJ(epsilon=epsilon, sigma=sigma, rc=rc, ro=ro, smooth=True)

steps = 1000
dyn = VelocityVerlet(atoms, dt=5.0 * units.fs)

start = time.monotonic()
dyn.run(steps)
elapsed = time.monotonic() - start

mean_step_time = elapsed / steps
print("Simulation took {} seconds total".format(elapsed))
print("{} seconds/step on average".format(mean_step_time))

