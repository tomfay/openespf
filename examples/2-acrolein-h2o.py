import sys
sys.path.append('../pyespf_refactor/')
from pyscf import gto, scf, dft
from pyscf.data import radii
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
from openespf import QMMMSystem
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# set up the pyscf QM system
mol = gto.M('data2/mol0.xyz',unit="Angstrom",basis="def2-SVP",charge=0,verbose=4)
mf = dft.RKS(mol)
mf.xc = "PBE0"
mf.kernel()
dm = mf.make_rdm1()

# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("data2/droplet0.pdb")
modeller = Modeller(pdb.getTopology() , pdb.getPositions())
modeller.delete([r for r in modeller.topology.residues() if r.name == "UNL"])
positions = modeller.getPositions()
topology = modeller.getTopology()

forcefield = ForceField("data2/h2o.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("CPU")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)

start = timer()
U_MM = simulation.context.getState(getEnergy=True).getPotentialEnergy()
end = timer()
print("MM energy time:",end-start,"s")


# information about the QM-MM interaction
multipole_order = 1
multipole_method = "espf"
rep_type_info = [{"N_eff":6.0,"R_dens":1.71/radii.BOHR,"rho(R_dens)":1.0e-3},{"N_eff":1.0,"R_dens":1.54/radii.BOHR,"rho(R_dens)":1.0e-3}]
mm_rep_types = [0,1,1]*int(len(positions)/3)
rep_cutoff = 20.
qm_damp = [0.0001**(1./6.)]*len(mol.atom_charges())
qm_thole = [0.39]*len(mol.atom_charges())


qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method)
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
#qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)
qmmm_system.qm_system.dm_guess = dm 

# get positions
mm_positions = simulation.context.getState(getPositions=True).getPositions()
mm_unit = "nanometer"
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr"

# set QM/MM positions
qmmm_system.setPositions(qm_positions=qm_positions,qm_unit=qm_unit,mm_positions=mm_positions,mm_unit=mm_unit)

# get the QM/MM system energy
E_qmmm_0 = qmmm_system.getEnergy()
print("QM/MM Energy [AU]:")
print(E_qmmm_0)

