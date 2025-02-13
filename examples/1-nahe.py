import sys
sys.path.append('../pyespf_refactor/')
from pyscf import gto, scf, dft
from pyscf.data import radii
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
from QMMMSystem import QMMMSystem
import matplotlib.pyplot as plt


# set up the pyscf QM system
mol = gto.M(atom='Na 0 0 0',unit="Bohr",basis="6-31G",charge=1)
mf = dft.RKS(mol)
mf.xc = "HF"

# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("data1/he.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("data1/he.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("CPU")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)


# information about the QM-MM interaction
multipole_order = 1
multipole_method = "espf"
rep_type_info = [{"N_eff":2.0,"R_dens":1.34/radii.BOHR,"rho(R_dens)":1.0e-3}]
mm_rep_types = [0]
rep_cutoff = 20.
qm_damp = [0.0001**(1./6.)]*len(mol.atom_charges())
qm_thole = [0.39]*len(mol.atom_charges())


qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method)
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)

# get positions
mm_positions = simulation.context.getState(getPositions=True).getPositions()
mm_unit = "nanometer"
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr"

mm_positions = [Vec3(1e2,0,0)]*nanometer
# set QM/MM positions
qmmm_system.setPositions(qm_positions=qm_positions,qm_unit=qm_unit,mm_positions=mm_positions,mm_unit=mm_unit)

# get the QM/MM system energy
E_qmmm_0 = qmmm_system.getEnergy()
print("QM/MM Energy [AU]:")
print(E_qmmm_0)

#exit()

R_vals = np.linspace(6.,2.0,num=50) / radii.BOHR
energies = np.zeros(R_vals.shape)
for n,R in enumerate(R_vals):
    mm_positions = np.array([[R,0,0]])
    mm_unit = "Bohr"
    qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit)
    E_qmmm = qmmm_system.getEnergy()
    dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.qm_system.dm_guess = dm
    energies[n] = 1.0 * E_qmmm

print(1.2/(R_vals**4))
plt.plot(R_vals*radii.BOHR,(energies-E_qmmm_0)*1e3,label="QM/MM ESPF-DRF")
plt.plot(R_vals*radii.BOHR,(-0.5*1.2/(R_vals**4))*1e3,label="-α/2R^4")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Interaction energy [mH]")
plt.legend()

#plt.ylim(-0.2,0.6)
plt.show()