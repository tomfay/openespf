# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
A simple example calculating the energy of a QM Ne ion with 2 polarisable MM He atoms:
Ne  HeHe
The separation between Ne and the He dimer is scanned. The He atoms are positioned at (0,-2,0)A and (0,+2,0) A respectively.
The Ne atom is scanned along (R,0,0)A for -10 < R < 10
'''

# import pyscf for setting up the QM part of the calculation
from pyscf import gto, scf, dft
# import OpenMM for setting up the MM part of the calculation
from openmm.app import *
from openmm import *
from openmm.unit import *
# import the QMMMSystem object from the OpenESPF paackage
from openespf import QMMMSystem
import openespf.Data as Data

#import numpy and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt

# set up the Pyscf QM system
mol = gto.M(atom='Ne 0 0 0',unit="Bohr",basis="pc-1",charge=0)
# The DFT method is chosen to be a long-range-corrected ωPBEh functional
mf = dft.RKS(mol)
mf.xc = "HYB_GGA_XC_LRC_WPBEH"

# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("2-inputs/he2.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("2-inputs/he.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("Reference")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)

# information about the QM-MM interaction
multipole_order = 1 # 0=charges, 1=charges+dipoles for QM ESPF multipole operators
multipole_method = "espf" # "espf" or "mulliken" type multipole operators
# information for the exchange-repulsion model (atomic units)
rep_type_info = [{"N_eff":2.0,"R_dens":1.34*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
mm_rep_types = [0,0]
rep_cutoff = 20.
# information about how the QM multipole - MM induced dipole interactions are damped (OpenMM units)
qm_damp = [0.0001**(1./6.)]*len(mol.atom_charges())
qm_thole = [0.39]*len(mol.atom_charges())

# create the QMMMSystem object that performs the QM/MM energy calculations
qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method)
# set additional parameters for the exchange repulsion + damping of electrostatics
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)
qmmm_system.mm_system.use_prelim_mpole = False

# get positions for the QM and MM atoms
mm_positions = simulation.context.getState(getPositions=True).getPositions()
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr


# get the reference QM and MM energies
mf.kernel()
E_qm = mf.energy_tot() # in hartree
E_mm = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value * Data.KJMOL_TO_HARTREE # in hartree
E_qmmm_0 = E_qm + E_mm

# get density matrix from calculation
dm = mf.make_rdm1()
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm

# set up a grid of separations in atomic units
R_vals = np.linspace(10.,-10.0,num=100) * Data.ANGSTROM_TO_BOHR
energies = np.zeros(R_vals.shape)
for n,R in enumerate(R_vals):
    # set the MM atom positions
    qm_positions = np.array([[R,0,0]])
    qm_unit = "Bohr"
    qmmm_system.setPositions(qm_positions=qm_positions,qm_unit=qm_unit)
    # get the energy
    E_qmmm = qmmm_system.getEnergy()
    # get density matrix
    dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.qm_system.dm_guess = dm
    # save enegy
    energies[n] =  E_qmmm + 0

print("Interaction energies [Hartree]:")
print(energies-E_qmmm_0)
#np.savetxt("./2-inputs/int-energies-prelim.dat",np.vstack((R_vals,energies-E_qmmm_0)).T,delimiter=',',header='Separation [Bohr],Interaction energy [Hartree]')

# plot energies 
plt.plot(R_vals*Data.BOHR_TO_ANGSTROM,(energies-E_qmmm_0)*1e3,label="QM/MM ESPF-DRF")
plt.xlabel("Ne $x$ coordinate [Angstrom]")
plt.ylabel("Interaction energy [mH]")
plt.legend()

plt.show()