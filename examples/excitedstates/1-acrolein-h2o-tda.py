# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of acrolein and a singlet H2O molecule
CH2=CH-CH=O H-O-H
The separation between the C=O O atom and H of H2O is scanned. 
Different excited states are calculated with TDA-TDDFT
'''

# import pyscf for setting up the QM part of the calculation
from pyscf import gto, scf, dft, tdscf
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


# set up the Pyscf QM system u
mol = gto.M(atom="1-inputs/acrolein.xyz",unit="Angstrom",basis="pc-1",charge=0)
# The DFT method is chosen to be a long-range-corrected functional with density fitting
mf = dft.RKS(mol)
mf.xc = "PBE0"
#mf = mf.density_fit(auxbasis="weigendjkfit")
mf.kernel()
resp = tdscf.TDA(mf)
resp.nstates = 2


# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("1-inputs/h2o.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("3-inputs/h2o.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("OpenCL")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)

# information about the QM-MM interaction
multipole_order = 1 # 0=charges, 1=charges+dipoles for QM ESPF multipole operators
multipole_method = "espf" # "espf" or "mulliken" type multipole operators
# information for the exchange-repulsion model (atomic units)
rep_type_info = [{"N_eff":4.0+0.669,"R_dens":1.71*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0-0.3345,"R_dens":1.54*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
mm_rep_types = [0,1,1]
rep_cutoff = 20.
# information about how the QM multipole - MM induced dipole interactions are damped (OpenMM units)
qm_damp = [0.001**(1./6.)]*len(mol.atom_charges())
qm_thole = [0.39]*len(mol.atom_charges())
#qm_thole = [2.13]*len(mol.atom_charges())

# create the QMMMSystem object that performs the QM/MM energy calculations
qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method,qm_resp=resp)
# set additional parameters for the exchange repulsion + damping of electrostatics
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)

# get positions for the QM and MM atoms
mm_positions_ref = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr

# centre the O atom in acrolein
qm_positions -= qm_positions[3,:]
qmmm_system.setPositions(qm_positions=qm_positions,qm_unit=qm_unit)

# get the reference QM and MM energies

E_qm = mf.energy_tot() # in hartree
E_mm = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value * Data.KJMOL_TO_HARTREE # in hartree
E_qmmm_0 = E_qm + E_mm

# get density matrix from calculation
dm = mf.make_rdm1()
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm

# set up a grid of separations in nanometres units
R_vals = np.linspace(5.,1.5,num=20) * 0.1
energies = []
for n,R in enumerate(R_vals):
    # set the MM atom positions
    mm_positions = np.array([[R,0,0]])+mm_positions_ref
    mm_unit = "nanometer"
    qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit)
    # get the energy
    E_qmmm = qmmm_system.getEnergy()
    # get density matrix
    dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.qm_system.dm_guess = dm
    # save enegy
    energies.append(E_qmmm)


energies = np.array(energies)
nstates = energies.shape[1]
# plot energies 
for n in range(0,nstates):
    plt.plot(R_vals*1.0e1,(energies[:,n]-E_qmmm_0)*1e3,label="QM/MM ESPF-DRF state "+str(n))
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Electronic state energies [mH]")
plt.legend()

plt.show()

resp.__init__(mf)
resp.kernel()
resp.analyze()
E_exc = resp.e

E_refs = np.zeros((nstates))
E_refs[0] = E_qmmm_0
E_refs[1:] = E_qmmm_0+E_exc
# plot energies 
for n in range(0,nstates):
    plt.plot(R_vals*1.0e1,(energies[:,n]-E_refs[n])*1e3,label="QM/MM ESPF-DRF state "+str(n))
plt.xlabel("Separation [Angstrom]")
plt.ylabel("State interaction energies [mH]")
plt.legend()

plt.show()