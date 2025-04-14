# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
A simple example calculating the enery gradient of a QM Na+ ion with a polarisable MM He atom:
Na+  He
The separation between Na+ and He is scanned.
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



# set up the Pyscf QM system using HF/cc-pVDZ
mol = gto.M(atom='Na 0 0 0',unit="Bohr",basis="cc-pVDZ",charge=1)
mf = scf.RHF(mol).density_fit()

# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("1-inputs/he.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("1-inputs/he.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("CPU")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)


# information about the QM-MM interaction
multipole_order = 1 # 0=charges, 1=charges+dipoles for QM ESPF multipole operators
multipole_method = "espf" # "espf" or "mulliken" type multipole operators
# information for the exchange-repulsion model (atomic units)
rep_type_info = [{"N_eff":2.0,"R_dens":1.34*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
mm_rep_types = [0]
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
qmmm_system.mm_system.resp_mode_force = "linear"

# get positions for the QM and MM atoms
mm_positions = simulation.context.getState(getPositions=True).getPositions()
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)

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
R_vals = np.linspace(6.,2.0,num=50) * Data.ANGSTROM_TO_BOHR
energies = np.zeros(R_vals.shape)
forces_qm = np.zeros((R_vals.shape[0],qm_positions.shape[0],3))
forces_mm = np.zeros((R_vals.shape[0],1,3))
for n,R in enumerate(R_vals):
    # set the MM atom positions
    mm_positions = np.array([[R,0,0]])
    mm_unit = "Bohr"
    qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit)
    # get the energy
    E_qmmm,F_qm,F_mm = qmmm_system.getEnergyForces()
    # get density matrix
    dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.qm_system.dm_guess = dm
    # save enegy
    energies[n] =  E_qmmm + 0
    forces_qm[n,:,:] = F_qm + 0
    forces_mm[n,:,:] = F_mm + 0

# calculate the fourth-order finite difference forces
dR = R_vals[1]-R_vals[0]
N_R = len(R_vals)
E = energies
f_num_4 = np.zeros((N_R))
for n in range(2,N_R-2):
    #print(n)
    f_num_4[n] = (1.0/12.0)*E[n-2] - (2.0/3.0) * E[n-1] +  (2.0/3.0) * E[n+1] - (1.0/12.0)*E[n+2]

f_num_4 *= -(1.0/dR)
f_num_4[0:2] = np.nan
f_num_4[[N_R-2,N_R-1]] = np.nan

# plot energies and model -α/2R^4 expected at long range
plt.plot(R_vals[2:-2]*Data.BOHR_TO_ANGSTROM,(f_num_4[2:-2])*1e3,label="Numerical forces")
axes={0:"x",1:"y",2:"z"} 
for x in range(0,3):
    plt.plot(R_vals[2:-2]*Data.BOHR_TO_ANGSTROM,(forces_mm[2:-2,0,x])*1e3,'--',label="MM force "+axes[x])
    plt.plot(R_vals[2:-2]*Data.BOHR_TO_ANGSTROM,(-forces_qm[2:-2,0,x])*1e3,':',label="-QM force "+axes[x])
#plt.plot(R_vals*Data.BOHR_TO_ANGSTROM,(-0.5*1.2/(R_vals**4))*1e3,'--',label="-α/2R^4")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Force [mH/bohr]")
plt.legend()

plt.show()