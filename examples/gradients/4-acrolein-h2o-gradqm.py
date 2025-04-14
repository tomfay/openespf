# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of acrolein and a singlet H2O molecule
CH2=CH-CH=O H-O-H
The separation between the C=O O atom and H of H2O is scanned.
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


# set up the Pyscf QM system u
mol = gto.M(atom="4-inputs/acrolein.xyz",unit="Angstrom",basis="pc-0",charge=0)
# The DFT method is chosen to be a long-range-corrected functional with density fitting
mf = dft.RKS(mol)
mf.xc = "PBE0"
#mf = mf.density_fit(auxbasis="weigendjkfit")


# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("4-inputs/h2o.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("3-inputs/h2o.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("CPU")
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
qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method)
# set additional parameters for the exchange repulsion + damping of electrostatics
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)

# get positions for the QM and MM atoms
mm_positions_ref = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions_ref,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)

# centre the O atom in acrolein
qm_positions -= qm_positions[3,:]
qmmm_system.setPositions(qm_positions=qm_positions,qm_unit=qm_unit)
qm_positions_ref = qm_positions+0

# get the reference QM and MM energies
mf.kernel()
E_qm = mf.energy_tot() # in hartree
E_mm = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value * Data.KJMOL_TO_HARTREE # in hartree
E_qmmm_0 = E_qm + E_mm

# get density matrix from calculation
dm = mf.make_rdm1()
# set as initial guess for QM/MM calculation (not required!)
#qmmm_system.qm_system.dm_guess = dm

# position of H-O-H H atom in nm
R_HOH = np.array([0.25,0,0])
# set up a grid of separations in nanometres units
R_vals = np.linspace(0.5,-0.5,num=7) * 0.025
energies = np.zeros((3,R_vals.shape[0]))
forces_qm = np.zeros((3,R_vals.shape[0],qm_positions.shape[0],3))
forces_mm = np.zeros((3,R_vals.shape[0],mm_positions_ref.shape[0],3))
for x in range(0,3):
    n_x = np.array([0.,0.,0.])
    n_x[x] = 1.0
    for n,R in enumerate(R_vals):
        # set dm guess
        #qmmm_system.qm_system.dm_guess = dm
        # set the MM atom positions
        mm_positions = mm_positions_ref + R_HOH.reshape((1,3))
        qm_positions = qm_positions_ref+0
        qm_positions[3,:] += R * n_x *Data.NM_TO_BOHR
        mm_unit = "nanometer"
        qm_unit = "Bohr" # get QM positions in Bohr
        qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)
        # get the energy
        E_qmmm,F_qm,F_mm = qmmm_system.getEnergyForces()
        # get density matrix
        #dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
        #qmmm_system.qm_system.dm_guess = dm
        # save enegy
        energies[x,n] = E_qmmm + 0
        forces_qm[x,n,:,:] = F_qm + 0
        forces_mm[x,n,:,:] = F_mm + 0


# calculate the fourth-order finite difference forces
dR = (R_vals[1]-R_vals[0])*Data.NM_TO_BOHR
N_R = len(R_vals)
forces_num = []
forces_an = []
for x in range(0,3):
    E = energies[x,:]+0
    f_num_4 = np.zeros((N_R))
    for n in range(2,N_R-2):
        #print(n)
        f_num_4[n] = (1.0/12.0)*E[n-2] - (2.0/3.0) * E[n-1] +  (2.0/3.0) * E[n+1] - (1.0/12.0)*E[n+2]

    f_num_4 *= -(1.0/dR)
    f_num_4[0:2] = np.nan
    f_num_4[N_R-2] = np.nan
    f_num_4[N_R-1] = np.nan
    forces_num.append(f_num_4)
    forces_an.append(forces_qm[x,:,3,x])

print("Numerical forces") 
R_vals_num = R_vals[2:-2]
for f in forces_num:
    print(f[2:-2])    
print("Analytical forces") 
for f in forces_an:
    print(f[2:-2])   
# plot energies 
axes = {0:"x",1:"y",2:"z"}
for x in range(0,3):
    plt.plot(R_vals*1.0e1,(energies[x,:]-E_qmmm_0)*1e3,label="Energies along "+axes[x])
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Energy [mH]")
plt.legend()
plt.show()
    
# plot energies 
axes = {0:"x",1:"y",2:"z"}
for x in range(0,3):
    plt.plot(R_vals_num*1.0e1,(forces_num[x][2:-2])*1e3,label="Numerical "+axes[x])
    plt.plot(R_vals*1.0e1,(forces_an[x])*1e3,'--',label="Analytical "+axes[x])
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Force [mH/bohr]")
plt.legend()

plt.show()