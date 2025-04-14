# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of methanal and a singlet H2O molecule
H2C=O H-O-H
The HA-O-HB HA position is scanned in x,y,z directions and analytic gradients are calculated

In this example a pre-limit for of the dipoles is used in the MM energy expansion. So the dipole is represented as 
two point charges charge = +/- dipole/dr at x = +/- dr/2. This resolves an error that occurs on some linux CPU systems.
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
mol = gto.M(atom="7-inputs/methanal.xyz",unit="Angstrom",basis="STO-3G",charge=0)
# The DFT method is chosen to be a long-range-corrected functional with density fitting
mf = dft.RKS(mol)
mf.xc = "B3LYP"
#mf = scf.RHF(mol)
mf.kernel()
resp = tdscf.TDA(mf)
resp.singlet = True
resp.kernel()
resp.nstates = 8

grad_resp = resp.nuc_grad_method()
grad_resp.kernel()
#mf.xc = "PBE0"
#mf = mf.density_fit(auxbasis="weigendjkfit")


# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("7-inputs/h2o.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("7-inputs/h2o.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("Reference")
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

qmmm_system.mm_system.use_prelim_mpole = False # set to true to use pre-limit form of dipoles in the energy expansion
qmmm_system.mm_system.prelim_dr = 5.0e-3 # default value is 1.0e-2
qmmm_system.mm_system.resp_mode_force = "linear"

qmmm_system.qm_system.jdrf_pre = 1.0 
qmmm_system.qm_system.kdrf_pre = 1.0 

# get positions for the QM and MM atoms
mm_positions_ref = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions_ref,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)

# centre the O atom in acrolein
qm_positions -= qm_positions[1,:]
qmmm_system.setPositions(qm_positions=qm_positions,qm_unit=qm_unit)


# get the reference QM and MM energies
mf.kernel()
E_qm = mf.energy_tot() # in hartree
E_mm = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value * Data.KJMOL_TO_HARTREE # in hartree
E_qmmm_0 = E_qm + E_mm

# get density matrix from calculation
dm = mf.make_rdm1()
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm

# position of H-O-H H atom in nm
R_HOH = np.array([0.2,0,0])
# set up a grid of separations in nanometres units
R_vals = np.linspace(0.5,-0.5,num=7) * 0.025
energies = np.zeros((3,R_vals.shape[0]))
energies_resp = np.zeros((3,R_vals.shape[0]))
forces_qm = np.zeros((3,R_vals.shape[0],qm_positions.shape[0],3))
forces_mm = np.zeros((3,R_vals.shape[0],mm_positions_ref.shape[0],3))
forces_resp_qm = np.zeros((3,R_vals.shape[0],qm_positions.shape[0],3))
forces_resp_mm = np.zeros((3,R_vals.shape[0],mm_positions_ref.shape[0],3))
state = 1
qmmm_system.resp_states = [state]
J = 1
for x in range(0,3):
    n_x = np.array([0.,0.,0.])
    n_x[x] = 1.0
    for n,R in enumerate(R_vals):
        # set dm guess
        qmmm_system.qm_system.dm_guess = dm
        # set the MM atom positions
        mm_positions = mm_positions_ref + R_HOH[None,:]
        mm_positions[J,:] += R * n_x
        mm_unit = "nanometer"
        qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit)
        # get the energy
        E_qmmm,F_qm,F_mm,F_resp_qm,F_resp_mm = qmmm_system.getEnergyForces()
        # get density matrix
        #dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
        #qmmm_system.qm_system.dm_guess = dm
        # save enegy
        energies[x,n] = E_qmmm[0] + 0
        energies_resp[x,n] = E_qmmm[state] + 0
        forces_qm[x,n,:,:] = F_qm + 0
        forces_mm[x,n,:,:] = F_mm + 0
        forces_resp_qm[x,n,:,:] = F_resp_qm[0] + 0
        forces_resp_mm[x,n,:,:] = F_resp_mm[0] + 0


# calculate the fourth-order finite difference forces
dR = (R_vals[1]-R_vals[0])*Data.NM_TO_BOHR
N_R = len(R_vals)
forces_num = []
forces_an = []
forces_resp_num = []
forces_resp_an = []
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
    forces_an.append(forces_mm[x,:,J,x])
    
    E = energies_resp[x,:]+0
    f_num_4_resp = np.zeros((N_R))
    for n in range(2,N_R-2):
        #print(n)
        f_num_4_resp[n] = (1.0/12.0)*E[n-2] - (2.0/3.0) * E[n-1] +  (2.0/3.0) * E[n+1] - (1.0/12.0)*E[n+2]

    f_num_4_resp *= -(1.0/dR)
    f_num_4_resp[0:2] = np.nan
    f_num_4_resp[N_R-2] = np.nan
    f_num_4_resp[N_R-1] = np.nan
    
    forces_resp_num.append(f_num_4_resp)
    forces_resp_an.append(forces_resp_mm[x,:,J,x])

print("Numerical forces") 
R_vals_num = R_vals[2:-2]
for f in forces_num:
    print(f[2:-2])     
print("Analytical forces") 
for f in forces_an:
    print(f[2:-2])   

print("Numerical resp forces") 
R_vals_num = R_vals[2:-2]
for f in forces_resp_num:
    print(f[2:-2])     
print("Analytical resp forces") 
for f in forces_resp_an:
    print(f[2:-2])   

print("Analytical - numerical resp forces") 
for f_an,f_num in zip(forces_resp_an,forces_resp_num):
    print(f_an[2:-2]-f_num[2:-2])  

# plot energies 
axes = {0:"x",1:"y",2:"z"}
for x in range(0,3):
    plt.plot(R_vals*1.0e1,(energies[x,:]-E_qmmm_0)*1e3,label="Energies along "+axes[x])
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Ground state energy [mH]")
plt.legend()
plt.show()
    
# plot energies 
axes = {0:"x",1:"y",2:"z"}
for x in range(0,3):
    plt.plot(R_vals_num*1.0e1,(forces_num[x][2:-2])*1e3,label="Numerical "+axes[x])
    plt.plot(R_vals*1.0e1,(forces_an[x])*1e3,'--',label="Analytical "+axes[x])
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Ground state force [mH/bohr]")
plt.legend()

plt.show()
    
# plot energies 
axes = {0:"x",1:"y",2:"z"}
for x in range(0,3):
    plt.plot(R_vals*1.0e1,(energies_resp[x,:]-E_qmmm_0)*1e3,label="Energies along "+axes[x])
plt.xlabel("Separation [Angstrom]")
plt.ylabel("TDDFT Energy [mH]")
plt.legend()
plt.show()
    
# plot energies 
axes = {0:"x",1:"y",2:"z"}
for x in range(0,3):
    plt.plot(R_vals_num*1.0e1,(forces_resp_num[x][2:-2])*1e3,label="Numerical "+axes[x])
    plt.plot(R_vals*1.0e1,(forces_resp_an[x])*1e3,'--',label="Analytical "+axes[x])
    #plt.plot(R_vals*1.0e1,(forces_an[x])*1e3,':',label="Analytical S0"+axes[x])
plt.xlabel("Separation [Angstrom]")
plt.ylabel("TDDFT Force [mH/bohr]")
plt.legend()

plt.show()