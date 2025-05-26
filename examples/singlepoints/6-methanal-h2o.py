# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of methanal and a singlet H2O molecule
H2C=O H-O-H
The HA-O-HB HA position is scanned in x,y,z directions and analytic gradients are calculated

In this example a pre-limit for of the dipoles is used in the MM energy expansion. So the dipole is represented as 
two point charges charge = +/- dipole/dr at x = +/- dr/2. This resolves an error that occurs on some linux CPU systems.
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
mol = gto.M(atom="6-inputs/methanal.xyz",unit="Angstrom",basis="pc-1",charge=0)
mf = dft.RKS(mol)
#mol = gto.M(atom="6-inputs/methanal.xyz",unit="Angstrom",basis="pc-1",charge=0,spin=2)
#mf = dft.ROKS(mol)
mf.xc = "HF"

#mf.xc = "PBE0"
#mf = mf.density_fit(auxbasis="weigendjkfit")



# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("6-inputs/h2o.pdb")
topology = pdb.getTopology() 
positions = pdb.getPositions()
forcefield = ForceField("amoeba2018.xml")
#forcefield = ForceField("6-inputs/iamoeba_zero.xml")
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
qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method)
# set additional parameters for the exchange repulsion + damping of electrostatics
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)
qmmm_system.mm_system.damp_perm = True # default True
qmmm_system.mm_system.damp_charge_only = True # default True


#qmmm_system.mm_system.use_prelim_mpole = True # set to true to use pre-limit form of dipoles in the energy expansion
#qmmm_system.mm_system.prelim_dr = 5.0e-3 # default value is 1.0e-2

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
mf.dip_moment()
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm

# position of H-O-H H atom in nm
R_HOH = np.array([0.0,0,0])
# set up a grid of separations in nanometres units
R_vals = np.linspace(0.1,0.5,num=40) 
energies = np.zeros((R_vals.shape[0],))
forces_qm = np.zeros((3,R_vals.shape[0],qm_positions.shape[0],3))
forces_mm = np.zeros((3,R_vals.shape[0],mm_positions_ref.shape[0],3))
x = 0 
n_x = np.array([0.,0.,0.])
n_x[x] = 1.0
int_energies = [] 
mu = []
for n,R in enumerate(R_vals):
    # set dm guess
    qmmm_system.qm_system.dm_guess = dm
    # set the MM atom positions
    mm_positions = mm_positions_ref + R_HOH[None,:]
    mm_positions += R * n_x[None,:]
    mm_unit = "nanometer"
    qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit)
    # get the energy
    E_qmmm = qmmm_system.getEnergy()
    #E_qmmm,_,_ = qmmm_system.getEnergyForces()
    # get density matrix
    #dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    #qmmm_system.qm_system.dm_guess = dm
    # save enegy
    energies[n] = E_qmmm + 0
    int_energies.append( qmmm_system.qm_system.getInteractionEnergyDecomposition() )
    mu.append(qmmm_system.qm_system.mf_qmmm.dip_moment())
    

mus = np.linalg.norm(np.array(mu),axis=1)

print("Interaction energies [Hartree]:")
print(energies-E_qmmm_0)
#np.savetxt("./6-inputs/int-energies-nodamp.dat",np.vstack((R_vals,energies-E_qmmm_0,mus)).T,delimiter=',',header='Separation [nm],Interaction energy [Hartree],Dipole moment [au]')


# plot energies 
plt.rc('font', family='Helvetica') 
plt.plot(R_vals*1.0e1,(energies-E_qmmm_0)*1e3,label="Interaction energy ")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Energy [mH]")
plt.legend()
plt.show()

# plot energies 
plt.rc('font', family='Helvetica') 
plt.plot(R_vals*1.0e1,mus,label="Dipole moment")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Dipole moment [D]")
plt.legend()
plt.show()

decomp = {}
decomp_atm = {}
for key in int_energies[0].keys():
    decomp[key] = np.zeros(R_vals.shape)
for key in int_energies[0].keys():
    decomp_atm[key] = np.zeros((R_vals.shape[0],qm_positions.shape[0]))
atm_int = np.zeros((R_vals.shape[0],qm_positions.shape[0]))
atm_int_nostatic = np.zeros((R_vals.shape[0],qm_positions.shape[0]))
for n,decomp_n in enumerate(int_energies):
    for key in decomp_n.keys():
        decomp[key][n] = np.sum(decomp_n[key])
        if not key=="self":
            decomp_atm[key][n] = decomp_n[key]
            

#print(decomp_atm)            
decomp["self"] -= E_qm
keys = ["static","pol_mf","pol_fluct","rep","self"]
tot_int = np.zeros(R_vals.shape)
for key in keys:
    plt.plot(R_vals*1.0e1,decomp[key]*1e3,'--',label=key)
    tot_int += 1.0*decomp[key]
    atm_int += decomp_atm[key]
    if not key=="static":
        atm_int_nostatic += decomp_atm[key]
#print(tot_int)

plt.rc('font', family='Helvetica') 
plt.plot(R_vals*1.0e1,(energies-E_qmmm_0)*1e3,label="Interaction energy ")
#plt.plot(R_vals*1.0e1,tot_int*1e3,'-.',label="Decomp sum")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Energy [mH]")
plt.legend()
plt.show()

plt.rc('font', family='Helvetica') 
for J in range(0,mol.natm):
    plt.plot(R_vals*1.0e1,atm_int[:,J]*1e3,'--',label="Atom "+mol.elements[J]+" interaction energy")
plt.plot(R_vals*1.0e1,(energies-E_qmmm_0)*1e3,label="Total interaction energy")
#plt.plot(R_vals*1.0e1,np.sum(atm_int,axis=1)*1e3,label="Total interaction energy")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Energy [mH]")
plt.legend()
plt.show()

plt.rc('font', family='Helvetica') 
for J in range(0,mol.natm):
    plt.plot(R_vals*1.0e1,atm_int_nostatic[:,J]*1e3,'--',label="Atom "+mol.elements[J]+" interaction energy")
plt.plot(R_vals*1.0e1,(energies-E_qmmm_0)*1e3,label="Total interaction energy")
#plt.plot(R_vals*1.0e1,np.sum(atm_int,axis=1)*1e3,label="Total interaction energy")
plt.xlabel("Separation [Angstrom]")
plt.ylabel("Energy [mH]")
plt.legend()
plt.show()