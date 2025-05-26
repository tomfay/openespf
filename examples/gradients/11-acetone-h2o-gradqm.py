# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of methanal and a singlet H2O molecule
H2C=O H-O-H
The HA-O-HB HA position is scanned in x,y,z directions and analytic gradients are calculated

In this example a pre-limit for of the dipoles is used in the MM energy expansion. So the dipole is represented as 
two point charges charge = +/- dipole/dr at x = +/- dr/2. This resolves an error that occurs on some linux CPU systems.
'''

# import pyscf for setting up the QM part of the calculation
from pyscf import gto, scf, dft, tdscf, lib
#import dftd3.pyscf as disp
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
from timeit import default_timer as timer

#dft.libxc.XCFUN_ENABLED = False
#lib.num_threads(2)
#print("Num threads:",lib.num_threads())
with open('./10-inputs/basis_def2svpp.dat', 'r') as f:
    basis_svpp = f.read()

# set up the Pyscf QM system u
mol = gto.M(atom="11-inputs/acetone-s1-geom2.xyz",unit="Angstrom",basis=basis_svpp,charge=0)
# The DFT method is chosen to be a long-range-corrected functional with density fitting
auxbasis = "weigendjkfit"
mf = dft.RKS(mol)
mf = mf.density_fit(auxbasis=auxbasis)
mf.xc = "PBE0"
mf.grids.level = 0

#mf.nlc = False
#print(mf.disp)
#print()
#mf.disp = "d3bj"
#mf = mf.density_fit(auxbasis="ccpvdzjkfit")
#mf = sgx.sgx_fit(mf, pjs=False)
#mf.with_df.dfj = True

#mf = scf.RHF(mol)
#mf.conv_tol = 1.0e-14
#mf.max_cycle = 2000
mf.kernel()
#print(scf.dispersion.get_dispersion(mf,disp="d3bj"))

resp = tdscf.TDA(mf)
resp.nstates = 3
resp.singlet = True
resp.kernel()
#d3 = disp.DFTD3Dispersion(mol, xc="PBE0")
#print(d3.kernel())
#print(disp.energy(mf).run())

#grad_resp = resp.nuc_grad_method()
#grad_resp.kernel()
#mf.xc = "PBE0"
#mf = mf.density_fit(auxbasis="weigendjkfit")


# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("7-inputs/h2o.pdb")
topology = pdb.getTopology() 
topology.setUnitCellDimensions((2.2,2.2,2.2))
positions = pdb.getPositions()
forcefield = ForceField("amoeba2018.xml")
system = forcefield.createSystem(topology,nonbondedMethod=PME,nonbondedCutoff=1.0*nanometer)
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff) 
platform = Platform.getPlatformByName("OpenCL")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)

# information about the QM-MM interaction
multipole_order = 0 # 0=charges, 1=charges+dipoles for QM ESPF multipole operators
multipole_method = "espf" # "espf" or "mulliken" type multipole operators
# information for the exchange-repulsion model (atomic units)
rep_type_info = [{"N_eff":4.0+0.669,"R_dens":1.71*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0-0.3345,"R_dens":1.54*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
rep_type_info = [{"N_eff":4.0+0.51966,"R_dens":(1.71*(1-0.51966)+2.03*0.51966)*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0 - 0.51966*0.5,"R_dens":(1.54*(1-0.51966*0.5))*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
rep_type_info = [{"N_eff":4.0+0.51966,"R_dens":(1.71)*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0 - 0.51966*0.5,"R_dens":(1.54)*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
mm_rep_types = [0,1,1]
rep_cutoff = 20.
# information about how the QM multipole - MM induced dipole interactions are damped (OpenMM units)
#qm_damp = [0.001**(1./6.)]*len(mol.atom_charges())
qm_damp_dict = {"H":0.496e-3**(1./6.),"C":1.334e-3**(1./6.),"O":0.873e-3**(1./6.)}
qm_damp = [qm_damp_dict[mol.atom_symbol(k)] for k in range(0,len(mol.atom_charges()))]
print(qm_damp)
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
mm_positions_ref += np.array([0.,0.1,-0.05])[None,:]
mm_positions_ref[[0,2],:] += np.array([0.05,0.,0.])[None,:]
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions_ref,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)

# centre the C=O carbon atom in acrolein
qm_positions -= qm_positions[0,:]
qm_positions_ref = qm_positions+0
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

# set up a grid of separations in nanometres units
R_vals = np.linspace(5.,0.8,num=20) * 0.1
energies = []
for n,R in enumerate(R_vals):
    # set the MM atom positions
    mm_positions = np.array([[R,0,0]])+mm_positions_ref
    mm_unit = "nanometer"
    qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit)
    # get the energy
    try:
        E_qmmm = qmmm_system.getEnergy()
        #E_qmmm,_,_,_,_ = qmmm_system.getEnergyForces()
        #q0,qq0 = qmmm_system.qm_system.getMeanTotalMultipoles()
        #q,qq = qmmm_system.qm_system.getMeanTotalMultipolesResp()
        #print("SCF multipoles : ", q0)
        #print("TDDFT multipoles : ", q0+q)
    except:
        E_qmmm = [np.nan] * len(energies[0])
    # get density matrix
    dm = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.qm_system.dm_guess = dm
    # save enegy
    energies.append(E_qmmm)


x_pbc_qm,x_pbc_mm = qmmm_system.getPositions(enforce_pbc=True)
x_pbc = np.vstack((x_pbc_qm,x_pbc_mm)) *Data.BOHR_TO_ANGSTROM
atoms = [mol.atom_symbol(k) for k in range(0,len(qm_damp))] + ["O", "H","H"]
file = open("11-inputs/acetone-h2o.xyz","w")
file.write(str(len(atoms))+"\n")
file.write("coordinates"+"\n")
for n in range(0,x_pbc.shape[0]):
    file.write(' '.join([atoms[n],str(x_pbc[n,0]),str(x_pbc[n,1]),str(x_pbc[n,2])])+"\n")
file.close()

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