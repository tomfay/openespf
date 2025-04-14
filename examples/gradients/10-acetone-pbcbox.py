# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of acrolein and H2O in a periodic box of H2O
'''

# import pyscf for setting up the QM part of the calculation
from pyscf import gto, scf, dft, tdscf, df
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


# Get info MM He system and set up the OpenMM simulation object
#pdb = PDBFile("6-inputs/acrolein-bigpbcbox.pdb")
pdb = PDBFile("10-inputs/acetone-waterbox.pdb")
modeller = Modeller(pdb.getTopology() , pdb.getPositions())
modeller.delete([r for r in modeller.topology.residues() if r.name == "UNL"])
positions = modeller.getPositions()
topology = modeller.getTopology()
positions = modeller.getPositions()
forcefield = ForceField("amoeba2018.xml")
system = forcefield.createSystem(topology,nonbondedMethod=PME,nonbondedCutoff=1.0*nanometer)
#system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("Reference")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)

# set up the Pyscf QM system 
modeller = Modeller(pdb.getTopology() , pdb.getPositions())
modeller.delete([r for r in modeller.topology.residues() if r.name == "HOH"])
residue = [r for r in modeller.topology.residues()][0]
qm_positions = np.array(modeller.getPositions()._value) * 10. # in Angstrom
atom =  [[atom.element.symbol , qm_positions[n,:]] for n,atom in enumerate(residue.atoms())] 
# the def2-SV(P) basis 
with open('./10-inputs/basis_def2svpp.dat', 'r') as f:
    basis_svpp = f.read()

mol = gto.M(atom=atom,unit="Angstrom",basis={"C":"pcseg1","O":"pcseg1","H":"pcseg0"},charge=0,verbose=2)
mol = gto.M(atom=atom,unit="Angstrom",basis="pcseg-0",charge=0,verbose=2)
auxbasis = "weigendjkfit"
#auxbasis = df.make_auxbasis(mol)
#auxbasis = "ccpvdzjkfit"
# The DFT method is chosen to be ωB97X-D3/def2-SVP
mf = dft.RKS(mol)
mf.xc = "PBE0"
#mf.xc = "HF,LYP"
mf.grids.level = 0
#mf = scf.RHF(mol)
#mf = mf.density_fit(auxbasis=auxbasis)

resp = tdscf.TDA(mf)
resp.nstates = 3
resp.singlet = True


# information about the QM-MM interaction
multipole_order = 1 # 0=charges, 1=charges+dipoles for QM ESPF multipole operators
multipole_method = "espf" # "espf" or "mulliken" type multipole operators
# information for the exchange-repulsion model (atomic units)
rep_type_info = [{"N_eff":4.0+0.669,"R_dens":1.71*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0-0.3345,"R_dens":1.54*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
rep_type_info = [{"N_eff":4.0+0.51966,"R_dens":(1.71*(1-0.51966)+2.03*0.51966)*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0 - 0.51966*0.5,"R_dens":(1.54*(1-0.51966*0.5))*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
rep_type_dict = {"O":0,"H":1}
mm_rep_types = [rep_type_dict[atom.element.symbol] for atom in topology.atoms()]
rep_cutoff = 8.
# information about how the QM multipole - MM induced dipole interactions are damped (OpenMM units)
qm_damp_dict = {"H":0.496e-3**(1./6.),"C":1.334e-3**(1./6.),"O":0.873e-3**(1./6.)}
qm_damp = [qm_damp_dict[atom.element.symbol] for atom in modeller.topology.atoms()]
#qm_damp = [1.0e-3**(1./6.)]*len(mol.atom_charges())
qm_thole = [0.39]*len(mol.atom_charges())

# create the QMMMSystem object that performs the QM/MM energy calculations
qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method,qm_resp=resp)
# set additional parameters for the exchange repulsion + damping of electrostatics
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)

# get positions for the QM and MM atoms
mm_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)

# get the reference QM and MM energies
start = timer()
mf.kernel()
print("Vacuum SCF time: ",timer()-start,"s")
resp = tdscf.TDA(mf)
resp.nstates = 3
resp.singlet = True
resp.kernel()
print("Vacuum total calculation time: ",timer()-start,"s")
start = timer()
mf.nuc_grad_method().run()
print("Vacuum SCF gradient calculation time: ",timer()-start,"s")
start = timer()
resp.nuc_grad_method().run()
print("Vacuum TDDFT gradient calculation time: ",timer()-start,"s")

dipole_vac = mf.dip_moment()
print("Dipole magnitude (Vac) = " , np.linalg.norm(dipole_vac)," Debye")
E_qm = mf.energy_tot() # in hartree
E_mm = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value * Data.KJMOL_TO_HARTREE # in hartree
E_qmmm_0 = E_qm + E_mm
print("Excitation energies = ", (resp.e)*Data.HARTREE_TO_EV)
print("E(QM) = ",E_qm)
print("E(MM) = ",E_mm)

# get density matrix from calculation
dm = mf.make_rdm1()
dm = None
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm


# Do the QM/MM calculation
#E_qmmm_terms = qmmm_system.getEnergy(return_terms=True)
# Unlike in previosu examples, the above gives a breakdown of some of the terms in the QM/MM Energy
#print("E(QM/MM) terms = ", E_qmmm_terms)
#E_qmmm = np.sum(np.array([E_qmmm_terms[k] for k in list(E_qmmm_terms.keys())]))
#qmmm_system.mm_system.resp_mode = "quadratic"
#E_qmmm,_,_ = qmmm_system.getEnergyForces()
E_qmmm = qmmm_system.getEnergy()
print("Excitation energies = ", (E_qmmm[1:]-E_qmmm[0])*Data.HARTREE_TO_EV)
print("E(QM/MM) = ",E_qmmm)
print("E(QM/MM) - E(QM) - E(MM) = ",E_qmmm - E_qmmm_0)

#qmmm_system.qm_system.resp = None
# The dipole moment of the QM system can be accessed as follows, as expected it is polarised relative to the vacuum
dipole_aq = qmmm_system.qm_system.mf_qmmm.dip_moment()
print("Dipole magnitude (Aq) = " , np.linalg.norm(dipole_aq)," Debye")
print("Magnitude of change in dipole moment = " , np.linalg.norm(dipole_aq-dipole_vac)," Debye")

# Do the QM/MM calculation
qmmm_system.print_info = True
qmmm_system.qm_system.dm_guess = dm
start = timer()
E_qmmm,F_qm,F_mm,F_qm_resp,F_mm_resp = qmmm_system.getEnergyForces()
#E_qmmm,F_qm,F_mm = qmmm_system.getEnergyForces()
print("Quadratic calculation time:", timer()-start, "s")
print("E(QM/MM) (quadratic) = ", E_qmmm)
# do the calcualtion with linear scaling forces
qmmm_system.qm_system.dm_guess = dm
#qmmm_system.mm_system.resp_mode_force = "linear"
start = timer()
E_qmmm_lin,F_qm_lin,F_mm_lin,F_qm_resp_lin,F_mm_resp_lin = qmmm_system.getEnergyForces()
#E_qmmm_lin,F_qm_lin,F_mm_lin = qmmm_system.getEnergyForces()
print("Linear calculation time:", timer()-start, "s")
print("E(QM/MM) (linear) = ", E_qmmm)


