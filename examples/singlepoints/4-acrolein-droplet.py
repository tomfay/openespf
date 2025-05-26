# Author: Thomas Fay <tom.patrick.fay@gmail.com>

'''
An interaction of acrolein and in a droplet of H2O
CH2=CH-CH=O
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


# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("4-inputs/acrolein-droplet.pdb")
modeller = Modeller(pdb.getTopology() , pdb.getPositions())
modeller.delete([r for r in modeller.topology.residues() if r.name == "UNL"])
positions = modeller.getPositions()
topology = modeller.getTopology()
positions = modeller.getPositions()
forcefield = ForceField("4-inputs/h2o.xml")
system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
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
mol = gto.M(atom=atom,unit="Angstrom",basis="def2-SVP",charge=0)
# The DFT method is chosen to be ωB97X-D3/def2-SVP
mf = dft.RKS(mol)
mf.xc = "PBE0"
#mf = mf.density_fit()


# information about the QM-MM interaction
multipole_order = 1 # 0=charges, 1=charges+dipoles for QM ESPF multipole operators
multipole_method = "espf" # "espf" or "mulliken" type multipole operators
# information for the exchange-repulsion model (atomic units)
rep_type_info = [{"N_eff":4.0+0.669,"R_dens":1.71*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3},
                {"N_eff":1.0-0.3345,"R_dens":1.54*Data.ANGSTROM_TO_BOHR,"rho(R_dens)":1.0e-3}] 
rep_type_dict = {"O":0,"H":1}
mm_rep_types = [rep_type_dict[atom.element.symbol] for atom in topology.atoms()]
rep_cutoff = 10.
# information about how the QM multipole - MM induced dipole interactions are damped (OpenMM units)
qm_damp = [0.001**(1./6.)]*len(mol.atom_charges())
qm_thole = [0.39]*len(mol.atom_charges())

# create the QMMMSystem object that performs the QM/MM energy calculations
qmmm_system = QMMMSystem(simulation,mf,multipole_order=multipole_order,multipole_method=multipole_method)
# set additional parameters for the exchange repulsion + damping of electrostatics
qmmm_system.setupExchRep(rep_type_info,mm_rep_types,cutoff=rep_cutoff,setup_info=None)
qmmm_system.mm_system.setQMDamping(qm_damp,qm_thole)
qmmm_system.mm_system.use_prelim_mpole = False

# get positions for the QM and MM atoms
mm_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)

# get the reference QM and MM energies
mf.kernel()
E_qm = mf.energy_tot() # in hartree
E_mm = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value * Data.KJMOL_TO_HARTREE # in hartree
E_qmmm_0 = E_qm + E_mm
print("E(QM) = ",E_qm)
print("E(MM) = ",E_mm)

# get density matrix from calculation
dm = mf.make_rdm1()
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm

# Do the QM/MM calculation
E_qmmm = qmmm_system.getEnergy()
print("E(QM/MM) = ",E_qmmm)
print("E(QM/MM) - E(QM) - E(MM) = ",E_qmmm - E_qmmm_0)
#np.savetxt('./4-inputs/energies.dat',np.array([E_qmmm,E_qmmm_0,E_qmmm-E_qmmm_0]),header='E_QMMM,E_QM+E_MM,Interaction [Hartree]')

# get interaction energy decomposition into atom-wise terms
int_energies = qmmm_system.qm_system.getInteractionEnergyDecomposition(print_decomp=True)
print(int_energies)