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
from openespf.QMMultipole import QMMultipole, getESPMatrix
import openespf.Data as Data

#import numpy and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt


# set up the Pyscf QM system 
mol = gto.M(atom="1-inputs/h2o.xyz",unit="Angstrom",basis="def2-TZVPP",charge=0)
# The DFT method is chosen to be a long-range-corrected functional with density fitting
mf = dft.RKS(mol)
mf.xc = "HF"

mf.kernel()

n_grid = 100 
R_min = 5.0*Data.ANGSTROM_TO_BOHR 
R_max = 1000.0*Data.ANGSTROM_TO_BOHR 
grid_coords = np.zeros([n_grid,3])
x = 2
grid_coords[:,x] = np.linspace(R_min,R_max,n_grid)
esp_op = mol.intor('int1e_grids',grids=grid_coords)

dm = mf.make_rdm1()

esp_el = -np.einsum('nm,knm->k',dm,esp_op)
Z = np.array(mol.atom_charges())
R = mol.atom_coords()
N = R.shape[0]
R_kA = R[None,:,:]-grid_coords[:,None,:]
r_kA = np.linalg.norm(R_kA,axis=2)
r_kA_inv = 1.0/r_kA
#print(r_kA.shape)
esp_nuc = np.einsum('A,kA',Z,r_kA_inv)

multipole = QMMultipole(mol=mol,multipole_order=0)
Q_op = np.array(multipole.getMultipoleOperators())
print(Q_op.shape)
Q_el = np.einsum('nm,Anm->A',dm,Q_op) 
print(Q_el+Z)
q_espf = Q_el[0:N]+Z
esp_espf = np.einsum('A,kA->k',q_espf,r_kA_inv)
esp_espf_el = esp_espf-esp_nuc

multipole = QMMultipole(mol=mol,multipole_order=1)
Q2_op = np.array(multipole.getMultipoleOperators())
D = getESPMatrix(grid_coords,R,1)
Q2_el = np.einsum('nm,Anm->A',dm,Q2_op) 
esp_espf2_el = np.einsum('a,ka->k',Q2_el,D)
esp_espf2 = esp_espf2_el + esp_nuc

np.savetxt('./1-outputs/esp-out-axis-'+str(x)+'.out',np.array([grid_coords[:,x]*Data.BOHR_TO_ANGSTROM,esp_nuc,esp_el,esp_espf_el,esp_espf2_el]).T)

#plt.plot(grid_coords[:,0]*Data.BOHR_TO_ANGSTROM,esp_el+esp_nuc)
#plt.plot(grid_coords[:,0],esp_nuc)
#plt.plot(grid_coords[:,0],esp_el)
#plt.plot(grid_coords[:,0]*Data.BOHR_TO_ANGSTROM,esp_espf)
plt.loglog(grid_coords[:,x]*Data.BOHR_TO_ANGSTROM,np.abs(esp_el+esp_nuc-esp_espf))
plt.loglog(grid_coords[:,x]*Data.BOHR_TO_ANGSTROM,np.abs(esp_el+esp_nuc-esp_espf2))
plt.show()

