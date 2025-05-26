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
from scipy.optimize import minimize

from timeit import default_timer as timer


# Get info MM He system and set up the OpenMM simulation object
pdb = PDBFile("1-inputs/acetone-waterbox-eq.pdb")
frame = -1
modeller_full = Modeller(pdb.getTopology() , pdb.getPositions(frame=frame))
modeller = Modeller(pdb.getTopology() , pdb.getPositions(frame=frame))
modeller.delete([r for r in modeller.topology.residues() if r.name == "UNL"])
positions = modeller.getPositions()
topology = modeller.getTopology()
positions = modeller.getPositions()
forcefield = ForceField("amoeba2018.xml")
system = forcefield.createSystem(topology,nonbondedMethod=PME,nonbondedCutoff=1.0*nanometer)
#system = forcefield.createSystem(topology,nonbondedMethod=NoCutoff)
platform = Platform.getPlatformByName("OpenCL")
integrator = VerletIntegrator(1e-16*picoseconds)
simulation = Simulation(topology, system, integrator,platform)
simulation.context.setPositions(positions)

# set up the Pyscf QM system 
modeller = Modeller(pdb.getTopology() , pdb.getPositions(frame=frame))
modeller.delete([r for r in modeller.topology.residues() if r.name == "HOH"])
residue = [r for r in modeller.topology.residues()][0]
qm_positions = np.array(modeller.getPositions()._value) * 10. # in Angstrom
atom =  [[atom.element.symbol , qm_positions[n,:]] for n,atom in enumerate(residue.atoms())] 
# the def2-SV(P) basis 
with open('./1-inputs/basis_def2svpp.dat', 'r') as f:
    basis_svpp = f.read()

#mol = gto.M(atom=atom,unit="Angstrom",basis={"C":"pcseg1","O":"pcseg1","H":"pcseg0"},charge=0,verbose=2)
mol = gto.M(atom=atom,unit="Angstrom",basis=basis_svpp,charge=0,verbose=2)
#mol = gto.M(atom=atom,unit="Angstrom",basis="pcseg-0",charge=0,verbose=2)



auxbasis = "weigendjkfit"
#auxbasis = df.make_auxbasis(mol)
#auxbasis = "ccpvdzjkfit"
# The DFT method is chosen to be ωB97X-D3/def2-SVP
mf = dft.RKS(mol)
#mf.xc = "PBE0"
mf.xc = "HF"
mf.grids.level = 0
#mf = scf.RHF(mol)
mf = mf.density_fit(auxbasis=auxbasis)
mf.max_cycle = 120

resp = tdscf.TDA(mf)
resp.nstates = 2
resp.singlet = True
resp = None


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
rep_type_dict = {"O":0,"H":1}
mm_rep_types = [rep_type_dict[atom.element.symbol] for atom in topology.atoms()]
rep_cutoff = 10.
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
qmmm_system.mm_system.damp_perm =  True

qmmm_system.setupWCARepulsion(radius_type="vdw",gamma=0.6,R0=0.,epsilon=1.0e-1)

# get positions for the QM and MM atoms
mm_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
mm_unit = "nanometer" # default units for OpenMM is nm
qm_positions = mf.mol.atom_coords(unit="Bohr")
qm_unit = "Bohr" # get QM positions in Bohr
qmmm_system.setPositions(mm_positions=mm_positions,mm_unit=mm_unit,qm_positions=qm_positions,qm_unit=qm_unit)


# get density matrix from calculation
dm = None
# set as initial guess for QM/MM calculation (not required!)
qmmm_system.qm_system.dm_guess = dm
qmmm_system.mm_system.resp_mode_force = "linear"
x0 = np.vstack((qm_positions,mm_positions*Data.NM_TO_BOHR))
x0 = x0.flatten()
x_mm_0 = mm_positions.flatten()*Data.NM_TO_BOHR
x_qm_0 = qm_positions.flatten()
N_qm = qm_positions.shape[0]
N_mm = mm_positions.shape[0]

# most abundant isotope masses for the QM atoms
mass_dict = {"O":15.9949,"H":1.0078,"C":12.0000}
masses_qm = [mass_dict[mol.atom_symbol(k)] for k in range(0,N_qm)]
masses = np.array(masses_qm+[system.getParticleMass(n)._value for n in range(0,system.getNumParticles())])*Data.DALTON_TO_ME
inv_masses = 1.0/masses
#print(np.array([mol.atom_symbol(k) in ["C","O"] for k in range(0,N_qm)]))
heavy_atoms = np.where(np.array([mol.atom_symbol(k) in ["C","O"] for k in range(0,N_qm)]))[0]
#print(heavy_atoms.shape)

state = 0
qmmm_system.resp_states=[]

pe = qmmm_system.getEnergy()
#print("Initial excitation energies:", (pe[1:]-pe[0])*Data.HARTREE_TO_EV, "eV")

# aligns a set of positions to a reference set of positions. The rotation is U
def align_structures(x,x_ref):
    if np.max(np.abs(x-x_ref))<1.0e-8 :
        return x, np.eye(3)
    
    R = x.T.dot(x_ref)
    V, Sigma, WT = np.linalg.svd(R)
    chi = np.sign(np.linalg.det(R))
    U = (WT.T.dot( np.diag(np.array([1.0,1.0,chi])))).dot(V.T)
    x_new = x.dot(U.T)
    return x_new, U

def engradqm(x_in):
    # get QM and MM positions
    x = x_in.reshape((int(len(x_in)/3),3))
    x_qm = x[0:N_qm,:]
    x_mm = x_mm_0.reshape((N_mm,3))
    qmmm_system.setPositions(qm_positions=x_qm,qm_unit="Bohr",mm_positions=x_mm,mm_unit="Bohr")
    pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()
    qmmm_system.qm_system.dm_guess = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    grad = -(f_qm_resp).flatten()
    print(pe)
    return pe[state], grad

def engrad(x_in):
    # get QM and MM positions
    x = x_in.reshape((N_qm+N_mm,3))
    x_qm = x[0:N_qm,:]
    x_mm = x[N_qm:,:]
    qmmm_system.setPositions(qm_positions=x_qm,qm_unit="Bohr",mm_positions=x_mm,mm_unit="Bohr")
    #qmmm_system.qm_system.dm_guess = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()
    grad = - np.vstack((f_qm,f_mm)) 
    #grad = -(np.vstack((f_qm_resp,f_mm_resp))).flatten()
    
    return pe[state], grad

def get_forces_resp(x):
    # gets acceleration for QM/MM system + potential energy
    
    # get QM and MM positions
    x_qm = x[0:N_qm,:]
    x_mm = x[N_qm:,:]
    qmmm_system.qm_system.dm_guess = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.setPositions(qm_positions=x_qm,qm_unit="Bohr",mm_positions=x_mm,mm_unit="Bohr")
    #qmmm_system.qm_system.dm_guess = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    try:
        pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()
    except:
        try:
            qmmm_system.qm_system.dm_guess = None
            pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()
        except:
            print("Crash in energy/force calculation. Saving geometry.")
            x_pbc_qm,x_pbc_mm = qmmm_system.getPositions(enforce_pbc=True)
            x_pbc = np.vstack((x_pbc_qm,x_pbc_mm))
            with open("./1-outputs/crash-coordinates.pdb","w") as f:
                PDBFile.writeFile(modeller_full.getTopology(),x_pbc*Data.BOHR_TO_NM*nanometer,file=f)
            #qmmm_system.qm_system.dm_guess = None
            #qmmm_system.qm_system.mf_qmmm.init_guess = "atom"
            #pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()  

    #f = np.vstack((f_qm_resp[0],f_mm_resp[0])) 
    f = np.vstack((f_qm,f_mm)) 
    return f,pe

def get_forces_noresp(x):
    # gets acceleration for QM/MM system + potential energy
    
    # get QM and MM positions
    x_qm = x[0:N_qm,:]
    x_mm = x[N_qm:,:]
    qmmm_system.qm_system.dm_guess = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    qmmm_system.setPositions(qm_positions=x_qm,qm_unit="Bohr",mm_positions=x_mm,mm_unit="Bohr")
    #qmmm_system.qm_system.dm_guess = qmmm_system.qm_system.mf_qmmm.make_rdm1()
    try:
        pe,f_qm,f_mm = qmmm_system.getEnergyForces()
    except:
        try:
            qmmm_system.qm_system.dm_guess = None
            pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()
        except:
            print("Crash in energy/force calculation. Saving geometry.")
            x_pbc_qm,x_pbc_mm = qmmm_system.getPositions(enforce_pbc=True)
            x_pbc = np.vstack((x_pbc_qm,x_pbc_mm))
            with open("./1-outputs/crash-coordinates.pdb","w") as f:
                PDBFile.writeFile(modeller_full.getTopology(),x_pbc*Data.BOHR_TO_NM*nanometer,file=f)
            #qmmm_system.qm_system.dm_guess = None
            #qmmm_system.qm_system.mf_qmmm.init_guess = "atom"
            #pe,f_qm,f_mm,f_qm_resp,f_mm_resp = qmmm_system.getEnergyForces()  

    #f = np.vstack((f_qm_resp[0],f_mm_resp[0])) 
    f = np.vstack((f_qm,f_mm)) 
    return f,[pe]

if resp is None:
    get_forces = get_forces_noresp
else:
    get_force = get_forces_resp

T = 298.0
kB_T = T * Data.KELVIN_TO_HARTREE
beta  = 1.0/kB_T
tau_0 = 1.0e2 * Data.FS_TO_ATOMICTIME
gamma = 1.0/tau_0
gamma = 0.0
dt_fs = 0.5
dt = dt_fs * Data.FS_TO_ATOMICTIME
print("dt = ", dt, "gamma = ",gamma)
c1 = np.exp(-gamma*dt*0.5)
c2 = np.sqrt((1.0-c1*c1)*masses/beta)
if gamma == 0.0:
    c1 = 1.0
    c2 = 0.0 * c2

def ff_langevin_step(x_in,p_in,f_in):
    
    # first stochastic part
    R = np.random.randn(p_in.shape[0],p_in.shape[1])
    p_0 = p_in
    p = c1 * p_in + c2[:,None] * R
    rescale = (np.linalg.norm(p,axis=1) / np.linalg.norm(p_0,axis=1))
    p = rescale[:,None] * p_0
    
    # Velocoty-verlet like part
    p += (0.5*dt)*f_in
    
    x = x_in + dt * inv_masses[:,None] * p
    f,pe = get_forces(x)
    p += (0.5*dt)*f
    
    # second stochastic part
    R = np.random.randn(p_in.shape[0],p_in.shape[1])
    p_0 = p
    p = c1 * p + c2[:,None] * R
    rescale = (np.linalg.norm(p,axis=1) / np.linalg.norm(p_0,axis=1))
    p = rescale[:,None] * p_0
    
    # determine the final kinetic energy
    ke = 0.5*np.sum(inv_masses[:,None]*p*p)
    
    return x, p, f, pe, ke

# set-up for fast-forward langevin molecular dynamics
n_steps = 2000 
x = np.vstack((qm_positions,mm_positions*Data.NM_TO_BOHR))
np.random.seed(seed=0)
p = np.random.randn(x.shape[0],x.shape[1]) * np.sqrt(kB_T*masses)[:,None]
print("Initial force starting...")
f,pe = get_forces(x)
print("Initial force done.")
ke = 0.5*np.sum(inv_masses[:,None]*p*p)
e_tot = pe[state] + ke
e_tot_0 = e_tot
x_ref = x[heavy_atoms,:] + 0.
x_ref = x_ref - np.mean(x_ref,axis=0)[None,:]

# saved geometries
pdbfile = open("1-outputs/acetone-pbcbox-nve-hf-0-5fs-1ps.pdb","w")
PDBFile.writeHeader(modeller_full.getTopology(),file=pdbfile)
datafile = open("1-outputs/acetone-pbcbox-nve-hf-0-5fs-1ps.csv","w")
tdmfile = open("1-outputs/acetone-pbcbox-nve-hf-0-5fs-tdm-1ps.csv","w")
n=0
data = [(0)*dt_fs,ke,pe[state]+ke] + [pe_n for pe_n in pe]
datafile.write(','.join([str(x) for x in data])+'\n')
if resp is not None:
    tdm = qmmm_system.qm_system.resp_qmmm.transition_dipole()
    tdm_data = list(tdm.flatten())
    tdmfile.write(','.join([str(x) for x in tdm_data])+'\n')
print("step: "+str(0)," energy [Ha] : ", str(e_tot), "change in energy [Ha] : ", str(e_tot-e_tot_0), "pe :", str(pe), "ke : ", str(ke))
    
# run molecular dynamics
start = timer()
for n in range(0,n_steps):
    x, p, f, pe, ke = ff_langevin_step(x,p,f)
    #PDBFile.writeFile(modeller_full.getTopology(),x*Data.BOHR_TO_NM*nanometer,"frame"+str(n+1)+".pdb")
    if (n+1)%1 == 0:
        e_tot = pe + ke
        print("Time elapsed : " , timer() - start , "s")
        print("step: "+str(n+1)," energy [Ha] : ", str(e_tot), "change in energy [Ha] : ", str(e_tot-e_tot_0), "pe :", str(pe), "ke : ", str(ke))
        data = [(n+1)*dt_fs,ke,pe[state]+ke] + [pe_n for pe_n in pe]
        datafile.write(','.join([str(x) for x in data])+'\n')
        if resp is not None:
            tdm = qmmm_system.qm_system.resp_qmmm.transition_dipole()
            x_mol = x[heavy_atoms,:]+0.
            x_mol = x_mol - np.mean(x_mol,axis=0)[None,:]
            x_new,U = align_structures(x_mol,x_ref)
            tdm_data = list(tdm.flatten()) + list((tdm.dot(U.T)).flatten())
            tdmfile.write(','.join([str(x) for x in tdm_data])+'\n')
    if (n+1)%10 == 0:
        x_pbc_qm,x_pbc_mm = qmmm_system.getPositions(enforce_pbc=True)
        x_pbc = np.vstack((x_pbc_qm,x_pbc_mm))
        PDBFile.writeModel(modeller_full.getTopology(),x_pbc*Data.BOHR_TO_NM*nanometer,file=pdbfile,modelIndex=n+1)
        tdmfile.flush()
        datafile.flush()
        pdbfile.flush()
        

#qmmm_system.setPositions(qm_positions=x_qm_min,qm_unit="Bohr",mm_positions=x_mm_min,mm_unit="Bohr")
#x_pbc_qm,x_pbc_mm = qmmm_system.getPositions(enforce_pbc=True)
#x_pbc = np.vstack((x_pbc_qm,x_pbc_mm))
#PDBFile.writeModel(modeller_full.getTopology(),x_pbc*Data.BOHR_TO_NM*nanometer,file=pdbfile)
PDBFile.writeFooter(modeller_full.getTopology(),file=pdbfile)
pdbfile.close()
datafile.close()
tdmfile.close()