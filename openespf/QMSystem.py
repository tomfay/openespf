from pyscf import gto, scf, dft, tddft, ao2mo, lib
import numpy as np
from scipy.linalg import sqrtm, inv, solve
from .QMMultipole import QMMultipole
from copy import deepcopy, copy


'''
A wrapper for PySCF methods for calculating QM component of QM/MM energies and forces.
'''

class QMSystem:
    def __init__(self,qm_mf,qm_resp=None,int_method="drf",rep_method="exch",multipole_order=0,multipole_method="espf"):
        self.mf = copy(qm_mf)
        self.mol = self.mf.mol
        self.resp = qm_resp
        self.int_method = int_method
        self.rep_method = rep_method
        self.multipole_order = multipole_order
        self.multipole_method = multipole_method
        self.positions = self.mol.atom_coords(unit="Bohr")
        self.multipole = QMMultipole(mol=self.mol,multipole_order=self.multipole_order,multipole_method=self.multipole_method)
        self.dm_guess=None
        return
    
    def getPositions(self):
        return self.positions
    
    def setPositions(self,positions):
        '''
        Sets positions of the QM atoms. Assumes units of Bohr.
        [Need to check that this correctly updates self.mf.mol as well]
        '''
        atom = self.mol._atom
        new_atom = [ [atom[A][0],positions[A,:]] for A in range(0,positions.shape[0])]
        self.mol.atom = new_atom
        self.mol.unit = "Bohr"
        self.mol.build()
        self.positions = self.mol.atom_coords(unit="Bohr")
        return
    
    def getMultipoleOperators(self,mol=None):
        '''
        Gets the QM multipole operators
        '''
        if mol is None:
            mol = self.mol
        Q = self.multipole.getMultipoleOperators(mol=mol)
        return Q
    
    def createModifiedSCF(self):
        '''
        Creates a modified scf/dft object to perform QM/MM QM+interaction energy calculation.
        '''
        
        # this implementation ionly runs probably with the KS type objects. HF type objects are converted to KS
        self.mf = self.mf.to_ks()
        
        # copy the scf object - methods *should* be modifiable in mf_qmmm without modifying original
        self.mf_qmmm = copy(self.mf)
        self.mf_int = copy(self.mf)
        self.mf_int.xc = "HF"
        self.mf_int.omega = 0.0
        
        # get the multipole moment operators
        self.Q = self.multipole.getMultipoleOperators(mol=self.mf_qmmm.mol)
        
        # get the overlap matrix and its inverse & store these for future use
        self.S = self.mf.mol.intor('int1e_ovlp')
        self.S_inv = inv(self.S)
        
        # get the nuclear charges
        Z = self.mf.mol.atom_charges()
        
        # get N_QM and N_Q
        N_Q = self.Q.shape[0]
        N_QM = len(Z) 
        
        # set up the polarization energy expansion
        if self.int_method == "drf":
            # first get the u_0 term from the nuclear charges: u_0 = Σ_Α Z_A U_1A + (1/2) Σ_Α,Β Z_A U_2AB Z_B 
            u_nuc_ind = np.einsum('B,AB->A',Z,self.U_2[0:N_QM,0:N_QM])
            u_0 = np.einsum('A,A', (self.U_1[0:N_QM] + 0.5*u_nuc_ind) ,Z) 
            # second get the u_1 term: sum of direct plus induction from nuclear charges
            u_1 = self.U_1 + np.einsum('aB,B->a',self.U_2[:,0:N_QM],Z)
            # lastly get the u_2 term: just the bare 
            u_2 = self.U_2
            # store these in the object
            self.u_0 = u_0 
            self.u_1 = u_1
            self.u_2 = u_2
        
        # set up the correction to the one electron Hamiltonian from embedding
        if self.int_method == "drf":
            # first the u_0 term
            #h_int = (u_0/self.mf.mol.nelectron) * self.S
            # the induction term
            h_int = np.einsum('a,anm->nm',u_1,self.Q)
            # finally get the self-energy term
            self.Q_Sinv = np.einsum('ank,km->anm',self.Q,self.S_inv)
            self.u2_Q = np.einsum('ab,bnm->anm',u_2,self.Q)
            h_int = h_int + 0.5*np.einsum('ank,akm->nm',self.Q_Sinv,self.u2_Q)
            # combine the interaction and QM system hamiltonians
            h_qm = self.mf.get_hcore()
            h_qmmm = h_qm + h_int
            self.mf_int.get_hcore = lambda *args : (h_int)
            self.mf_qmmm.get_hcore = lambda *args : (h_qmmm)
        
        # set up corrections to the generator of v_eff, j and k
        if self.int_method == "drf":
            # set-up the DRF part of the J and K matrix calculation
            self.mf_int.get_j = lambda *args,**kwargs  : self.get_j_drf(args[1])
            self.mf_int.get_k = lambda *args,**kwargs  : self.get_k_drf(args[1])
            self.mf_int.get_jk = lambda *args,**kwargs: (self.get_j_drf(args[1]),self.get_k_drf(args[1]))
            # modify the get_veff function the scf_qmmm object used for the SCF calculation
            self.mf_qmmm.get_veff = lambda *args,**kwargs: self.get_veff_drf(self.mf_qmmm,args[1])
            
        # add modifications to the system for the repulsion term
        if self.rep_method == "exch":
            h_exchrep = self.getExchRepHamiltonian()
            h_int = h_int + h_exchrep
            h_qmmm = h_qmmm + h_exchrep
            self.mf_int.get_hcore = lambda *args : (h_int)
            self.mf_qmmm.get_hcore = lambda *args : (h_qmmm)

        return
    
    

    def get_veff_drf(self,mf_obj,dm):
        '''
        The modified 
        '''
        if dm is None:
            dm = mf_obj.make_rdm1()
        # get vxc from scf without added DRF terms
        vxc = self.mf.get_veff(dm=dm)
        vxc_drf = self.mf_int.get_veff(dm=dm)
        #print(dir(vxc_drf))
        ecoul = vxc.ecoul
        exc = vxc.exc 
        vj = vxc.vj + vxc_drf.vj
        vk = vxc.vk + vxc_drf.vk
        vxc = lib.tag_array(vxc+vxc_drf, ecoul=(ecoul+vxc_drf.ecoul), exc=(exc+vxc_drf.exc), vj=vj, vk=vk)

        return vxc
    
    def get_j_drf(self,dm):
        '''
        Generates the coulomb matrix J for a given density matrix dm arising from the DRF 2-e interaction
        '''
        if len(dm.shape)==2:
            q = np.einsum('Aij,ij->A',self.Q,dm)
            return np.einsum('A,Aij->ij',q,self.u2_Q)
        elif len(dm.shape)==3:
            q = np.einsum('Aij,nij->nA',self.Q,dm)
            #j = np.sum(np.einsum('nA,Aij->nij',q,self.UQ),axis=0)
            #return np.array([j,j])
            return np.einsum('nA,Aij->nij',q,self.u2_Q)

    def get_k_drf(self,dm):
        '''
        Generates the exchange matrix K for a given density matrix dm arising from the DRF 2-e interaction
        '''
        if len(dm.shape)==2:
            dmQ = np.einsum('jk,Bkl->Bjl',dm,self.Q)
            return np.einsum('Aij,Ajl->il',self.u2_Q,dmQ)
        elif len(dm.shape)==3:
            dmQ = np.einsum('njk,Bkl->nBjl',dm,self.Q)
            return np.einsum('Aij,nAjl->nil',self.u2_Q,dmQ)
    
  
    def setPolarizationEnergyResp(self,pol_resp):
        if pol_resp["units"] in ["AU","au"]:
            conv_U_0 = 1 
            conv_U_1 = 1 
            conv_U_2 = 1
        else:
            print("Warning: unknown polarization energy units.")
        self.U_0 = pol_resp["U_0"] * conv_U_0
        self.U_1 = pol_resp["U_1"] * conv_U_1
        self.U_2 = pol_resp["U_2"] * conv_U_2
        return

    def setExchRepParameters(self,rep_info,mm_positions):
        '''
        Sets up all the repsulion info for calculating 
        '''
        self.rep_info = rep_info
        self.rep_positions = np.empty((0,3))
        self.rep_types = []
        # get the cut-off distance for adding the interaction
        cutoff = self.rep_info["cutoff"]
        # find all MM sites in the cut off and get the type for adding repulsion
        for A in range(0,mm_positions.shape[0]):
            d_A = np.linalg.norm(self.positions - mm_positions[A,:].reshape((1,3)),axis=1)
            if np.any(d_A<cutoff):
                self.rep_positions = np.vstack((self.rep_positions,mm_positions[A,:]))
                self.rep_types.append(rep_info["MM types"][A])
        
        return 
    
    def getExchRepHamiltonian(self,mol=None):
        '''
        Generates the exchange-repulsion Hamiltonian
        '''
        if mol is None:
            mol = self.mol 
        
        qo_info = self.rep_info["type quasi-orbital info"]
        N_AO = mol.nao
        h_exchrep = np.zeros((N_AO,N_AO))
        for B in range(0,self.rep_positions.shape[0]):
            type_B = self.rep_types[B]
            if qo_info[type_B]["orbital type"] == "sto-ng":
                basis = {"GHOST": qo_info[type_B]["basis"]}
                mol_B = gto.M(atom=[ [ "GHOST",self.rep_positions[B,:] ] ],basis=basis,unit="Bohr")
                mol_comb = mol+mol_B
                N = mol.nbas
                M = mol_comb.nbas
                h_exchrep = h_exchrep + (0.5*qo_info[type_B]["N_eff"]) * mol_comb.intor("cint2e_sph",shls_slice=(0,N,N,M,N,M,0,N)).reshape((N_AO,N_AO))
        return h_exchrep
    
    def getEnergy(self,units_out="AU",return_terms=False):
        '''
        Get the energy of the QM system + QM/MM interaction
        '''
        # run the ground state SCF QM/MM calculation
        self.createModifiedSCF()
        self.mf_qmmm.kernel(dm=self.dm_guess)
        
        # get the energy
        E = self.mf_qmmm.energy_tot()
        
        # do unit conversion as needed
        if units_out in ["AU","au","Hartree","hartree"]:
            conv = 1.0
        elif units_out in ["OpenMM","kJ/mol"]:
            Eh = 2625.4996352210997 # hartree in kJ/mol
            conv = Eh
        
        E = E * conv
        if return_terms:
            return {"QM+int":E,"u_0":self.u_0}
        else:
            return E+self.u_0
