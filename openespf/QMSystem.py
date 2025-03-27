from pyscf import gto, scf, dft, tddft, ao2mo, lib, grad
from pyscf.dft import numint
import numpy as np
from scipy.linalg import sqrtm, inv, solve
from .QMMultipole import QMMultipole
from copy import deepcopy, copy


'''
A wrapper for PySCF methods for calculating QM component of QM/MM energies and forces.
'''

def getDist(x_A_set,x_B,pbc=None):
    if pbc is None:
        dx = (x_A_set - x_B[None,:])
        return np.linalg.norm(dx,axis=1), dx
    else:
        dx = x_A_set - x_B[None,:]
        dx_ni = dx - pbc[None,:]*np.round(dx/pbc[None,:])
        return np.linalg.norm(dx_ni,axis=1), dx_ni

def calculateGradOvlp(mol,A):
    N_AO = mol.nao
    N_bas = mol.nbas
    bas_start,bas_end,ao_start,ao_end = mol.aoslice_by_atom()[A]
    ao_inds = np.arange(ao_start,ao_end)
    ip = mol.intor('int1e_ipovlp',shls_slice=(bas_start,bas_end,0,N_bas))
    gradA_S_tot = np.zeros((3,N_AO,N_AO))
    gradA_S_tot[:,ao_inds,:] -= ip
    gradA_S_tot[:,:,ao_inds] -= np.swapaxes(ip,1,2)
    return gradA_S_tot

class QMSystem:
    def __init__(self,qm_mf,qm_resp=None,int_method="drf",rep_method="exch",multipole_order=0,multipole_method="espf"):
        self.mf = qm_mf.copy()
        self.mol = self.mf.mol
        self.resp = qm_resp
        self.int_method = int_method
        self.rep_method = rep_method
        self.multipole_order = multipole_order
        self.multipole_method = multipole_method
        self.positions = self.mol.atom_coords(unit="Bohr")
        self.multipole = QMMultipole(mol=self.mol,multipole_order=self.multipole_order,multipole_method=self.multipole_method)
        self.dm_guess=None
        self.drf_method = "get_jk"
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
        #self.mf.build(self.mol)
        self.mf.reset()
        self.multipole.reset()
        self.multipole.mol = self.mol
        if self.resp is not None:
            self.resp._scf = self.mf
            self.resp.mol = self.mol
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
        #self.mf = self.mf.to_ks()
        
        # copy the scf object - methods *should* be modifiable in mf_qmmm without modifying original
        self.mf_qmmm = (deepcopy(self.mf))
        self.mf_int = (deepcopy(self.mf))
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
            
        # set up 2-e integrals if using incore
        if self.int_method == "drf" and hasattr(self.mf_qmmm,'incore_anyway'):
            if self.mf_qmmm.incore_anyway:
                eri = self.mf_qmmm.mol.intor('cint2e_sph', aosym='s1')
                eri += np.einsum('Aij,Akl->ijkl',self.Q,self.u2_Q,optimize=True)
                self.mf_qmmm._eri = ao2mo.restore(8, eri, self.mf_qmmm.mol.nao)
        
        # set up corrections to the generator of v_eff, j and k
        if self.int_method == "drf" and self.drf_method == "get_veff":
            # set-up the DRF part of the J and K matrix calculation
            #self.mf_int.get_j = lambda *args,**kwargs  : self.get_j_drf(args[1])
            #self.mf_int.get_k = lambda *args,**kwargs  : self.get_k_drf(args[1])
            #self.mf_int.get_jk = lambda *args,**kwargs: (self.get_j_drf(args[1]),self.get_k_drf(args[1]))
            # modify the get_veff function the scf_qmmm object used for the SCF calculation
            self.mf_qmmm.get_veff = lambda *args,**kwargs: self.get_veff_drf(self.mf_qmmm,args[1],**kwargs)
        elif self.int_method == "drf" and self.drf_method == "get_jk":
            if "HF" in self.mf.__class__.__name__ :
                self.mf_qmmm.get_jk = lambda *args,**kwargs: self.get_jk_mod(*args,**kwargs,vk_scal=1.0)
            elif self.mf_qmmm._numint.libxc.is_hybrid_xc(self.mf.xc):
                omega, alpha, hyb = self.mf_qmmm._numint.rsh_and_hybrid_coeff(self.mf.xc, spin=self.mf.mol.spin)
                # scaling of vk_drf depends on method
                if omega == 0:
                    vk_scal = hyb
                elif alpha == 0 :
                    vk_scal = hyb
                elif hyb == 0 :
                    vk_scal = 1.0/alpha
                else: 
                    vk_scal = 1.0/alpha
                self.mf_qmmm.get_jk = lambda *args,**kwargs: self.get_jk_mod(*args,**kwargs,vk_scal=vk_scal)
            else:
                xc_split = self.mf.xc.split(',') 
                if len(xc_split) == 2:
                    xc_split[0] += "+1.0*HF"
                    xc_new = xc_split[0] + xc_split[1]
                else:
                    xc_new = self.mf.xc + "+1.0*HF"
                #self.mf.xc = xc_new 
                self.mf_qmmm.xc = xc_new
                self.mf_qmmm.get_jk = lambda *args,**kwargs: self.get_jk_mod(*args,**kwargs,vk_scal=None)
            
                
                
            
        # add modifications to the system for the repulsion term
        if self.rep_method == "exch":
            h_exchrep = self.getExchRepHamiltonian()
            h_int = h_int + h_exchrep
            h_qmmm = h_qmmm + h_exchrep
            self.mf_int.get_hcore = lambda *args : (h_int)
            self.mf_qmmm.get_hcore = lambda *args : (h_qmmm)

        return
    
    def createModifiedResp(self):
        '''
        Creates a modified response energy object (e.g. CIS, TDA, TDDFT)
        '''
        if self.resp is None:
            raise Exception("Response object is not set. Cannot create a modified version.")
            
        self.resp_qmmm = self.resp.copy()
        self.resp_qmmm._scf = self.mf_qmmm 
        
        if self.int_method == "drf" and self.drf_method == "get_veff":
            # modify the response object
            singlet = self.resp_qmmm.singlet
            vresp = self.mf.gen_response(singlet=singlet,mo_coeff=self.resp_qmmm._scf.mo_coeff, mo_occ=self.resp_qmmm._scf.mo_occ,hermi=0)
            vresp_drf = self.mf_int.gen_response(singlet=singlet,mo_coeff=self.resp_qmmm._scf.mo_coeff, mo_occ=self.resp_qmmm._scf.mo_occ,hermi=0)
            self.resp_qmmm._scf.gen_response = lambda singlet=singlet,hermi=0: (lambda *args2 : vresp(*args2)+vresp_drf(*args2))
        elif self.int_method == "drf" and self.drf_method == "get_jk":
            # no modification should be needed if the methodf is "get_jk" need to check this
            self.resp_qmmm = self.resp_qmmm
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
        if vxc.vk is not None:
            vk = vxc.vk + vxc_drf.vk
        else:
            vk = vxc_drf.vk
        vxc = lib.tag_array(vxc+vxc_drf, ecoul=(ecoul+vxc_drf.ecoul), exc=(exc+vxc_drf.exc), vj=vj, vk=vk)

        return vxc
    
    def get_j_drf(self,dm):
        '''
        Generates the coulomb matrix J for a given density matrix dm arising from the DRF 2-e interaction
        '''
        if type(dm)==type((None,)) or type(dm)==type([]):
            j = []
            for dm_n in dm:
                q = np.einsum('Aij,ij->A',self.Q,dm_n)
                j.append(np.einsum('A,Aij->ij',q,self.u2_Q))
            if type(dm)==type((None,)):
                return tuple(j)
            elif type(dm)==type([]):
                return j
        elif len(dm.shape)==2:
            q = np.einsum('Aij,ij->A',self.Q,dm)
            return np.einsum('A,Aij->ij',q,self.u2_Q)
        elif len(dm.shape)==3:
            q = np.einsum('Aij,nij->nA',self.Q,dm)
            #j = np.sum(np.einsum('nA,Aij->nij',q,self.UQ),axis=0)
            #return np.array([j,j])
            return np.einsum('nA,Aij->nij',q,self.u2_Q)

    def get_k_drf(self,dm,vk_scal=1.0):
        '''
        Generates the exchange matrix K for a given density matrix dm arising from the DRF 2-e interaction
        '''
        if type(dm)==type((None,)) or type(dm)==type([]):
            k = []
            for dm_n in dm:
                dmQ = np.einsum('jk,Bkl->Bjl',dm_n,self.Q)
                k.append((1.0/vk_scal)*np.einsum('Aij,Ajl->il',self.u2_Q,dmQ))
            if type(dm)==type((None,)):
                return tuple(k)
            elif type(dm)==type([]):
                return k
        if len(dm.shape)==2:
            dmQ = np.einsum('jk,Bkl->Bjl',dm,self.Q)
            return (1.0/vk_scal)*np.einsum('Aij,Ajl->il',self.u2_Q,dmQ)
        elif len(dm.shape)==3:
            dmQ = np.einsum('njk,Bkl->nBjl',dm,self.Q)
            return (1.0/vk_scal)*np.einsum('Aij,nAjl->nil',self.u2_Q,dmQ)
    
    def get_jk_mod(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None,vk_scal=1.0):
        #print({**locals()})
        if mol is None: mol = self.mf_qmmm.mol
        if dm is None: dm = self.mf_qmmm.make_rdm1()
        #print("omega = " , omega, "hyb = ", hyb)
        #print("omega = ",omega,"hyb = ",hyb,"get_j,get_k = ",with_j,with_k)
        #vj, vk = self.mf.get_jk(mol=mol, dm=dm, hermi=hermi, with_j=with_j, with_k=with_k, omega=omega)
        vj, vk = self.mf.get_jk(mol=mol, dm=dm, hermi=hermi, with_j=with_j, with_k=with_k, omega=omega)
        
        if with_j:
            vj += self.get_j_drf(dm)
        if with_k:
            if vk_scal is not None:
                vk += self.get_k_drf(dm,vk_scal=vk_scal)
            else:
                vk = self.get_k_drf(dm)
        
        return vj, vk
        
  
    def setPolarizationEnergyResp(self,pol_resp):
        if pol_resp["units"] in ["AU","au"]:
            conv_U_0 = 1 
            conv_U_1 = 1 
            conv_U_2 = 1
            conv_d = 1 
        else:
            print("Warning: unknown polarization energy units.")
        self.U_0 = pol_resp["U_0"] * conv_U_0
        self.U_1 = pol_resp["U_1"] * conv_U_1
        self.U_2 = pol_resp["U_2"] * conv_U_2
        if "F_0_qm" in pol_resp.keys():
            self.F_0_qm = pol_resp["F_0_qm"] * conv_U_0 * conv_d
            self.F_1_qm = pol_resp["F_1_qm"] * conv_U_1 * conv_d
            self.F_2_qm = pol_resp["F_2_qm"] * conv_U_2 * conv_d
            self.F_0_mm = pol_resp["F_0_mm"] * conv_U_0 * conv_d
            self.F_1_mm = pol_resp["F_1_mm"] * conv_U_1 * conv_d
            self.F_2_mm = pol_resp["F_2_mm"] * conv_U_2 * conv_d
           
        return

    def setExchRepParameters(self,rep_info,mm_positions,pbc=None):
        '''
        Sets up all the repsulion info for calculating 
        '''
        self.rep_info = rep_info
        self.rep_positions = np.empty((0,3))
        self.rep_types = []
        self.rep_atoms = []
        # get the cut-off distance for adding the interaction
        cutoff = self.rep_info["cutoff"]
        # find all MM sites in the cut off and get the type for adding repulsion
        for A in range(0,mm_positions.shape[0]):
            #d_A = np.linalg.norm(self.positions - mm_positions[A,:].reshape((1,3)),axis=1)
            d_A,dx_A = getDist(self.positions,mm_positions[A,:],pbc=pbc)
            if np.any(d_A<cutoff):
                n = np.argmin(d_A)
                rep_position = self.positions[n,:]-dx_A[n,:]
                self.rep_positions = np.vstack((self.rep_positions,rep_position[None,:]))
                self.rep_types.append(rep_info["MM types"][A])
                self.rep_atoms.append(A)
        
        return 
    

    
    def getEnergy(self,units_out="AU",return_terms=False):
        '''
        Get the energy of the QM system + QM/MM interaction
        '''
        self.multipole.reset()
        # run the ground state SCF QM/MM calculation
        self.createModifiedSCF()
        self.mf_qmmm.kernel(dm=self.dm_guess)
        
        # get the energy
        E = self.mf_qmmm.energy_tot()
        
        # get response energy
        if self.resp is not None:
            self.createModifiedResp()
            self.resp_qmmm.kernel()
            E = [E]+[E_n+E for E_n in self.resp_qmmm.e]
        
        # do unit conversion as needed
        if units_out in ["AU","au","Hartree","hartree"]:
            conv = 1.0
        elif units_out in ["OpenMM","kJ/mol"]:
            Eh = 2625.4996352210997 # hartree in kJ/mol
            conv = Eh
        
        #E = E * conv
        if self.resp is None:
            if return_terms:
                return {"QM+int":E* conv,"u_0":self.u_0* conv}
            else:
                return (E+self.u_0)* conv
        else:
            if return_terms:
                return [{"QM+int":E_n* conv,"u_0":self.u_0* conv} for E_n in E]
            else:
                return [(E_n+self.u_0)* conv for E_n in E]

    def createModifiedGradSCF(self):
        '''
        Get the gradient of the QM + int energy with respect to a QM atom A
        '''
        self.grad_mf_qmmm = self.mf_qmmm.nuc_grad_method()
        self.grad_mf = self.mf.nuc_grad_method()
        if not "HF" in self.mf.__class__.__name__:
            if not self.mf.xc == "HF":
                self.grad_mf_qmmm.grid_response=True
                self.grad_mf.grid_response=True
        
        # modification fo hcore - probably gonna abandon this approach
        #hcore_deriv = self.grad_mf.hcore_generator(self.mf_qmmm.mol)
        #hcore_int_deriv = self.hcore_int_generator()
        #self.grad_mf_qmmm.hcore_generator = lambda *args : (lambda atm_id : hcore_deriv(atm_id) + hcore_int_deriv(atm_id))
        
        #self.grad_mf_qmmm.extra_force = lambda *args: self.getExtraForceInt(*args)
        self.grad_mf_qmmm.extra_force = self.getExtraForceInt
        
        
        return 
    
    def getExtraForceInt(self,atom_id,envs):
        '''
        returns the QM/MM interaction component of the forces
        '''
        #print("Extra force for atom:",atom_id)
        dm0 = self.mf_qmmm.make_rdm1()
        # get the number of atomic orbitals
        # make the dm a N_sigma x N_AO x N_AO array so UKS, ROKS and RKS can be treated consistently
        dm0 = np.array(dm0)
        N_AO = dm0.shape[-1]
        if len(dm0.shape)==2:
            dm0 = dm0.reshape((1,N_AO,N_AO))
        dm = np.einsum('snm->nm',dm0)
        F = np.zeros(self.F_0_qm[atom_id,:].shape)
        if self.int_method == "drf":
            # first the terms that do not involve gradients of the charge operators
            # first deal with the zero electron part
            Z = self.mf_qmmm.mol.atom_charges()
            N_QM = len(Z)
            F_ind = np.einsum('xaA,A->xa',self.F_2_qm[atom_id,:,:,0:N_QM],Z)
            F += np.einsum('xA,A->x',self.F_1_qm[atom_id,:,0:N_QM],Z) + np.einsum('xA,A->x',F_ind[:,0:N_QM],0.5*Z)
            # 1e induction term
            F += np.einsum('a,xa->x',self.av_Q,F_ind+self.F_1_qm[atom_id,:,:])
            # 1e self term + 2e term
            F += 0.5 * np.einsum('xab,ab->x',self.F_2_qm[atom_id,:,:,:],self.av_QQ_1e+self.av_QQ_2e)
            #print("") 
            #F_nogradQ = F+0
            #print("F_nogradQ=",F)
            
            # the multipole operator gradient terms
            grad_Q = 1.0*self.multipole.getGradMultipoleOperators(atom_id)
            av_grad_Q = (np.einsum('xanm,nm->xa',grad_Q,dm,optimize=True))
            #av_grad_Q = 0.5*(np.einsum('xanm,nm->xa',grad_Q,dm)+np.einsum('xanm,mn->xa',grad_Q,dm))
            #print(av_grad_Q)
            # u_1 term
            F -= np.einsum('xa,a->x',av_grad_Q,self.u_1)
            # self term
            dm0_grad_Q = np.einsum('snm,xanl->sxaml',dm0,grad_Q,optimize=True)
            dm_grad_Q = np.einsum('sxaml->xaml',dm0_grad_Q,optimize=True)
            dm_grad_Q_Sinv = np.einsum('xaml,lk->xamk',dm_grad_Q,self.S_inv,optimize=True)
            F -= np.einsum('xamk,akm->x',dm_grad_Q_Sinv,self.u2_Q,optimize=True) # no factor of 1/2 because of dQ/dx Q + Q dQ/dx
            # grad S inv term
            grad_S = calculateGradOvlp(self.mol,atom_id)
            grad_Sinv = np.einsum('xnl,lm->xnm',grad_S,self.S_inv,optimize=True)
            grad_Sinv = -np.einsum('nl,xlm->xnm',self.S_inv,grad_Sinv,optimize=True)
            grad_Sinv_Q = np.einsum('xnl,alm->xanm',grad_Sinv,self.Q,optimize=True)
            A = np.einsum('anl,xalm->xnm',self.u2_Q,grad_Sinv_Q,optimize=True)
            F -= 0.5*np.einsum('xnm,nm->x',A,dm)
            #print(np.einsum('xamk,akm->x',dm_grad_Q_Sinv,self.u2_Q))
            # the 2e terms
            grad_QQ_2e_c = np.einsum('xa,b->xab',av_grad_Q,self.av_Q,optimize=True)
            grad_QQ_2e_c += np.einsum('xa,b->xba',av_grad_Q,self.av_Q,optimize=True)
            dm0_Q = self.dm_Q
            grad_QQ_2e_x = np.einsum('sxanm,sbmn->xab',dm0_grad_Q,dm0_Q,optimize=True)
            grad_QQ_2e_x += np.einsum('xab->xba',grad_QQ_2e_x,optimize=True)
            if dm0.shape[0]==1:
                grad_QQ_2e_x *= 0.5 
            F -= 0.5*np.einsum('xab,ab->x',grad_QQ_2e_c - grad_QQ_2e_x , self.u_2)
            #print("F_gradQ=",F-F_nogradQ)
        
        if self.rep_method == "exch":
            F += self.getExchRepForceQM(dm,atom_id)
        
        F -= self.grad_mf.extra_force(atom_id,envs)
        #F *= 0
        return -F
    
    def hcore_int_generator(self):
        '''
        Unused - should probably delete
        '''
        N_AO = self.mol.nao 

        def h_int_deriv(atm_id):
            h1 = np.zeros((3,N_AO,N_AO))
            # add the DRF terms
            if self.int_method == "drf":
                grad_Q = self.multipole.getGradMultipoleOperators(atm_id)
                du_0,du_1,du_2 = self.getGradQMPolExp(atm_id)
                # first the linear induction term
                h1 += np.einsum('xanm,a->xnm',grad_Q,self.u_1)
                h1 += np.einsum('anm,xa->xnm',self.Q,du_1)
                
                # second the self energy term
                grad_S = calculateGradOvlp(self.mol,atm_id)
                grad_Sinv = np.einsum('xnl,lm->xnm',grad_S,self.S_inv)
                grad_Sinv = -np.einsum('nl,xlm->xnm',self.S_inv,grad_Sinv)
                grad_Q_Sinv = np.einsum('xanm,mp->xanp',grad_Q,self.S_inv)
                A = 0.5 * np.einsum('xanm,amp->xnp',grad_Q_Sinv,self.u2_Q)
                h1 += A + A.transpose((0,2,1))
                #h1 += 0.5 * np.einsum('xanm,amp->xnp',grad_Q_Sinv,self.u2_Q)
                #h1 += 0.5 * np.einsum('anm,xapm->xnp',self.u2_Q,grad_Q_Sinv)
                grad_Sinv_Q = np.einsum('xnl,alm->xanm',grad_Sinv,self.Q)
                h1 += 0.5 * np.einsum('anl,xalm->xnm',self.u2_Q,grad_Sinv_Q)
                du_2_Q = np.einsum('xab,bnm->xanm',du_2,self.Q)
                h1 += 0.5 * np.einsum('anl,xalm->xnm',self.Q_Sinv,du_2_Q)
                #h1 += self.S.reshape((1,N_AO,N_AO)) * du_0.reshape(3,1,1)
            
            if self.rep_method == "exch":
                h1 += np.zeros((3,N_AO,N_AO))
            
            print(np.max(np.abs(h1-h1.transpose((0,2,1)) )))
            return h1 
        
        return h_int_deriv
    
    def getForces(self,units_out="AU",return_terms=False):
        '''
        
        '''
        
        # create a modified SCF/DFT gradient object
        self.createModifiedGradSCF()
        
        F = -self.grad_mf_qmmm.kernel()
        #f_0 = self.getForceQMu0()
        f_0 = np.zeros(F.shape)
        #print(F)
        # do unit conversion as needed
        if units_out in ["AU","au","Hartree","hartree"]:
            conv = 1.0
        elif units_out in ["OpenMM","kJ/mol"]:
            Eh = 2625.4996352210997 # hartree in kJ/mol
            a0 = 0.52917721092e-1 # bohr in nm
            conv = Eh/a0
        
        #E = E * conv
        if self.resp is None:
            if return_terms:
                return {"QM+int":F* conv,"u_0":f_0* conv}
            else:
                return (F+f_0)* conv
        else:
            if return_terms:
                return [{"QM+int":F_n* conv,"u_0":f_0* conv} for F_n in F]
            else:
                return [(F_n+f_0)* conv for F_n in F]
    
    def getGradQMPolExp(self,atm_id):
        # set up the gradient of the polarization energy expansion 
        Z = self.mf.mol.atom_charges()
        N_QM = len(Z)
        # first get the u_0 term from the nuclear charges: u_0 = Σ_Α Z_A U_1A + (1/2) Σ_Α,Β Z_A U_2AB Z_B 
        du_nuc_ind = -np.einsum('B,xAB->xA',Z,self.F_2_qm[atm_id,:,0:N_QM,0:N_QM])
        du_0 = np.einsum('xA,A->x', (-self.F_1_qm[atm_id,:,0:N_QM] + 0.5*du_nuc_ind) ,Z) 
        # second get the u_1 term: sum of direct plus induction from nuclear charges
        du_1 = -self.F_1_qm[atm_id,:,:] - np.einsum('xaB,B->xa',self.F_2_qm[atm_id,:,:,0:N_QM],Z)
        # lastly get the u_2 term: just the bare U_2
        du_2 = -self.F_2_qm[atm_id,:,:,:]
        
        return du_0, du_1, du_2
    
    def getGradQMu0(self,atm_id):
        # set up the gradient of the polarization energy expansion 
        Z = self.mf.mol.atom_charges()
        N_QM = len(Z)
        # first get the u_0 term from the nuclear charges: u_0 = Σ_Α Z_A U_1A + (1/2) Σ_Α,Β Z_A U_2AB Z_B 
        du_nuc_ind = -np.einsum('B,xAB->xA',Z,self.F_2_qm[atm_id,:,0:N_QM,0:N_QM])
        du_0 = np.einsum('xA,A->x', (-self.F_1_qm[atm_id,:,0:N_QM] + 0.5*du_nuc_ind) ,Z) 
        
        return du_0
    
    def getForceQMu0(self,atm_list=None):
        if atm_list is None:
            f_0 = np.zeros(self.positions.shape)
            for A in range(0,self.positions.shape[0]):
                f_0[A,:] = -self.getGradQMu0(A)
            return f_0
    
    def getForcesMM(self):
        '''
        Returns the force on MM atoms
        '''
        F = np.zeros(self.F_0_mm.shape)
        if self.int_method == "drf":
            # first deal with the zero electron part
            Z = self.mf_qmmm.mol.atom_charges()
            N_QM = len(Z)
            F_ind = np.einsum('kxaA,A->kxa',self.F_2_mm[:,:,:,0:N_QM],Z)
            F += np.einsum('kxA,A->kx',self.F_1_mm[:,:,0:N_QM],Z) + np.einsum('kxA,A->kx',F_ind[:,:,0:N_QM],0.5*Z)

            # next the induction 1e terms
            av_Q, av_QQ_1e, av_QQ_2e = self.getMeanMultipoles()
            # 1e induction term
            F += np.einsum('a,kxa',av_Q,F_ind+self.F_1_mm)
            # 1e self term
            F += 0.5 * np.einsum('kxab,ab->kx',self.F_2_mm,av_QQ_1e)
            # 2e term
            F += 0.5 * np.einsum('kxab,ab->kx',self.F_2_mm,av_QQ_2e)
            
            # save the average charges
            self.av_Q = av_Q
            self.av_QQ_1e = av_QQ_1e
            self.av_QQ_2e = av_QQ_2e
        
        if self.rep_method == "exch":
            dm = self.mf_qmmm.make_rdm1()
            if len(dm.shape)>2:
                dm = dm[0,:,:] + dm[1,:,:]
            for k,B in enumerate(self.rep_atoms): # loop over atoms in the cutoff radius
                F_B = self.getExchRepForceMM(dm,k)
                F[B,:] += F_B
                
                
        
        return F
    
    def getMeanMultipoles(self,dm=None,dm_Q=None,comb_coulexch=True):
        '''
        Gets the mean multipoles Σ_i<Q_a,i> Σ_i<Q_a,i Q_b,i>, Σ_i=/=j <Q_a,i Q_b,j>
        '''
        if dm is None:
            dm = self.mf_qmmm.make_rdm1()
        # get the number of atomic orbitals
        # make the dm a N_sigma x N_AO x N_AO array so UKS, ROKS and RKS can be treated consistently
        dm = np.array(dm)
        N_AO = dm.shape[-1]
        if len(dm.shape)==2:
            dm = dm.reshape((1,N_AO,N_AO))
        # make dm_Q tensor N_Q x N_AO x N_AO
        if dm_Q is None:
            dm_Q = np.einsum('sln,alm->sanm',dm,self.Q)
        # get the average Q
        av_Q = np.einsum('snm,anm->a',dm,self.Q)
        # get the average Q_a Q_b from the 1e density [S^-1 Q]_amn = [Q S^-1]_anm because S and Q and symmetric
        av_QQ_1e = np.einsum('sanm,bnm->ab',dm_Q,self.Q_Sinv)
        # get the 2e coulomb part
        av_QQ_2e_c = np.einsum('a,b->ab',av_Q,av_Q)
        # get the 2e exchange part
        av_QQ_2e_x = np.einsum('sanm,sbmn->ab',dm_Q,dm_Q)
        if dm.shape[0] == 1:
            av_QQ_2e_x = 0.5 * av_QQ_2e_x
        av_QQ_2e = av_QQ_2e_c - av_QQ_2e_x
        self.dm_Q = dm_Q
        if comb_coulexch:
            return av_Q, av_QQ_1e, av_QQ_2e
        else:
            return av_Q, av_QQ_1e, av_QQ_2e_c, av_QQ_2e_x
    
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
    
    def getExchRepDerivMMHamilontian(self,B,mol=None):
        '''
        Generates the derivative of the exchange-repulsion Hamiltonian for a given MM atom A
        '''
        if mol is None:
            mol = self.mol 
        
        qo_info = self.rep_info["type quasi-orbital info"]
        N_AO = mol.nao
        grad_h_exchrep = np.zeros((3,N_AO,N_AO))
        type_B = self.rep_types[B]
        if qo_info[type_B]["orbital type"] == "sto-ng":
            basis = {"GHOST": qo_info[type_B]["basis"]}
            mol_B = gto.M(atom=[ [ "GHOST",self.rep_positions[B,:] ] ],basis=basis,unit="Bohr")
            mol_comb = mol+mol_B
            N = mol.nbas
            M = mol_comb.nbas
            grad_h_exchrep +=  (-0.5*qo_info[type_B]["N_eff"]) * mol_comb.intor("cint2e_ip1_sph",shls_slice=(N,M,0,N,0,N,N,M)).reshape((3,N_AO,N_AO))
            
            #for alpha in range(0,3):
            #    grad_h_exchrep[alpha,:,:] = grad_h_exchrep[alpha,:,:] + grad_h_exchrep[alpha,:,:].T
            
            grad_h_exchrep += grad_h_exchrep.swapaxes(1, 2)

        return grad_h_exchrep
    
    def getExchRepDerivQMHamilontian(self,A,mol=None):
        '''
        Generates the derivative of the exchange-repulsion Hamiltonian for a given QM atom A
        '''
        if mol is None:
            mol = self.mol 
        bas_start_A, bas_end_A, ao_start_A, ao_end_A = mol.aoslice_by_atom()[A]
        N_AO_A = ao_end_A - ao_start_A
        N_AO = mol.nao
        
        qo_info = self.rep_info["type quasi-orbital info"]
        grad_h_exchrep = np.zeros((3,N_AO,N_AO))
        for k,B in enumerate(self.rep_atoms): # loop over atoms in the cutoff distance
            type_B = self.rep_types[k]
            if qo_info[type_B]["orbital type"] == "sto-ng":
                basis = {"GHOST": qo_info[type_B]["basis"]}
                mol_B = gto.M(atom=[ [ "GHOST",self.rep_positions[k,:] ] ],basis=basis,unit="Bohr")
                mol_comb = mol+mol_B
                N = mol.nbas
                M = mol_comb.nbas
                grad_h_exchrep[:,ao_start_A:ao_end_A,:] +=  (-0.5*qo_info[type_B]["N_eff"]) * mol_comb.intor("cint2e_ip1_sph",shls_slice=(bas_start_A,bas_end_A,N,M,N,M,0,N)).reshape((3,N_AO_A,N_AO))
                #grad_h_exchrep[:,ao_start_A:ao_end_A,:] +=  (-0.5*qo_info[type_B]["N_eff"]) * mol_comb.intor("int2e_ip1",shls_slice=(bas_start_A,bas_end_A,N,M,N,M,0,N)).reshape((3,N_AO_A,N_AO))

                
        #for alpha in range(0,3):
        #    grad_h_exchrep[alpha,:,:] = grad_h_exchrep[alpha,:,:] + grad_h_exchrep[alpha,:,:].T
        
        grad_h_exchrep += grad_h_exchrep.swapaxes(1, 2)
        
        return grad_h_exchrep
    
    def getExchRepForceQM(self,dm,A,mol=None):
        if mol is None:
            mol = self.mol
        grad_h_exchrep_A = self.getExchRepDerivQMHamilontian(A,mol=mol)
        F = - np.einsum('xnm,nm',grad_h_exchrep_A,dm,optimize=True)
        return F
    
    def getExchRepForceMM(self,dm,B,mol=None):
        if mol is None:
            mol = self.mol
        grad_h_exchrep_B = self.getExchRepDerivMMHamilontian(B,mol=mol)
        F = - np.einsum('xnm,nm',grad_h_exchrep_B,dm,optimize=True)
        return F
    
    def getInteractionEnergyDecomposition(self,print_decomp=False):
        
        
        int_energies={}
        # get mean multipoles
        if self.int_method == "drf":
            av_Q, av_QQ_1e, av_QQ_2e_c, av_QQ_2e_x = self.getMeanMultipoles(comb_coulexch=False)
            av_Q_av_Q = np.outer(av_Q,av_Q)
            dQ_dQ_1e =  av_QQ_1e - av_Q_av_Q
            Z = np.zeros(av_Q.shape)
            N_Q = len(Z)
            N_QM = self.mf_qmmm.mol.natm
            Z[0:N_QM] = self.mf_qmmm.mol.atom_charges()
            
            av_q = av_Q + Z
            av_qq = av_QQ_1e + av_QQ_2e_c - av_QQ_2e_x + np.outer(av_Q,Z) + np.outer(Z,av_Q) + np.outer(Z,Z)
            av_qq_mf = av_QQ_2e_c + np.outer(av_Q,Z) + np.outer(Z,av_Q) + np.outer(Z,Z) 
            av_qq_fluct = av_QQ_1e - av_QQ_2e_x 
            # atom-wise contributions to the electrostatic energy (from perm+MM induced dipoles)
            E_static = av_q * self.U_1
            E_static_atm = np.sum(E_static.reshape(int(N_Q/N_QM),N_QM),axis=0)
            # atomwise polarisation energy
            E_pol = 0.25*np.einsum('ab,ab->a',av_qq,self.U_2) + 0.25*np.einsum('ab,ab->b',av_qq,self.U_2) 
            E_pol_atm = np.sum(E_pol.reshape(int(N_Q/N_QM),N_QM),axis=0)
            if print_decomp : print("Electrostatic embedding energy (E_A = U_1A <q_A>) [AU]:")
            if print_decomp : print(E_static_atm)
            if print_decomp : print("Total electrostatic embedding energy [AU]:",np.sum(E_static))
            if print_decomp : print("Polarization embedding energy (E_A = sum_B E_AB = (1/2) sum_B <q_A q_B>U_2AB)[AU]:")
            if print_decomp : print(E_pol_atm)
            if print_decomp : print("Total polarization embedding energy [AU]:",np.sum(E_pol))
            # mean field contribution to polarization energy
            E_pol_mf = 0.25*np.einsum('ab,ab->a',av_qq_mf,self.U_2) + 0.25*np.einsum('ab,ab->b',av_qq_mf,self.U_2) 
            E_pol_mf_atm = np.sum(E_pol_mf.reshape(int(N_Q/N_QM),N_QM),axis=0)
            if print_decomp : print("E_pol = E_pol_mf + E_pol_fluct")
            if print_decomp : print("Mean-field polarization embedding energy [AU]:")
            if print_decomp : print(E_pol_mf_atm)
            if print_decomp : print("Total mean-field polarization embedding energy [AU]:",np.sum(E_pol_mf))
            # fluctuation field contribution to polarization energy
            E_pol_fluct = 0.25*np.einsum('ab,ab->a',av_qq_fluct,self.U_2) + 0.25*np.einsum('ab,ab->b',av_qq_fluct,self.U_2) 
            E_pol_fluct_atm = np.sum(E_pol_fluct.reshape(int(N_Q/N_QM),N_QM),axis=0)
            if print_decomp : print("Fluctuation polarization embedding energy [AU]:")
            if print_decomp : print(E_pol_fluct_atm)
            if print_decomp : print("Total fluctuation polarization embedding energy [AU]:",np.sum(E_pol_fluct))
            # exchange contribution to polarisation energy
            E_pol_x = 0.25*np.einsum('ab,ab->a',-av_QQ_2e_x,self.U_2) + 0.25*np.einsum('ab,ab->b',-av_QQ_2e_x,self.U_2) 
            E_pol_x_atm = np.sum(E_pol_x.reshape(int(N_Q/N_QM),N_QM),axis=0)
            if print_decomp : print("Exchange polarization embedding energy [AU]:")
            if print_decomp : print(E_pol_x_atm)
            if print_decomp : print("Total exchange polarization embedding energy [AU]:",np.sum(E_pol_x))
            # 1e fluct contribution to polarisation energy
            E_pol_1efluct = 0.25*np.einsum('ab,ab->a',av_QQ_1e,self.U_2) + 0.25*np.einsum('ab,ab->b',av_QQ_1e,self.U_2) 
            E_pol_1efluct_atm = np.sum(E_pol_1efluct.reshape(int(N_Q/N_QM),N_QM),axis=0)
            if print_decomp : print("1e fluct polarization embedding energy [AU]:")
            if print_decomp : print(E_pol_1efluct_atm)
            if print_decomp : print("Total 1e fluct polarization embedding energy [AU]:",np.sum(E_pol_1efluct))
            
            if print_decomp : print("Total DRF energy [AU]:",np.sum(E_pol)+np.sum(E_static))
            int_energies["static"] = E_static_atm
            int_energies["pol"] = E_pol_atm
            int_energies["pol_mf"]=E_pol_mf_atm
            int_energies["pol_fluct"]=E_pol_fluct_atm
            int_energies["pol_xfluct"]=E_pol_x_atm
            int_energies["pol_1efluct"]=E_pol_1efluct_atm
            
            
        if self.rep_method =="exch":
            h_exchrep = self.getExchRepHamiltonian()
            dm = np.array(self.mf_qmmm.make_rdm1())
            if len(dm.shape)>2:
                dm = dm[0,:,:] + dm[1,:,:]
            E_rep = np.einsum('nm,nm',dm,h_exchrep)
            
            E_rep_atm = np.zeros((self.mf_qmmm.mol.natm,))
            for A in range(0,self.mf_qmmm.mol.natm):
                bas_start,bas_end,ao_start,ao_end = self.mf_qmmm.mol.aoslice_by_atom()[A]
                h_exchrep_A = np.zeros(h_exchrep.shape)
                h_exchrep_A[ao_start:ao_end,:] += 0.5 * h_exchrep[ao_start:ao_end,:]
                h_exchrep_A[:,ao_start:ao_end] += 0.5 * h_exchrep[:,ao_start:ao_end]
                E_rep_atm[A] = np.einsum('nm,nm',dm,h_exchrep_A)
            if print_decomp : print("Repulsion energy (Mulliken-style atom-wise decomposition) [AU]:",E_rep_atm)
            if print_decomp : print("Total repulsion energy [AU]:",E_rep)
            int_energies["rep"]=E_rep_atm
            
            
            
        return int_energies
        
