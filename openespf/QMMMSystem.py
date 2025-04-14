'''
The QMMMSystem object handles all energy+force evaluations for the full QM/MM system.
'''

from .MMSystem import MMSystem
from .QMSystem import QMSystem
from pyscf import gto
from scipy.optimize import fsolve
import numpy as np
from timeit import default_timer as timer

class QMMMSystem:
    '''
    The QMMMSystem wraps all of the methods for calculating energy and forces with the ESPF-DRF, ESPF-MFPol and ESPF-EE 
    methods. 

    All quantities are in atomic units.
    '''
    
    
    def __init__(self,mm_simulation,qm_scf,qm_resp=None,int_method="drf",rep_method="exch",multipole_order=0,multipole_method="espf"):
        
        # interaction methods
        self.int_method = int_method
        self.rep_method = rep_method 
        self.multipole_order = multipole_order
        self.multipole_method = multipole_method
        
        
        
        # set up the QM system
        self.qm_system = QMSystem(qm_scf,qm_resp=qm_resp,int_method=self.int_method,rep_method=self.rep_method,multipole_order=self.multipole_order,multipole_method=self.multipole_method)
        self.qm_positions = self.qm_system.getPositions()
        
        # set up the MM system
        self.mm_system = MMSystem(mm_simulation)
        self.mm_positions = self.mm_system.getPositions()
        self.mm_system.setQMPositions(self.qm_positions)
        self.residues = self.getMMResidues()
        
        # printing
        self.print_info = False
        
        # uses PBC?
        self.pbc_dims = self.mm_system.getPBC()
        
        # resp state properties/forces default value
        self.resp_states = [1]
        
        return
    
    
    def setPositions(self,qm_positions=None,mm_positions=None,qm_unit="Bohr",mm_unit="Bohr"):
        '''
        Sets positions of QM and MM atoms in the QMMMSystem. 
        The QMMMSystem uses atomic units as its default internal units.
        '''
        if not mm_positions is None:
            if mm_unit in ["Bohr","bohr","AU","au"]:
                mm_conv = 1.0
            elif mm_unit in ["nanometer"]:
                mm_conv = 1.0/0.52917721092e-1
            elif mm_unit in ["Angstrom","Ang","A","angstrom","ang"]:
                mm_conv = 1.0/0.52917721092e0
            self.mm_system.setPositions(mm_positions,units_in=mm_unit)
            self.mm_positions = self.mm_system.getPositions()
        
        if not qm_positions is None:
            if qm_unit in ["Bohr","bohr","AU","au"]:
                qm_conv = 1.0 
            elif qm_unit in ["nanometer"]:
                qm_conv = 1.0/0.52917721092e-1
            elif qm_unit in ["Angstrom","Ang","A","angstrom","ang"]:
                qm_conv = 1.0/0.52917721092e0
            
            self.qm_system.setPositions(qm_positions * qm_conv)
            self.qm_positions = self.qm_system.getPositions()
        
        return
    
    def getPositions(self,enforce_pbc=False):
        if not enforce_pbc:
            return self.qm_positions, self.mm_positions
        else:
            x_qm = self.enforcePBC(self.qm_positions)
            x_mm = np.zeros(self.mm_positions.shape)
            for res in self.residues:
                x_mm[res,:] = self.enforcePBC(self.mm_positions[res,:])
            return x_qm, x_mm
    
    def getEnergy(self,return_terms=False):
        '''
        
        '''
        
        
        # get the MM system energy terms
        if self.int_method in ["drf","mf"] :
            # get a dictionary of polarization energy resps
            start = timer()
            pol_resp = self.mm_system.getPolarizationEnergyResp(self.qm_positions,self.multipole_order,position_units="Bohr")
            end = timer()
            if self.print_info : print("Pol resp time:",end-start,"s")
            start = end
            # set the polarization response 
            self.qm_system.setPolarizationEnergyResp(pol_resp)
        
        if self.rep_method == "exch":
            # set up the repulsion
            self.qm_system.setExchRepParameters(self.rep_info,self.mm_positions,pbc=self.pbc_dims)
        
        
        
        # set up a dictionary for decomposition of the energy terms
        energy_terms = {} 
        # get the MM energy
        if self.int_method == "drf":
            energy_terms["mm electrostatics"] = pol_resp["U_0"]
        else:
            energy_terms["mm electrostatics"] = self.mm_system.getElectrostaticEnergy()
        start = timer()
        energy_terms["mm remainder"] = self.mm_system.getEnergy() - energy_terms["mm electrostatics"]
        end = timer()
        if self.print_info : print("MM energy time:",end-start,"s")
        # get the QM + interaction energy 
        # TODO - modify to deal with multiple energies from excited states possibly return a list of dictionaries, one for each state
        start = timer()
        qm_energy_terms = self.qm_system.getEnergy(return_terms=True)
        
        if type(qm_energy_terms) is type([]):
            energy_terms = [{ **energy_terms, **state_energy_terms} for state_energy_terms in qm_energy_terms]
        else:
            energy_terms = { **energy_terms, **qm_energy_terms}
        end = timer()
        if self.print_info : print("QM energy time:",end-start,"s")
        
        # return the energy
        if return_terms:
            return energy_terms
        else: 
            if type(energy_terms) is not type([]):
                return np.sum(np.array([energy_terms[k] for k in list(energy_terms.keys())]))
            else:
                return np.array([np.sum(np.array([state_energy_terms[k] for k in list(state_energy_terms.keys())])) for state_energy_terms in energy_terms])
    
    def getEnergyForces(self,return_terms=False):
        '''
        
        '''
        
        
        # get the MM system energy terms
        if self.int_method in ["drf","mf"] :
            # get a dictionary of polarization energy resps
            start = timer()
            pol_resp = self.mm_system.getPolarizationEnergyForceResp(self.qm_positions,self.multipole_order,position_units="Bohr")
            end = timer()
            if self.print_info : print("Pol resp time:",end-start,"s")
            start = end
            # set the polarization response 
            self.qm_system.setPolarizationEnergyResp(pol_resp)
        
        if self.rep_method == "exch":
            # set up the repulsion
            self.qm_system.setExchRepParameters(self.rep_info,self.mm_positions,pbc=self.pbc_dims)
        
        # set up a dictionary for decomposition of the energy terms
        energy_terms = {} 
        force_terms_mm = {}
        force_terms_qm = {}
        # get the MM energy
        if self.int_method == "drf":
            energy_terms["mm electrostatics"] = pol_resp["U_0"]
            force_terms_mm["mm electrostatics"] = pol_resp["F_0_mm"]
            force_terms_qm["mm electrostatics"] = pol_resp["F_0_qm"]
        else:
            energy_terms["mm electrostatics"],force_terms_mm["mm electrostatics"] = self.mm_system.getElectrostaticEnergyForces()
            force_terms_qm["mm electrostatics"] = np.zeros(self.qm_positions.shape)
        start = timer()
        #E_mm,F_mm = self.mm_system.getEnergyForces(terms="remainder")
        E_mm,F_mm = self.mm_system.getEnergyForces()
        energy_terms["mm remainder"] = E_mm - energy_terms["mm electrostatics"]
        force_terms_mm["mm remainder"] = F_mm - force_terms_mm["mm electrostatics"]
        end = timer()
        if self.print_info : print("MM energy time:",end-start,"s")
        # get the QM + interaction energy 
        # TODO - modify to deal with multiple energies from excited states possibly return a list of dictionaries, one for each state
        start = timer()
        qm_energy_terms = self.qm_system.getEnergy(return_terms=True)
        if self.print_info : print("QM energy time:",timer()-start,"s")
        if type(qm_energy_terms) is type([]):
            energy_terms = [{ **energy_terms, **state_energy_terms} for state_energy_terms in qm_energy_terms]
        else:
            energy_terms = { **energy_terms, **qm_energy_terms}
        
        start1 = timer()
        force_terms_mm["QM+int"] = self.qm_system.getForcesMM()
        if self.print_info : print("MM QM/MM force time:",timer()-start1,"s")
        start1 = timer()
        F_qm = self.qm_system.getForces(return_terms=True)
        if self.print_info : print("QM QM/MM qm-force time:",timer()-start1,"s")
        start1 = timer()
        if self.mm_system.resp_mode_force == "linear":
            q_0,qq_0 = self.qm_system.getMeanTotalMultipoles()
            F_qm_c, F_mm_c = self.getClassicalIntForce(qq_0,q_0,pol_resp)
            F_qm["QM+int"] += F_qm_c
            force_terms_mm["QM+int"] += F_mm_c
            F_qm_c_0 = F_qm_c+0.
            F_mm_c_0 = F_mm_c+0.
            self.qm_system.F_int += F_qm_c_0
        
        if self.print_info : print("QM QM/MM classical-force time:",timer()-start1,"s")
        
        

        if type(F_qm) is type([]):
            force_terms_qm = [{ **force_terms_qm, **F_n_qm} for F_n_qm in F_qm]
        else:
            force_terms_qm = { **force_terms_qm, **F_qm}
        end = timer()
        
        if self.print_info : print("QM energy/force time:",end-start,"s")
        
        if self.qm_system.resp is not None:
            f_resp_qm = np.zeros((len(self.resp_states),self.qm_positions.shape[0],3))
            f_resp_mm = np.zeros((len(self.resp_states),self.mm_positions.shape[0],3))
            for n,state in enumerate(self.resp_states):
                start1 = timer()
                f_resp_qm[n,:,:] = self.qm_system.getForcesRespQM(state=state)
                if self.print_info : print("QM QM/MM resp force time:",timer()-start1,"s")
                start1 = timer()
                f_resp_mm[n,:,:] = self.qm_system.getForcesRespMM()
                if self.print_info : print("MM QM/MM resp force time:",timer()-start1,"s")
                
                if self.mm_system.resp_mode_force == "linear":
                    q,qq = self.qm_system.getMeanTotalMultipolesResp()
                    q+=q_0
                    qq+=qq_0
                    F_qm_c, F_mm_c = self.getClassicalIntForce(qq,q,pol_resp)
                    f_resp_qm[n,:,:] += F_qm_c-F_qm_c_0 #+ force_terms_qm["QM+int"] -  F_qm_c_0
                    f_resp_mm[n,:,:] += F_mm_c-F_mm_c_0 #- force_terms_mm["QM+int"] # + force_terms_mm["QM+int"] #+ F_mm_c_0
                    
                self.qm_system.resetRespForce()
        
        
        
        
        # return the energy
        if return_terms:
            return energy_terms, force_terms_qm, force_terms_mm
        else: 
            f_tot_qm = np.sum([force_terms_qm[k] for k in list(force_terms_qm.keys())],axis=0)
            f_tot_mm = np.sum([force_terms_mm[k] for k in list(force_terms_mm.keys())],axis=0)
            if type(energy_terms) is not type([]):
                return np.sum(np.array([energy_terms[k] for k in list(energy_terms.keys())])), f_tot_qm, f_tot_mm
            if self.qm_system.resp is not None:
                f_resp_qm += f_tot_qm[None,:,:] - force_terms_qm["QM+int"]
                f_resp_mm += f_tot_mm[None,:,:] 
                
                return np.array([np.sum(np.array([state_energy_terms[k] for k in list(state_energy_terms.keys())])) for state_energy_terms in energy_terms]), f_tot_qm, f_tot_mm , f_resp_qm, f_resp_mm
            
    
    def setupExchRep(self,atom_type_info,mm_types,cutoff=12.0,setup_info=None):
        '''
        Sets up Exchange repsulion parameters for each MM atom
        '''
        if setup_info is None:
            setup_info={"quasi-orbital type":"sto-3g","fit exponential":True,"exp fit scal":0.75}
        
        
        # sets up repsulion information for each atom type
        type_quasiorb_info = []
        for i in range(0,len(atom_type_info)):
            type_quasiorb_info.append(self.getQuasiOrbitalInfo(atom_type_info[i],setup_info))
        
        # set up the rep_info dictionary
        # This is the information needed to set up the exchange repulsion
        # type quasi-orbital info contains N_eff, basis expansion in pyscf format and orbital type
        # MM types gives the atom type index for each MM atom, cutoff is the cut-off for the interaction in AU
        self.rep_info = {"type quasi-orbital info":type_quasiorb_info,"MM types":mm_types,"cutoff":cutoff}
        
        return
    
    def getQuasiOrbitalInfo(self,atom_type_info,setup_info):
        '''
        Gets quasi orbital info from atom type info
        '''
        
        # set up the reference basis function STO-nG type
        if setup_info["quasi-orbital type"] in ["sto-3g","sto-4g","sto-6g"]:
            orb_type = "sto-ng"
            mol_H = gto.M(atom="H 0 0 0",basis=setup_info["quasi-orbital type"],spin=1)
            sto_nG_info = np.array(mol_H._basis["H"][0][1:])
            beta_0 = sto_nG_info[:,0]
            c_0 = sto_nG_info[:,1]
            norm = 0.
            for i in range(0,len(c_0)):
                norm = norm + np.sum( (4.0*beta_0[i]*beta_0/((beta_0[i]+beta_0)**2))**(0.75) * c_0[i]*c_0)
            c_0 = c_0 / np.sqrt(norm)
            phi_nG = lambda r,gamma : (gamma**(1.5)) * np.sum(c_0 * (2.0*beta_0/np.pi)**(0.75) *np.exp(-(gamma * gamma) * beta_0 * r * r))
        
        # Because STO-nG may not describe wavefunction well at R_dens, first fit an exponential
        if setup_info["fit exponential"]:
            
            #func = lambda beta : (np.log(rho_0) -  np.log(self.N_val_MM[B]*(np.abs(beta/np.pi)**(3/2) ))+(-np.abs(beta)*R_0*R_0))
            R_0 = 1.0 * atom_type_info["R_dens"]
            rho_0 = atom_type_info["rho(R_dens)"]
            N_eff = atom_type_info["N_eff"]
            func_exp = lambda alpha : np.log(rho_0) - np.log((N_eff/(8.0*np.pi))*(np.abs(alpha)**3) * np.exp(-np.abs(alpha) * R_0))
            alpha_sol = np.abs(fsolve(func_exp,1.0,xtol=1e-8))
            # rho ~ N_eff exp(- alpha * r)
            R_fit = setup_info["exp fit scal"] * atom_type_info["R_dens"]
            rho_fit = (N_eff/(8.0*np.pi))*(alpha_sol**3) * np.exp(-alpha_sol * R_fit)
        else:
            R_fit = 1.0 * atom_type_info["R_dens"]
            rho_fit = atom_type_info["rho(R_dens)"]
            N_eff = atom_type_info["N_eff"]
        
        # fit the quasi density rho_quasi(R_fit) = rho_fit
        func = lambda gamma : rho_fit - N_eff*(phi_nG(R_fit,np.abs(gamma))**2)
        gamma_sol = fsolve(func,1.0,xtol=1e-8)
        beta = gamma_sol*gamma_sol * beta_0
        c = gamma_sol**(1.5) * c_0
        
        # create the basis function inform in pyscf format
        basis = [[0]+[ [beta[n],c[n]] for n in range(0,len(c))]]

        #return {"N_eff":1.0*N_eff,"coeffs":c,"exponents":beta}
        return {"N_eff":1.0*N_eff,"basis":basis,"orbital type":orb_type}
        
            
    def testEnergyFixedMultipoles(self,q):
        pol_resp = self.mm_system.getPolarizationEnergyForceResp(self.qm_positions,self.multipole_order,position_units="Bohr")
        U_0 = pol_resp["U_0"]
        U_1 = pol_resp["U_1"]
        U_2 = pol_resp["U_2"]
        E = U_0 + np.einsum('a,a',U_1,q) + 0.5 * np.einsum('a,ab,b',q,U_2,q)
        F_0_mm = pol_resp["F_0_mm"]
        F_1_mm = pol_resp["F_1_mm"]
        F_2_mm = pol_resp["F_2_mm"]
        F_mm = F_0_mm + np.einsum('kxa,a->kx',F_1_mm,q) + 0.5 * np.einsum('a,kxab,b->kx',q,F_2_mm,q)
        F_0_qm = pol_resp["F_0_qm"]
        F_1_qm = pol_resp["F_1_qm"]
        F_2_qm = pol_resp["F_2_qm"]
        F_qm = F_0_qm + np.einsum('kxa,a->kx',F_1_qm,q) + 0.5 * np.einsum('a,kxab,b->kx',q,F_2_qm,q)
        return E,F_qm,F_mm
        
    def enforcePBC(self,x):
        '''
        returns coordinates with centroid in box
        '''
        x0 = np.mean(x,axis=0) # centroid coordinates
        dx = x - x0[None,:] # difference from centroid
        dx_ni = dx - self.pbc_dims[None,:]*np.round(dx/self.pbc_dims[None,:])
        x0_inbox = x0 - (x0//self.pbc_dims)*self.pbc_dims # put centroid in box
        x_pbc = x0_inbox + dx_ni # put set of coords in box
        
        return x_pbc
        
    def getMMResidues(self):
        residue_groups = []
        for res in self.mm_system.simulation.topology.residues():
            atoms = [a.index for a in res.atoms()]
            residue_groups.append(atoms)
        return residue_groups
    
    def getClassicalIntForce(self,qq,q,pol_resp):
        '''
        This returns the classical portion DRF interaction force
        qq is the symmetrised <q_a q_b> matrix and q is <q_a> for the QM region
        '''
        
        # ensure that qq is symmetric
        qq = 0.5*(qq+qq.T)
        # get the eigenvalues and eigenvectors
        wq,vq = np.linalg.eigh(qq)
        # empty force vectors
        N_QM = self.qm_positions.shape[0]
        N_MM = self.mm_positions.shape[0]
        F_qm = np.zeros(self.qm_positions.shape)
        F_mm = np.zeros(self.mm_positions.shape)
        
        # for each eigenvector get the corresponding forces and add to the total force
        for n in range(0,len(wq)):
            if np.abs(wq[n])>1.0e-12:
                charges = vq[0:N_QM,n]
                if self.multipole_order>0:
                    dipoles = (vq[N_QM:(4*N_QM),n].reshape((3,N_QM))).T
                else:
                    dipoles = None
                F_qm_n, F_mm_n = self.mm_system.getProbeForces(self.qm_positions,charges,probe_dipoles=dipoles)
                F_qm += wq[n] * F_qm_n
                F_mm += wq[n] * F_mm_n
        
        # add correction for the linear terms
        q_corr = q - np.einsum('n,an->a',wq,vq)
        F_qm += np.einsum('kxa,a->kx',pol_resp["F_1_qm"],q_corr)
        F_mm += np.einsum('kxa,a->kx',pol_resp["F_1_mm"],q_corr)
        
        # add correction for the zero order term
        sum_wq = np.sum(wq)
        F_qm += (-sum_wq) * pol_resp["F_0_qm"]
        F_mm += (-sum_wq) * pol_resp["F_0_mm"]
        
        return F_qm, F_mm
