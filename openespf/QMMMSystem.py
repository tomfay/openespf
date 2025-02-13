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
        
        # printing
        self.print_info = True
        
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
            self.qm_system.setExchRepParameters(self.rep_info,self.mm_positions)
        
        # set up a dictionary for decomposition of the energy terms
        energy_terms = {} 
        # get the MM energy
        energy_terms["mm electrostatics"] = pol_resp["U_0"]
        start = timer()
        energy_terms["mm remainder"] = self.mm_system.getEnergy(terms="remainder")
        end = timer()
        if self.print_info : print("MM energy time:",end-start,"s")
        # get the QM + interaction energy 
        # TODO - modify to deal with multiple energies from excited states possibly return a list of dictionaries, one for each state
        start = timer()
        energy_terms = { **energy_terms, **self.qm_system.getEnergy(return_terms=True)}
        end = timer()
        if self.print_info : print("QM energy time:",end-start,"s")
        
        # return the energy
        if return_terms:
            return energy_terms
        else: 
            return np.sum(np.array([energy_terms[k] for k in list(energy_terms.keys())]))
        
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
        
            
        
        
        

