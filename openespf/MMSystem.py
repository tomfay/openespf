from openmm import *
import numpy as np
from copy import deepcopy
from timeit import default_timer as timer

class MMSystem:
    '''
    Class for handling ESPF-DRF and ESPF-DRF calculations using OpenMM to get electrostatic information
    
    All internal units are the same as OpenMM.
    All inputs/outputs default to atomic units.
    '''
    def __init__(self,simulation=None):
        

        # defaults for QM particle damping
        self.thole_default = 0.39
        self.damp_default = 0.0
        self.qm_thole = None
        self.qm_damp = None
        
        self.print_info = True
        
        # Get MM system simulation object
        if not simulation is None:
            self.simulation = simulation
            self.system = simulation.system
            self.positions = simulation.context.getState(getPositions=True).getPositions()
            self.topology = simulation.topology
        
        # create a new system object with the test charges/dipoles
        self.multipole_force = AmoebaMultipoleForce()
        for force in self.system.getForces():
            if force.getName() == "AmoebaMultipoleForce":
                self.multipole_force = deepcopy(force)
            else:
                force.setForceGroup(0)
        
        # create a new system object for test charges/dipoles and delete all forces
        self.multipole_system = deepcopy(self.system)
        for i in range(0,self.multipole_system.getNumForces()):
            self.multipole_system.removeForce(0)
        
        # add four particles to the force, initially set with zero charge,dipole etc.
        for i in range(0,4):
            c = 0.0
            d = [0.,0.,0.]
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = 0
            kz = 0
            kx = 0
            ky = 0
            thole = self.thole_default
            damp = self.damp_default
            pol = 0.0
            # add the particle to the force object
            self.multipole_force.addMultipole(c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            # add the particle to the system as well
            self.multipole_system.addParticle(0.0)
        
        # update the covalent maps for each test particle to remove nans
        N_MM = self.system.getNumParticles()
        for i in range(0,4):
            self.multipole_force.setCovalentMap(N_MM+i,0,[N_MM+j for j in range(0,4) if not j==i])
            self.multipole_force.setCovalentMap(N_MM+i,4,[N_MM+j for j in range(0,4) if not j==i])
        
        # add the new multipole force to the system
        self.multipole_system.addForce(self.multipole_force)
        
        # get the extended system topology and add the test particles
        self.multipole_topology = deepcopy(simulation.topology)
        #print("NumAtoms:",self.multipole_topology.getNumAtoms(),"NumRes:",self.multipole_topology.getNumResidues(),
        #     "NumChains:",self.multipole_topology.getNumChains())
        test_chain = self.multipole_topology.addChain()
        test_res = self.multipole_topology.addResidue("PRB",test_chain)
        for i in range(0,4):
            self.multipole_topology.addAtom(str(i+1),app.Element.getByAtomicNumber(2),test_res)
        #print("NumAtoms:",self.multipole_topology.getNumAtoms(),"NumRes:",self.multipole_topology.getNumResidues(),
        #     "NumChains:",self.multipole_topology.getNumChains())
        
        # set up the new Simulation object for the multipole force
        integrator = VerletIntegrator(1e-16)
        try:
            platform = simulation.platform
        except:
            platform = None
        
        self.multipole_simulation = app.Simulation(self.multipole_topology,self.multipole_system,integrator,
                                              platform)
        self.setProbePositions(self.positions)


        return
    
    def setPositions(self,positions,units_in="Bohr"):
        '''
        Updates positions of the MM particles in the simulation and in the multipole simulation objects
        '''
        if units_in == "nanometer":
            conv = 1 
        elif units_in in ["Angstrom","Ang","A","angstrom","ang"]:
            conv = 0.1
        elif units_in in ["Bohr","bohr","AU","au"]:
            conv = 0.52917721092e-1
        
        if type(positions[0]) == type(Vec3(0,0,0)*unit.nanometer):
            # case where positions has opennMM unit type
            self.positions = (conv*positions)._value*unit.nanometer
        elif type(positions[0]) == type(Vec3(0,0,0)):
            # case where positions is a list of Vec3 with no units
            self.positions = [conv*v for v in positions]*unit.nanometer
        else:
            # else convert N x 3 array to list Vec3
            N_MM = positions.shape[0]
            self.positions = [conv*Vec3(positions[A,0],positions[A,1],positions[A,2]) for A in range(0,N_MM) ]*unit.nanometer
        
        N_MM = len(self.positions)
        self.simulation.context.setPositions(self.positions)
        test_positions = self.multipole_simulation.context.getState(getPositions=True).getPositions()[N_MM:(N_MM+4)]
        self.setProbePositions(self.positions,test_positions=test_positions)
        
        return
    
    def setProbePositions(self,positions,test_positions=None):
        '''
        Updates positions in the multipole_simulation object.
        If test particle positions are not specified then then default to 0,0,0 (+ offset to avoid NaN in energy/forces)
        
        This only updates positions in the multipole_simulation object
        '''
        self.positions = positions
        if test_positions is None:
            test_positions = [Vec3(1e-6*i,0,0) for i in range(0,4)]*unit.nanometer
        self.multipole_simulation.context.setPositions(positions+test_positions)
        return
    
    def setQMPositions(self,qm_positions,units_in="Bohr"):
        '''
        Sets positions of the QM particles. Input can be list of Vec3 or N x 3 numpy array.
        '''
        if units_in == "nanometer":
            conv = 1.0
        elif units_in in ["Bohr","AU","au","bohr"]:
            conv = 0.52917721092e-1 
        elif units_in in ["Angstrom","Ang","A","angstrom","ang"]:
            conv = 0.1
            
        if type(qm_positions[0]) == type(Vec3(0,0,0)*unit.nanometer):
            # if already in list(Vec3) then directly copy
            self.qm_positions = (conv*qm_positions)._value*unit.nanometer
        elif type(qm_positions[0]) == type(Vec3(0,0,0)):
            self.qm_positions = [conv*v for v in qm_positions]*unit.nanometer
        else:
            # else convert N x 3 array to list Vec3
            self.qm_positions = []
            for A in range(0,qm_positions.shape[0]):
                self.qm_positions.append(conv*Vec3(qm_positions[A,0],qm_positions[A,1],qm_positions[A,2])) 
            self.qm_positions = self.qm_positions * unit.nanometer
        
        # check if qm_damp and qm_thole are None. If they are, expand to a list of Nones
        if self.qm_damp is None:
            self.qm_damp = [None] * len(qm_positions)
        if self.qm_thole is None:
            self.qm_thole = [None] * len(qm_positions)
        
        return
    
    def setQMDamping(self,damp,thole):
        '''
        Sets the damping parameters for the QM atoms to be used in the evaluation of QM-MM interactions.
        '''
        # note damp = (alpha)^(1/6) = (R_damp)^(1/2)
        self.qm_damp = damp
        # thole seems to be evaluated as exp(-thole_min * u^3), where thole_min is the minimum of thole parameters of inetracting sites
        self.qm_thole = thole
        
        return
    
    def resetMultipoleForce(self):
        '''
        Resets the state of the multipole_force object and the corresponding simulation to zero multipoles
        '''
        
        self.setTestParticleCharge([0.,0.,0.,0.],[0,1,2,3])
        
        return
    
    def getRefMultipoleEnergy(self):
        '''
        Gets the reference energy for the MM electrostatic system without 
        '''
        self.resetMultipoleForce()
        return self.multipole_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    def getRefMultipoleEnergyForces(self,as_numpy=True):
        '''
        Gets the reference energy for the MM electrostatic system without 
        '''
        self.resetMultipoleForce()
        return self.getMultipoleEnergyForces(as_numpy=as_numpy):
    
    def getMultipoleEnergy(self,get_forces=False):
        return self.multipole_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    def getMultipoleEnergyForces(self,as_numpy=True):
        state = self.multipole_simulation.context.getState(getEnergy=True,getForces=True)
        return state.getPotentialEnergy()._value, state.getForces(asNumpy=as_numpy)._value
    
    def getDiagTestParticleChargeMultipoleEnergy(self,U_MM=None):
        N_MM = self.system.getNumParticles()
        c = 1.0
        N_QM = len(self.qm_positions)
        if U_MM is None:
            U_MM = self.getRefMultipoleEnergy()._value
        U_lin = np.zeros((N_QM,))
        U_diag = np.zeros((N_QM,))
        U_plus = np.zeros((N_QM,))
        U_minus = np.zeros((N_QM,))
        
        # get energies of system with a test particle energy of +c
        start = timer()
        self.setTestParticleCharge(c,0)
        self.setTestParticleCharge([0,0,0],[1,2,3])

        for A in range(0,N_QM):
            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_plus[A] = self.getMultipoleEnergy()._value
            
        
        # get energies of system with a test particle energy of -c
        self.setTestParticleCharge(-c,0)
        for A in range(0,N_QM):
            self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_minus[A] = self.getMultipoleEnergy()._value
        
        U_lin = 0.5*(U_plus - U_minus)/c
        U_diag = ((U_plus + U_minus) - 2*U_MM)/(c*c)
        
        return U_MM,U_lin,U_diag
    
    def getDiagTestParticleChargeMultipoleEnergyForces(self,U_MM=None,F_MM_qm=None,F_MM_m=None):
        N_MM = self.system.getNumParticles()
        N_QM = len(self.qm_positions)
        c = 1.0
        
        # get the forces from the MM part
        F_MM_mm = np.zeros((N_MM,3))
        F_MM_qm = np.zeros((N_QM,3))
        if U_MM is None:
            U_MM, F = self.getRefMultipoleEnergyForces()
            F_MM_mm = F[0:N_MM,:]+0
            
        # set up variables for getting forces from diagonal components
        U_lin = np.zeros((N_QM,))
        U_diag = np.zeros((N_QM,))
        U_plus = np.zeros((N_QM,))
        U_minus = np.zeros((N_QM,))
        F_lin_mm = np.zeros((N_QM,N_MM,3))
        F_lin_qm = np.zeros((N_QM,N_QM,3))
        F_diag_mm = np.zeros((N_QM,N_MM,3))
        F_diag_qm = np.zeros((N_QM,N_QM,3))
        F_plus_mm = np.zeros((N_QM,N_MM,3))
        F_plus_qm = np.zeros((N_QM,N_QM,3))
        F_minus_mm = np.zeros((N_QM,N_MM,3))
        F_minus_qm = np.zeros((N_QM,N_QM,3))
        
        # get energies of system with a test particle energy of +c
        start = timer()
        self.setTestParticleCharge(c,0)
        self.setTestParticleCharge([0,0,0],[1,2,3])

        for A in range(0,N_QM):
            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_plus[A], F = self.getMultipoleEnergyForces()
            F_plus_mm[A,:,:] = F[0:N_MM,:]+0
            F_plus_qm[A,A,:] = F[N_MM,:]
            
        
        # get energies of system with a test particle energy of -c
        self.setTestParticleCharge(-c,0)
        for A in range(0,N_QM):
            self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_minus[A], F = self.getMultipoleEnergyForces()
            F_minus_mm[A,:,:] = F[0:N_MM,:]+0
            F_minus_qm[A,A,:] = F[N_MM,:]
        
        U_lin = 0.5*(U_plus - U_minus)/c
        U_diag = ((U_plus + U_minus) - 2*U_MM)/(c*c)
        F_lin_mm = 0.5*(F_plus_mm-F_minus_mm)/c
        F_diag_mm = ((F_plus_mm + F_minus_mm) - 2*F_MM_mm.reshape((1,N_MM,3)))/(c*c)
        F_lin_qm = 0.5*(F_plus_qm-F_minus_qm)/c
        F_diag_qm = ((F_plus_qm + F_minus_qm) - 2*F_MM_qm.reshape((1,N_QM,3)))/(c*c)
        
        return U_MM,U_lin,U_diag,F_MM_mm,F_MM_qm,F_lin_mm,F_lin_qm,F_diag_mm,F_diag_qm
    
    def getDiagTestParticleDipoleMultipoleEnergy(self,alpha,U_MM=None):
        if U_MM is None:
            U_MM = self.getRefMultipoleEnergy()._value
        N_MM = self.system.getNumParticles()
        c = 0.0
        d_0 = 1.0 
        d = [0.,0.,d_0]
        
        if alpha==0:
            n_alpha = Vec3(1,0,0)
        elif alpha==1:
            n_alpha = Vec3(0,1,0)
        elif alpha==2:
            n_alpha = Vec3(0,0,1)
            
        
        N_QM = len(self.qm_positions)
        U_MM = self.getRefMultipoleEnergy()._value
        U_lin = np.zeros((N_QM,))
        U_diag = np.zeros((N_QM,))
        U_plus = np.zeros((N_QM,))
        U_minus = np.zeros((N_QM,))
        
        # get energies of system with a test particle dipole = +d_0
        self.setTestParticleChargeDipole(c,d,[N_MM+1,0,0],0)
        self.setTestParticleCharge([0,0,0],[1,2,3])
        for A in range(0,N_QM):
            self.setTestParticleChargeDipole(c,d,[N_MM+1,0,0],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_plus[A] = self.getMultipoleEnergy()._value
        
        # get energies of system with a test particle dipole = -d_0
        d = [0.,0.,-d_0]
        self.setTestParticleChargeDipole(c,d,[N_MM+1,0,0],0)
        self.setTestParticleCharge([0,0,0],[1,2,3])
        for A in range(0,N_QM):
            self.setTestParticleChargeDipole(c,d,[N_MM+1,0,0],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_minus[A] = self.getMultipoleEnergy()._value
        
        U_lin = 0.5*(U_plus - U_minus)/d_0
        U_diag = ((U_plus + U_minus) - 2*U_MM)/(d_0*d_0)
        
        return U_MM,U_lin,U_diag
    
    def setTestParticleCharge(self,c,ind,thole=None,damp=None):
        if not type(c) == type([]):
            c = [c]
            ind = [ind]
        
        if thole is None:
            thole = [self.thole_default]*len(ind)
        elif not type(thole) == type([]):
            thole = [thole]
        if damp is None:
            damp = [self.damp_default]*len(ind)
        elif not type(damp) == type([]):
            damp = [damp]
        
        thole = [t if t is not None else self.thole_default for t in thole]
        damp = [d if d is not None else self.damp_default for d in damp]
            
        for c_i,i,thole_i,damp_i in zip(c,ind,thole,damp):
            N_MM = self.system.getNumParticles()
            index = N_MM + i
            d = [0.,0.,0.]
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = 0
            kz = 0
            kx = 0
            ky = 0
            
            pol = 0.0
            # set the particle parameters
            self.multipole_force.setMultipoleParameters(index,c_i,d,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
        
        self.multipole_force.updateParametersInContext(self.multipole_simulation.context)
        return
    
    def setTestParticleChargeDipole(self,c,d,k,ind,thole=None,damp=None):
        if not type(c) == type([]):
            c = [c]
            ind = [ind]
            d = [d]
            k = [k]
        
        if thole is None:
            thole = [self.thole_default]*len(ind)
        elif not type(thole) == type([]):
            thole = [thole]
        if damp is None:
            damp = [self.damp_default]*len(ind)
        elif not type(damp) == type([]):
            damp = [damp]
        
        thole = [t if t is not None else self.thole_default for t in thole]
        damp = [d if d is not None else self.damp_default for d in damp]
        
        for c_i,d_i,k_i,i,thole_i,damp_i in zip(c,d,k,ind,thole,damp):
            N_MM = self.system.getNumParticles()
            index = N_MM + i
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = 0
            kz = k_i[0]
            kx = k_i[1]
            ky = k_i[2]
            pol = 0.0
            # set the particle parameters
            self.multipole_force.setMultipoleParameters(index,c_i,d_i,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
        
        self.multipole_force.updateParametersInContext(self.multipole_simulation.context)
        return
    
    def getTestParticleChargeMultipoleEnergyExpansion(self):
        
        # get the ref, linear and quadratic terms in the energy expansion
        U_0, U_1,U_2_diag = self.getDiagTestParticleChargeMultipoleEnergy()
        N_QM = U_1.shape[0]
        
        # get off diagonal terms in the expansion
        c = 1.0 
        U_2 = np.zeros((N_QM,N_QM))
        U_2 = np.diag(U_2_diag)
        self.setTestParticleCharge([c,-c,0,0],[0,1,2,3])
        for A in range(0,N_QM):
            for B in range(0,A):
                thole = [self.qm_thole[A],self.qm_thole[B]]
                damp = [self.qm_damp[A],self.qm_damp[B]]
                self.setTestParticleCharge([c,-c,0,0],[0,1,2,3],thole=thole,damp=damp)
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[B]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,3)])*unit.nanometer
                self.setProbePositions(self.positions,test_positions=test_positions)
                U = self.getMultipoleEnergy()._value
                U_2[A,B] = (U - U_0 - (c*U_1[A]-c*U_1[B])-0.5*(c*c*U_2[A,A] + c*c*U_2[B,B]))/(-c*c)
                U_2[B,A] = U_2[A,B]
                
                
        
        return U_0, U_1, U_2
    
    def getTestParticleDipoleMultipoleEnergyExpansion(self,U_MM=None):
        
        # get the ref, linear and quadratic terms in the energy expansion
        U_0, U_1_x,U_2_diag_x = self.getDiagTestParticleDipoleMultipoleEnergy(0,U_MM=U_MM)
        U_0, U_1_y,U_2_diag_y = self.getDiagTestParticleDipoleMultipoleEnergy(1,U_MM=U_0)
        U_0, U_1_z,U_2_diag_z = self.getDiagTestParticleDipoleMultipoleEnergy(2,U_MM=U_0)
        N_QM = U_1_x.shape[0]
        U_1 = np.hstack((U_1_x,U_1_y,U_1_z))
        
        # get off diagonal terms in the expansion
        d_0 = 1.0e0
        d = [0.,0.,d_0]
        U_2 = np.diag(np.hstack((U_2_diag_x,U_2_diag_y,U_2_diag_z)))
        N_MM = self.system.getNumParticles()
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        
        self.setTestParticleChargeDipole([0.,0.],[d,d],[[N_MM+1,0,0],[N_MM+3,0,0]],[0,2])
        self.setTestParticleCharge([0.,0.],[1,3])
        for iA in range(0,3*N_QM):
            for jB in range(0,iA):
                A = int(iA % N_QM)
                B = int(jB % N_QM)
                i = int(iA//N_QM)
                j = int(jB//N_QM)
                thole = [self.qm_thole[A],self.qm_thole[B]]
                damp = [self.qm_damp[A],self.qm_damp[B]]
                self.setTestParticleChargeDipole([0.,0.],[d,d],[[N_MM+1,0,0],[N_MM+3,0,0]],[0,2],thole=thole,damp=damp)
                if not A==B:
                    test_positions = [self.qm_positions[A]._value,
                                      self.qm_positions[A]._value + 1e-2*n_alphas[i],
                                      self.qm_positions[B]._value,
                                      self.qm_positions[B]._value + 1e-2*n_alphas[j]]*unit.nanometer
                    self.setProbePositions(self.positions,test_positions=test_positions)
                    U = self.getMultipoleEnergy()._value
                    U_2[iA,jB] = (U - U_0 - (d_0*U_1[iA]+d_0*U_1[jB])-(0.5*d_0*d_0*U_2[iA,iA] + 0.5*d_0*d_0*U_2[jB,jB]))/(d_0*d_0)
                    U_2[jB,iA] = U_2[iA,jB]
        
        d = [0.,0.,np.sqrt(2.0)*d_0]
        self.setTestParticleChargeDipole(0.,d,[N_MM+1,0,0],0)
        self.setTestParticleCharge([0.,0.,0.],[1,2,3])
        for A in range(0,N_QM):
            for i in range(0,3):
                for j in range(0,i):
                    iA = N_QM*i + A 
                    jA = N_QM*j + A
                    self.setTestParticleChargeDipole(0.,d,[N_MM+1,0,0],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                    #print('i,j,A',i,j,A)
                    test_positions = [self.qm_positions[A]._value,
                                      self.qm_positions[A]._value + 1e-2*n_alphas[i]+ 1e-2*n_alphas[j],
                                      self.qm_positions[A]._value + 1e-6*n_alphas[0],
                                      self.qm_positions[A]._value + 2e-6*n_alphas[0]]*unit.nanometer
                    self.setProbePositions(self.positions,test_positions=test_positions)
                    U = self.getMultipoleEnergy()._value
                    #print(self.multipole_force.getLabFramePermanentDipoles(self.multipole_simulation.context))
                    U_2[iA,jA] = (U - U_0 - (d_0*U_1[iA]+d_0*U_1[jA])-(0.5*d_0*d_0*U_2[iA,iA] + 0.5*d_0*d_0*U_2[jA,jA]))/(d_0*d_0)
                    U_2[jA,iA] = U_2[iA,jA]
        
        return U_0, U_1, U_2
    
    
    def getTestParticleChargeDipoleCrossMultipoleEnergyExpansion(self,U_0,U_1,U_2_diag):
        d_0 = 1.0e0
        d = [0.,0.,d_0]
        c = 1.0
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        N_QM = int(U_1.shape[0]/4)
        N_MM = self.system.getNumParticles()
        self.setTestParticleChargeDipole(0.,d,[N_MM+3,0,0],2)
        self.setTestParticleCharge([c,0.,0.],[0,1,3])
        U_2_cross = np.zeros((N_QM,3*N_QM))
        
        # A has unit charge, B has unit dipole
        for A in range(0,N_QM):
            for j in range(0,3):
                for B in range(0,N_QM):
                    jB = j*N_QM+B
                    if not A==B:
                        test_positions = [self.qm_positions[A]._value,
                                          self.qm_positions[A]._value+1e-5*n_alphas[0],
                                          self.qm_positions[B]._value,
                                          self.qm_positions[B]._value+1e-2*n_alphas[j]]*unit.nanometer
                        self.setTestParticleChargeDipole(0.,d,[N_MM+3,0,0],2,thole=self.qm_thole[B],damp=self.qm_damp[B])
                        self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                        self.setProbePositions(self.positions,test_positions=test_positions)
                        U = self.getMultipoleEnergy()._value
                        U_2_cross[A,jB] = (U - U_0 - (c*U_1[A]+d_0*U_1[jB+N_QM])-0.5*(c*c*U_2_diag[A] + d_0*d_0*U_2_diag[jB+N_QM]))/(c*d_0)
        
        self.setTestParticleChargeDipole(c,d,[N_MM+1,0,0],0)
        self.setTestParticleCharge([0.,0.,0.],[1,2,3])
        for j in range(0,3):
            for A in range(0,N_QM):
                jA = j*N_QM+A
                test_positions = [self.qm_positions[A]._value,
                                  self.qm_positions[A]._value+1e-2*n_alphas[j],
                                  self.qm_positions[A]._value+1e-5*n_alphas[0],
                                  self.qm_positions[A]._value+2e-5*n_alphas[0]]*unit.nanometer
                self.setTestParticleChargeDipole(c,d,[N_MM+1,0,0],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                self.setProbePositions(self.positions,test_positions=test_positions)
                U = self.getMultipoleEnergy()._value
                U_2_cross[A,jA] = (U - U_0 - (c*U_1[A]+d_0*U_1[jA+N_QM])-0.5*(c*c*U_2_diag[A] + d_0*d_0*U_2_diag[jA+N_QM]))/(c*d_0)
        
        return U_2_cross
    
    def getTestParticleChargeDipoleMultipoleEnergyExpansion(self):
        
        # get the ref, linear and quadratic terms in the energy expansion for the charge part
        U_0, U_1_c, U_2_c = self.getTestParticleChargeMultipoleEnergyExpansion()
        U_0, U_1_d, U_2_d = self.getTestParticleDipoleMultipoleEnergyExpansion(U_MM=U_0)
        
        # full tensors
        U_1 = np.hstack((U_1_c,U_1_d))
        N_QM = U_1_c.shape[0]
        U_2 = np.zeros((4*N_QM,4*N_QM))
        N_Q = 4*N_QM
        U_2[0:N_QM,0:N_QM] = U_2_c
        U_2[N_QM:N_Q,N_QM:N_Q] = U_2_d
        U_2_cross = self.getTestParticleChargeDipoleCrossMultipoleEnergyExpansion(U_0,U_1,np.diag(U_2))
        U_2[0:N_QM,N_QM:N_Q] = U_2_cross
        U_2[N_QM:N_Q,0:N_QM] = U_2_cross.T
        
        return U_0, U_1, U_2
    
    def getPolarizationEnergyResp(self,qm_positions,multipole_order,position_units="Bohr",units_out="AU"):
        '''
        Gets a dictionary of the polarization energy response as a dictionary containing the U_n and info about the 
        units used. The assumed expansion form is
        U_pol = U_0 + sum_a q_a U_a + (1/2)sum_ab q_a q_b U_ab 
        '''
        
        # set positions
        self.setQMPositions(qm_positions,units_in=position_units)
        
        # get the expansion given the multipole order
        if multipole_order==0:
            U_0,U_1,U_2 = self.getTestParticleChargeMultipoleEnergyExpansion()
        elif multipole_order==1:
            U_0,U_1,U_2 = self.getTestParticleChargeDipoleMultipoleEnergyExpansion()
        else:
            raise Exception("Multipole order currently must be 0 or 1. multipole_order =",multipole_order,"is not allowed.")
        
        # collect everything together into a dictionary and convert units as needed
        if units_out == "OpenMM":
            pol_resp = {"U_0":U_0,"U_1":U_1,"U_2":U_2,"units":units_out} 
        elif units_out in ["AU","au"]:
            Eh = 2625.4996352210997 # hartree in kJ/mol
            a0 = 0.52917721092e-1 # bohr in nm
            conv = np.ones(U_1.shape)
            N_Q = conv.shape[0]
            if multipole_order==1:
                N_QM = int(N_Q/4)
                conv[N_QM:N_Q] = a0 
            pol_resp = {"U_0":U_0/Eh,"U_1":conv*U_1/(Eh) ,"U_2":(np.outer(conv,conv))*U_2/(Eh),"units":units_out} 
        
        return pol_resp
    
    def getPositions(self,as_numpy=True,no_openmmunit=True,units=None):
        '''
        Gets positions either as numpy or list. With or without openMM units. 
        If no_unit=False then the units conversion is ignored.
        '''
        if (units is None) and no_openmmunit:
            units = "Bohr" # defaults to output in atomic units
        elif (not units is None) and not no_openmmunit:
            print("Warning: \"units\" is ignored if no_openmmunit = True")
            
        if units in ["nanometer"]:
            conv = 1.0 
        elif units in ["Angstrom","Ang","A","angstrom","ang"]:
            conv = 1.0/0.1
        elif units in ["Bohr","bohr","AU","au"]:
            conv = 1.0/0.52917721092e-1
        
        if as_numpy and no_openmmunit:
            return conv*np.array([[R._value.x,R._value.y,R._value.z] for R in self.positions])
        elif as_numpy and not no_openmmunit:
            return np.array([[R._value.x,R._value.y,R._value.z] for R in self.positions]) * self.positions.unit
        elif not as_numpy and no_openmmunit:
            return ([[conv*R._value.x,conv*R._value.y,conv*R._value.z] for R in self.positions])
        elif not as_numpy and not no_openmmunit:
            return ([[R._value.x,R._value.y,R._value.z] for R in self.positions]) * self.positions.unit
        
    def getEnergy(self,terms=None,units_out="AU"):
        '''
        gets the energy of the MM syste
        '''
        
        
        if terms is None:
            energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        elif terms == "remainder":
            energy = self.simulation.context.getState(getEnergy=True,groups={0}).getPotentialEnergy()
        
        if units_out in ["AU","au","Hartree","hartree"]:
            Eh = 2625.4996352210997 # hartree in kJ/mol
            conv = 1.0/Eh
        elif units_out in ["OpenMM","kJ/mol"]:
            conv = 1.0
        
        return energy._value * conv
            