from openmm import *
import numpy as np
from copy import deepcopy
from timeit import default_timer as timer
from .Data import *
import openespf.Data as Data
import openespf.MultipoleForceExtras as extra

def getOpenMMPlatforms():
    N = Platform.getNumPlatforms()
    return [Platform.getPlatform(i).getName() for i in range(0,N)]
    
def getPlatformPropertiesInContext(context):
    platform = context.getPlatform()
    property_names = platform.getPropertyNames()
    properties = {}
    for name in property_names:
        properties[name] = platform.getPropertyValue(context,name)
    
    return properties
    

class MMSystem:
    '''
    Class for handling DREEM and DREEM calculations using OpenMM to get electrostatic information
    
    All internal units are the same as OpenMM.
    All inputs/outputs default to atomic units.
    '''
    def __init__(self,simulation=None):
        

        # defaults for QM particle damping
        self.thole_default = 0.39
        self.damp_default = 0.0
        self.qm_thole = None
        self.qm_damp = None
        self.test_dipole = 1.0e0*Data.BOHR_TO_NM
        self.test_charge = 1.0e1
        self.print_info = False
        #self.induced_dipole_error = 1.0e-8
        #self.max_iter_induced = 100
        self.resp_mode = "linear"
        self.prelim_dr = 1.0e-1*Data.BOHR_TO_NM # in nanometers
        #self.precision = 'double'
        avail_platforms = getOpenMMPlatforms()
        if "OpenCL" in avail_platforms or "GPU" in avail_platforms:
            self.use_prelim_mpole = False # use prelimit form of multipoles
        else:
            self.use_prelim_mpole = True
            print("Warning: OpenCL or GPU platform not detected so switching to pre-limit dipole mode. dr = ", self.prelim_dr, "nm ." )
            print("You can change this by manually setting mm_system.use_prelim_mpole = False.")
        
        # get precision of the context
        if simulation.context.getPlatform().getName() == 'OpenCL':
            self.precision = simulation.context.getPlatform().getPropertyValue(simulation.context,'Precision')
        
        self.damp_perm = True
        self.damp_charge_only = False
        self.damp_chargedipole_only = False
        self.resp_mode_force = "linear"
        
        # Get MM system simulation object
        if not simulation is None:
            self.simulation = simulation
            self.system = simulation.system
            self.positions = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()
            self.topology = simulation.topology
        
        # create a new system object with the test charges/dipoles
        self.multipole_force = AmoebaMultipoleForce()
        for force in self.system.getForces():
            if force.getName() == "AmoebaMultipoleForce":
                #force.setMutualInducedTargetEpsilon(self.induced_dipole_error)
                #force.setMutualInducedMaxIterations(self.max_iter_induced)
                force.setForceGroup(0)
                force.updateParametersInContext(simulation.context)
                self.multipole_force = deepcopy(force)
                
                #if self.multipole_force.getPolarizationType() == self.multipole_force.Mutual:
                #    print("Multipole forces uses Mutual polarization.")
                #elif self.multipole_force.getPolarizationType() == self.multipole_force.Direct:
                #    print("Multipole forces uses Direct polarization.")
                #elif self.multipole_force.getPolarizationType() == self.multipole_force.Extrapolated:
                #    print("Multipole forces uses Extrapolated polarization.")
                self.multipole_force.setNonbondedMethod(force.getNonbondedMethod())
                self.multipole_force.setPolarizationType(force.getPolarizationType())
                self.induced_dipole_error = force.getMutualInducedTargetEpsilon()
                self.max_iter_induced = force.getMutualInducedMaxIterations()
                if force.getNonbondedMethod() == force.PME:
                    print("PME parameters:" , force.getPMEParametersInContext(self.simulation.context))
                    self.multipole_force.setPMEParameters(*force.getPMEParametersInContext(self.simulation.context))
                    #self.multipole_force.setPMEParameters(force.getPMEParametersInContext(self.simulation.context)[0],6,6,6)
                    self.multipole_force.setCutoffDistance(force.getCutoffDistance())
                    
            else:
                force.setForceGroup(1)
                #force.updateParametersInContext(simulation.context)
        
        self.multipole_force.setMutualInducedTargetEpsilon(self.induced_dipole_error)
        self.multipole_force.setMutualInducedMaxIterations(self.max_iter_induced)
        
        # create a new system object for test charges/dipoles and delete all forces
        self.multipole_system = deepcopy(self.system)
        for i in range(0,self.multipole_system.getNumForces()):
            self.multipole_system.removeForce(0)
        
        # add four particles to the force, initially set with zero charge,dipole etc.
        for i in range(0,4):
            c = 0.0
            d = [0.,0.,0.]
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = self.multipole_force.NoAxisType
            kz = -1
            kx = -1
            ky = -1
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
            self.multipole_force.setCovalentMap(N_MM+i,self.multipole_force.Covalent12,[N_MM+j for j in range(0,4) if not j==i])
            self.multipole_force.setCovalentMap(N_MM+i,self.multipole_force.PolarizationCovalent11,[N_MM+j for j in range(0,4) if not j==i])
            
        
        
        
        # add the new multipole force to the system
        self.multipole_system.addForce(self.multipole_force)
        
        # get the extended system topology and add the test particles
        self.multipole_topology = deepcopy(simulation.topology)
       
        test_chain = self.multipole_topology.addChain()
        test_res = self.multipole_topology.addResidue("PRB",test_chain)
        for i in range(0,4):
            self.multipole_topology.addAtom(str(i+1),app.Element.getByAtomicNumber(2),test_res)
        atoms = [a for a in self.multipole_topology.atoms()]
        for i in range(0,4):
            for j in range(0,4):
                if not i==j:
                    self.multipole_topology.addBond(atoms[i+N_MM],atoms[j+N_MM])
        
        # set up the new Simulation object for the multipole force
        integrator = VerletIntegrator(1e-16)

        platform = Platform.getPlatformByName(simulation.context.getPlatform().getName())
        properties = getPlatformPropertiesInContext(simulation.context)
        self.multipole_simulation = app.Simulation(self.multipole_topology,self.multipole_system,integrator,
                                              platform,platformProperties=properties)
        
        
        if self.print_info:
            for i,force in enumerate(self.simulation.system.getForces()):
                print(force.getName(),force.getForceGroup())
            for force in self.multipole_simulation.system.getForces():
                print(force.getName(),force.getForceGroup())
        
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
            conv = Data.BOHR_TO_NM
        
        if type(positions[0]) == (type(Vec3(0,0,0)*unit.nanometer)):
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
        test_positions = self.multipole_simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions()[N_MM:(N_MM+4)]
        self.setProbePositions(self.positions,test_positions=test_positions)
        
        return
    
    
    
    def setProbePositions(self,positions,test_positions=None,inc_dir=False,inc_pol=False):
        '''
        Updates positions in the multipole_simulation object.
        If test particle positions are not specified then then default to 0,0,0 (+ offset to avoid NaN in energy/forces)
        
        This only updates positions in the multipole_simulation object
        '''
        self.positions = positions
        if test_positions is None:
            test_positions = [Vec3(1e-6*i,0,0) for i in range(0,4)]*unit.nanometer
        self.multipole_simulation.context.setPositions(positions+test_positions)
        if inc_dir:
            self.multipole_simulation_dir.context.setPositions(positions+test_positions)
        if inc_pol:
            self.multipole_simulation_pol.context.setPositions(positions+test_positions)
        return
    
    def setQMPositions(self,qm_positions,units_in="Bohr"):
        '''
        Sets positions of the QM particles. Input can be list of Vec3 or N x 3 numpy array.
        '''
        if units_in == "nanometer":
            conv = 1.0
        elif units_in in ["Bohr","AU","au","bohr"]:
            conv = Data.BOHR_TO_NM
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
        
        new = []
        for pos in self.qm_positions:
            new.append(Vec3(float(pos._value[0]),float(pos._value[1]),float(pos._value[2])))
        self.qm_positions = new*unit.nanometer
        
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
        return self.getMultipoleEnergyForces(as_numpy=as_numpy)
    
    def getMultipoleEnergy(self,get_forces=False):
        return self.multipole_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    def getMultipoleEnergyForces(self,as_numpy=True):
        state = self.multipole_simulation.context.getState(getEnergy=True,getForces=True)
        return state.getPotentialEnergy()._value, state.getForces(asNumpy=as_numpy)._value
    
    def getDiagTestParticleChargeMultipoleEnergy(self,U_MM=None):
        N_MM = self.system.getNumParticles()
        c = self.test_charge
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
        self.setTestParticleCharge([0.0,0.0,0.0],[1,2,3])

        for A in range(0,N_QM):
            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-3*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_plus[A] = self.getMultipoleEnergy()._value
            
        
        # get energies of system with a test particle energy of -c
        self.setTestParticleCharge(-c,0)
        for A in range(0,N_QM):
            self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-3*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_minus[A] = self.getMultipoleEnergy()._value
        
        U_lin = 0.5*(U_plus - U_minus)/c
        U_diag = ((U_plus + U_minus) - 2*U_MM)/(c*c)
        
        
        return U_MM,U_lin,U_diag
    
    def getDiagTestParticleChargeMultipoleEnergyForces(self,U_MM=None,F_MM_qm=None,F_MM_mm=None):
        N_MM = self.system.getNumParticles()
        N_QM = len(self.qm_positions)
        c = self.test_charge
        
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
        self.setTestParticleCharge([0.0,0.0,0.0],[1,2,3])

        for A in range(0,N_QM):
            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_plus[A], F = self.getMultipoleEnergyForces()
            F_plus_mm[A,:,:] = F[0:N_MM,:]+0
            F_plus_qm[A,A,:] = F[N_MM,:]+0
            
        
        # get energies of system with a test particle energy of -c
        self.setTestParticleCharge(-c,0)
        for A in range(0,N_QM):
            self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_minus[A], F = self.getMultipoleEnergyForces()
            F_minus_mm[A,:,:] = F[0:N_MM,:]+0
            F_minus_qm[A,A,:] = F[N_MM,:]+0
        
        U_lin = 0.5*(U_plus - U_minus)/c
        U_diag = ((U_plus + U_minus) - 2.0*U_MM)/(c*c)
        F_lin_mm = 0.5*(F_plus_mm-F_minus_mm)/c
        F_diag_mm = ((F_plus_mm + F_minus_mm) - 2*F_MM_mm.reshape((1,N_MM,3)))/(c*c)
        F_lin_qm = 0.5*(F_plus_qm-F_minus_qm)/c
        F_diag_qm = ((F_plus_qm + F_minus_qm) - 2*F_MM_qm.reshape((1,N_QM,3)))/(c*c)
        
        return U_MM,U_lin,U_diag,F_MM_mm,F_MM_qm,F_lin_mm,F_lin_qm,F_diag_mm,F_diag_qm
    
    def getTestParticleChargeMultipoleEnergyForcesExpansion(self):
        
        # get the ref, linear and quadratic terms in the energy expansion
        U_0,U_1,U_2_diag,F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_diag_mm,F_2_diag_qm = self.getDiagTestParticleChargeMultipoleEnergyForces()
        N_QM = U_1.shape[0]
        N_MM = self.system.getNumParticles()
        # get off diagonal terms in the expansion
        c = self.test_charge
        #U_2 = np.zeros((N_QM,N_QM))
        U_2 = np.diag(U_2_diag)
        F_2_mm = np.zeros((N_QM,N_QM,N_MM,3,))
        np.einsum('aakx->akx',F_2_mm)[:,:,:] = F_2_diag_mm
        F_2_qm = np.zeros((N_QM,N_QM,N_QM,3,))
        np.einsum('aakx->akx',F_2_qm)[:,:,:] = F_2_diag_qm
        
        self.setTestParticleCharge([c,-c,0,0],[0,1,2,3])
        for A in range(0,N_QM):
            for B in range(0,A):
                thole = [self.qm_thole[A],self.qm_thole[B]]
                damp = [self.qm_damp[A],self.qm_damp[B]]
                self.setTestParticleCharge([c,-c,0,0],[0,1,2,3],thole=thole,damp=damp)
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[B]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(1,3)])*unit.nanometer
                self.setProbePositions(self.positions,test_positions=test_positions)
                U, F = self.getMultipoleEnergyForces()
                U_2[A,B] = (U - U_0 - (c*U_1[A]-c*U_1[B])-0.5*(c*c*U_2[A,A] + c*c*U_2[B,B]))/(-c*c)
                U_2[B,A] = U_2[A,B]
                F_2_mm[A,B,0:N_MM,:] = (F[0:N_MM,:] - F_0_mm - (c*F_1_mm[A,0:N_MM,:]-c*F_1_mm[B,0:N_MM,:])-0.5*(c*c*F_2_mm[A,A,0:N_MM,:] + c*c*F_2_mm[B,B,0:N_MM,:]))/(-c*c)
                F_2_mm[B,A,0:N_MM,:] = F_2_mm[A,B,0:N_MM,:]
                F_2_qm[A,B,[A,B],:] = (F[[N_MM,(N_MM+1)],:] - F_0_qm[[A,B],:] - (c*F_1_qm[A,[A,B],:]-c*F_1_qm[B,[A,B],:])-0.5*(c*c*F_2_qm[A,A,[A,B],:] + c*c*F_2_qm[B,B,[A,B],:]))/(-c*c)
                F_2_qm[B,A,[A,B],:] = F_2_qm[A,B,[A,B],:]
            
        
        return U_0, U_1, U_2, F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm
    
    def getDiagTestParticleDipoleMultipoleEnergy(self,alpha,U_MM=None):
        if U_MM is None:
            U_MM = self.getRefMultipoleEnergy()._value
        N_MM = self.system.getNumParticles()
        d_0 = self.test_dipole 
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
        self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0,0,0],[1,2,3])
        for A in range(0,N_QM):
            self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_plus[A] = self.getMultipoleEnergy()._value
        
        # get energies of system with a test particle dipole = -d_0
        d = [0.,0.,-d_0]
        self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0,0,0],[1,2,3])
        for A in range(0,N_QM):
            self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
            test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions)
            U_minus[A] = self.getMultipoleEnergy()._value
        
        U_lin = 0.5*(U_plus - U_minus)/d_0
        U_diag = (U_plus + U_minus - 2*U_MM)/(d_0*d_0)
        
        return U_MM,U_lin,U_diag
    
    def getDiagTestParticleDipoleMultipoleEnergyForces(self,alpha,U_MM=None,F_MM_mm=None):
        if U_MM is None:
            U_MM = self.getRefMultipoleEnergy()._value
        N_MM = self.system.getNumParticles()
        N_QM = len(self.qm_positions)
        d_0 = self.test_dipole 
        d = [0.,0.,d_0]
        
        if alpha==0:
            n_alpha = Vec3(1,0,0)
        elif alpha==1:
            n_alpha = Vec3(0,1,0)
        elif alpha==2:
            n_alpha = Vec3(0,0,1)
            
        # get the forces from the MM part
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
        
        # get energies of system with a test particle dipole = +d_0
        self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0,0,0],[1,2,3])
        for A in range(0,N_QM):
            if not self.use_prelim_mpole:
                self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                self.setProbePositions(self.positions,test_positions=test_positions)
            else:
                self.setTestParticlePrelimDipole([0.],[n_alpha*(d_0)],[self.qm_positions[A]._value],[[1,0]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]])
            U_plus[A],F = self.getMultipoleEnergyForces()
            F_plus_mm[A,:,:] = F[0:N_MM,:]+0
            F_plus_qm[A,A,:] = F[N_MM,:]+F[N_MM+1,:] # force on dipole is sum of force on test particle with dipole + particle the dipole is defined wrt
        
        # get energies of system with a test particle dipole = -d_0
        d = [0.,0.,-d_0]
        self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0,0,0],[1,2,3])
        for A in range(0,N_QM):
            if not self.use_prelim_mpole:
                self.setTestParticleChargeDipole(0.0,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                self.setProbePositions(self.positions,test_positions=test_positions)
            else:
                self.setTestParticlePrelimDipole([0.],[n_alpha*(-d_0)],[self.qm_positions[A]._value],[[1,0]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]])
            U_minus[A],F = self.getMultipoleEnergyForces()
            F_minus_mm[A,:,:] = F[0:N_MM,:]+0
            F_minus_qm[A,A,:] = F[N_MM,:]+F[N_MM+1,:]
        
        U_lin = 0.5*(U_plus - U_minus)/d_0
        U_diag = ((U_plus + U_minus) - 2*U_MM)/(d_0*d_0)
        F_lin_mm = 0.5*(F_plus_mm-F_minus_mm)/d_0
        F_diag_mm = ((F_plus_mm + F_minus_mm) - 2*F_MM_mm.reshape((1,N_MM,3)))/(d_0*d_0)
        F_lin_qm = 0.5*(F_plus_qm-F_minus_qm)/d_0
        F_diag_qm = ((F_plus_qm + F_minus_qm) - 2*F_MM_qm.reshape((1,N_QM,3)))/(d_0*d_0)
        
        return U_MM,U_lin,U_diag,F_MM_mm,F_MM_qm,F_lin_mm,F_lin_qm,F_diag_mm,F_diag_qm
        
    
    def setTestParticleCharge(self,c,ind,thole=None,damp=None,inc_dir=False,inc_pol=False):
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
            axis_type = self.multipole_force.NoAxisType
            kz = -1
            kx = -1
            ky = -1
            
            pol = 0.0
            # set the particle parameters
            self.multipole_force.setMultipoleParameters(index,c_i,d,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
            if inc_dir:
                self.multipole_force_dir.setMultipoleParameters(index,c_i,d,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
            if inc_pol:
                self.multipole_force_pol.setMultipoleParameters(index,c_i,d,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
        
        self.multipole_force.updateParametersInContext(self.multipole_simulation.context)
        if inc_dir:
            self.multipole_force_dir.updateParametersInContext(self.multipole_simulation_dir.context)
        if inc_pol:
            self.multipole_force_pol.updateParametersInContext(self.multipole_simulation_pol.context)
        return
    
    def setTestParticleChargeDipole(self,c,d,k,ind,thole=None,damp=None,inc_dir=False,inc_pol=False):
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
            axis_type = self.multipole_force.ZOnly
            kz = k_i[0]
            kx = k_i[1]
            ky = k_i[2]
            pol = 0.0
            # set the particle parameters
            self.multipole_force.setMultipoleParameters(index,c_i,d_i,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
            if inc_dir:
                self.multipole_force_dir.setMultipoleParameters(index,c_i,d_i,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
            if inc_pol:
                self.multipole_force_pol.setMultipoleParameters(index,c_i,d_i,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
        
        self.multipole_force.updateParametersInContext(self.multipole_simulation.context)
        if inc_dir:
            self.multipole_force_dir.updateParametersInContext(self.multipole_simulation_dir.context)
        if inc_pol:
            self.multipole_force_pol.updateParametersInContext(self.multipole_simulation_pol.context)
        return
    
    def getTestParticleChargeMultipoleEnergyExpansion(self):
        
        # get the ref, linear and quadratic terms in the energy expansion
        U_0, U_1,U_2_diag = self.getDiagTestParticleChargeMultipoleEnergy()
        N_QM = U_1.shape[0]
        
        # get off diagonal terms in the expansion
        c = self.test_charge
        U_2 = np.zeros((N_QM,N_QM))
        U_2 = np.diag(U_2_diag)
        self.setTestParticleCharge([c,-c,0,0],[0,1,2,3])
        for A in range(0,N_QM):
            for B in range(0,A):
                thole = [self.qm_thole[A],self.qm_thole[B]]
                damp = [self.qm_damp[A],self.qm_damp[B]]
                self.setTestParticleCharge([c,-c],[0,1,2,3],thole=thole,damp=damp)
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
        d_0 = self.test_dipole
        d = [0.,0.,d_0]
        U_2 = np.diag(np.hstack((U_2_diag_x,U_2_diag_y,U_2_diag_z)))
        N_MM = self.system.getNumParticles()
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        
        self.setTestParticleChargeDipole([0.,0.],[d,d],[[N_MM+1,-1,-1],[N_MM+3,-1,-1]],[0,2])
        self.setTestParticleCharge([0.,0.],[1,3])
        for iA in range(0,3*N_QM):
            for jB in range(0,iA):
                A = int(iA % N_QM)
                B = int(jB % N_QM)
                i = int(iA//N_QM)
                j = int(jB//N_QM)
                thole = [self.qm_thole[A],self.qm_thole[B]]
                damp = [self.qm_damp[A],self.qm_damp[B]]
                self.setTestParticleChargeDipole([0.,0.],[d,d],[[N_MM+1,-1,-1],[N_MM+3,-1,-1]],[0,2],thole=thole,damp=damp)
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
        self.setTestParticleChargeDipole(0.,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0.,0.,0.],[1,2,3])
        for A in range(0,N_QM):
            for i in range(0,3):
                for j in range(0,i):
                    iA = N_QM*i + A 
                    jA = N_QM*j + A
                    self.setTestParticleChargeDipole(0.,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                    #print('i,j,A',i,j,A)
                    test_positions = [self.qm_positions[A]._value,
                                      self.qm_positions[A]._value + 1e-2*n_alphas[i]+ 1e-2*n_alphas[j],
                                      self.qm_positions[A]._value + 1e-3*n_alphas[0],
                                      self.qm_positions[A]._value + 2e-3*n_alphas[0]]*unit.nanometer
                    self.setProbePositions(self.positions,test_positions=test_positions)
                    U = self.getMultipoleEnergy()._value
                    #print(self.multipole_force.getLabFramePermanentDipoles(self.multipole_simulation.context))
                    U_2[iA,jA] = (U - U_0 - (d_0*U_1[iA]+d_0*U_1[jA])-(0.5*d_0*d_0*U_2[iA,iA] + 0.5*d_0*d_0*U_2[jA,jA]))/(d_0*d_0)
                    U_2[jA,iA] = U_2[iA,jA]
        
        return U_0, U_1, U_2
    
    def getTestParticleDipoleMultipoleEnergyForcesExpansion(self,U_MM=None,F_MM_mm=None):
        
        # get the ref, linear and quadratic terms in the energy expansion
        U_0,U_1_x,U_2_diag_x,F_0_mm,F_0_qm,F_1_x_mm,F_1_x_qm,F_2_x_mm,F_2_x_qm = self.getDiagTestParticleDipoleMultipoleEnergyForces(0,U_MM=U_MM,F_MM_mm=F_MM_mm)
        U_0,U_1_y,U_2_diag_y,F_0_mm,F_0_qm,F_1_y_mm,F_1_y_qm,F_2_y_mm,F_2_y_qm = self.getDiagTestParticleDipoleMultipoleEnergyForces(1,U_MM=U_0,F_MM_mm=F_0_mm)
        U_0,U_1_z,U_2_diag_z,F_0_mm,F_0_qm,F_1_z_mm,F_1_z_qm,F_2_z_mm,F_2_z_qm = self.getDiagTestParticleDipoleMultipoleEnergyForces(2,U_MM=U_0,F_MM_mm=F_0_mm)
        N_QM = U_1_x.shape[0]
        N_MM = self.system.getNumParticles()
        
        # set up the dipole linear expansion
        U_1 = np.hstack((U_1_x,U_1_y,U_1_z))
        F_1_mm = np.concatenate((F_1_x_mm,F_1_y_mm,F_1_z_mm),axis=0) # 3N_QM x N_MM x 3 array
        F_1_qm = np.concatenate((F_1_x_qm,F_1_y_qm,F_1_z_qm),axis=0) # 3N_QM x N_QM x 3 array
        
        # set diagonal terms in the expansion
        d_0 = self.test_dipole
        d = [0.,0.,d_0]
        U_2 = np.diag(np.hstack((U_2_diag_x,U_2_diag_y,U_2_diag_z)))
        F_2_mm = np.zeros((3*N_QM,3*N_QM,N_MM,3))
        F_2_qm = np.zeros((3*N_QM,3*N_QM,N_QM,3))
        np.einsum('aakx->akx',F_2_mm)[:,:,:] = np.concatenate((F_2_x_mm,F_2_y_mm,F_2_z_mm),axis=0)
        np.einsum('aakx->akx',F_2_qm)[:,:,:] = np.concatenate((F_2_x_qm,F_2_y_qm,F_2_z_qm),axis=0)
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        
        
        
        # get the off-diagonal terms for different sites
        self.setTestParticleChargeDipole([0.,0.],[d,d],[[N_MM+1,-1,-1],[N_MM+3,-1,-1]],[0,2])
        self.setTestParticleCharge([0.,0.],[1,3])
        for iA in range(0,3*N_QM):
            for jB in range(0,iA):
                A = int(iA % N_QM)
                B = int(jB % N_QM)
                i = int(iA//N_QM)
                j = int(jB//N_QM)
                thole = [self.qm_thole[A],self.qm_thole[B]]
                damp = [self.qm_damp[A],self.qm_damp[B]]
                self.setTestParticleChargeDipole([0.,0.],[d,d],[[N_MM+1,-1,-1],[N_MM+3,-1,-1]],[0,2],thole=thole,damp=damp)
                if not A==B:
                    if not self.use_prelim_mpole:
                        test_positions = [self.qm_positions[A]._value,
                                          self.qm_positions[A]._value + 1e-2*n_alphas[i],
                                          self.qm_positions[B]._value,
                                          self.qm_positions[B]._value + 1e-2*n_alphas[j]]*unit.nanometer
                        self.setProbePositions(self.positions,test_positions=test_positions)
                    else:
                        self.setTestParticlePrelimDipole([0.,0.],[n_alphas[i]*(d_0),n_alphas[j]*d_0],[self.qm_positions[A]._value,self.qm_positions[B]._value],[[1,0],[3,2]],thole=[self.qm_thole[A],self.qm_thole[B]],damp=[self.qm_damp[A],self.qm_damp[B]])
                    U, F = self.getMultipoleEnergyForces()
                    U_2[iA,jB] = (U - U_0 - (d_0*U_1[iA]+d_0*U_1[jB])-(0.5*d_0*d_0*U_2[iA,iA] + 0.5*d_0*d_0*U_2[jB,jB]))/(d_0*d_0)
                    U_2[jB,iA] = U_2[iA,jB]
                    F_mm = F[0:N_MM,:] +0
                    F_2_mm[iA,jB,:,:] = (F_mm - F_0_mm - (d_0*F_1_mm[iA,:,:]+d_0*F_1_mm[jB,:,:])-(0.5*d_0*d_0*F_2_mm[iA,iA,:,:] + 0.5*d_0*d_0*F_2_mm[jB,jB,:,:]))/(d_0*d_0)
                    F_A = F[N_MM,:] + F[N_MM+1,:] 
                    F_B = F[N_MM+2,:] + F[N_MM+3,:]
                    F_2_qm[iA,jB,A,:] = (F_A - F_0_qm[A,:] - (d_0*F_1_qm[iA,A,:]+d_0*F_1_qm[jB,A,:])-(0.5*d_0*d_0*F_2_qm[iA,iA,A,:] + 0.5*d_0*d_0*F_2_qm[jB,jB,A,:]))/(d_0*d_0)
                    F_2_qm[iA,jB,B,:] = (F_B - F_0_qm[B,:] - (d_0*F_1_qm[iA,B,:]+d_0*F_1_qm[jB,B,:])-(0.5*d_0*d_0*F_2_qm[iA,iA,B,:] + 0.5*d_0*d_0*F_2_qm[jB,jB,B,:]))/(d_0*d_0)
                    F_2_qm[jB,iA,[A,B],:] = F_2_qm[iA,jB,[A,B],:]
                    F_2_mm[jB,iA,:,:] = F_2_mm[iA,jB,:,:]
        
        
        # get the off-diagonal terms for the same sites
        d = [0.,0.,np.sqrt(2.0)*d_0]
        self.setTestParticleChargeDipole(0.,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0.,0.,0.],[1,2,3])
        
        for A in range(0,N_QM):
            for i in range(0,3):
                for j in range(0,i):
                    iA = N_QM*i + A 
                    jA = N_QM*j + A
                    if not self.use_prelim_mpole:
                        self.setTestParticleChargeDipole(0.,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                        #print('i,j,A',i,j,A)
                        test_positions = [self.qm_positions[A]._value,
                                          self.qm_positions[A]._value + 1e-2*n_alphas[i]+ 1e-2*n_alphas[j],
                                          self.qm_positions[A]._value + 1e-3*n_alphas[0],
                                          self.qm_positions[A]._value + 2e-3*n_alphas[0]]*unit.nanometer
                        self.setProbePositions(self.positions,test_positions=test_positions)
                    else:
                        self.setTestParticlePrelimDipole([0.],[n_alphas[i]*(d_0)+n_alphas[j]*d_0],[self.qm_positions[A]._value],[[1,0]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]])
                    
                    U,F = self.getMultipoleEnergyForces()
                    
                    #print(self.multipole_force.getLabFramePermanentDipoles(self.multipole_simulation.context))
                    U_2[iA,jA] = (U - U_0 - (d_0*U_1[iA]+d_0*U_1[jA])-(0.5*d_0*d_0*U_2[iA,iA] + 0.5*d_0*d_0*U_2[jA,jA]))/(d_0*d_0)
                    U_2[jA,iA] = U_2[iA,jA]
                    F_mm = F[0:N_MM,:] +0 
                    F_A = F[N_MM,:] + F[N_MM+1,:] 
                    F_2_mm[iA,jA,:,:] = (F_mm - F_0_mm - (d_0*F_1_mm[iA,:,:]+d_0*F_1_mm[jA,:,:])-(0.5*d_0*d_0*F_2_mm[iA,iA,:,:] + 0.5*d_0*d_0*F_2_mm[jA,jA,:,:]))/(d_0*d_0)
                    F_2_qm[iA,jA,A,:] = (F_A - F_0_qm[A,:] - (d_0*F_1_qm[iA,A,:]+d_0*F_1_qm[jA,A,:])-(0.5*d_0*d_0*F_2_qm[iA,iA,A,:] + 0.5*d_0*d_0*F_2_qm[jA,jA,A,:]))/(d_0*d_0)
                    F_2_mm[jA,iA,:,:] = F_2_mm[iA,jA,:,:]
                    F_2_qm[jA,iA,A,:] = F_2_qm[iA,jA,A,:]
        
        #print("Dipole-dipole contributions to expansion")
        #ax = ["x","y","z"]
        #for i in range(0,3):
        #    for A in range(0,N_QM):
        #        for j in range(0,3):
        #            for B in range(0,N_QM):
        #                iA = N_QM*i + A 
        #                jB = N_QM*j + B
        #                print("iA,jB",ax[i],A,ax[j],B)
        #                print(F_2_mm[iA,jB,:,:])
        
        return U_0, U_1, U_2, F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm
    
    
    def getTestParticleChargeDipoleCrossMultipoleEnergyExpansion(self,U_0,U_1,U_2_diag):
        d_0 = self.test_dipole
        d = [0.,0.,d_0]
        c = self.test_charge
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        N_QM = int(U_1.shape[0]/4)
        N_MM = self.system.getNumParticles()
        self.setTestParticleChargeDipole(0.,d,[N_MM+3,-1,-1],2)
        self.setTestParticleCharge([c,0.,0.],[0,1,3])
        U_2_cross = np.zeros((N_QM,3*N_QM))
        
        # A has unit charge, B has unit dipole
        for A in range(0,N_QM):
            for j in range(0,3):
                for B in range(0,N_QM):
                    jB = j*N_QM+B
                    if not A==B:
                        if not self.use_prelim_mpole:
                            test_positions = [self.qm_positions[A]._value,
                                              self.qm_positions[A]._value+1e-5*n_alphas[0],
                                              self.qm_positions[B]._value,
                                              self.qm_positions[B]._value+1e-2*n_alphas[j]]*unit.nanometer
                            self.setTestParticleChargeDipole(0.,d,[N_MM+3,-1,-1],2,thole=self.qm_thole[B],damp=self.qm_damp[B])
                            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                            self.setProbePositions(self.positions,test_positions=test_positions)
                        else:
                            self.setTestParticlePrelimDipole([c,0.],[Vec3(0.,0.,0.),n_alphas[j]*d_0],[self.qm_positions[A]._value,self.qm_positions[B]._value],[[1,0],[3,2]],thole=[self.qm_thole[A],self.qm_thole[B]],damp=[self.qm_damp[A],self.qm_damp[B]])
                    
                        U = self.getMultipoleEnergy()._value
                        U_2_cross[A,jB] = (U - U_0 - (c*U_1[A]+d_0*U_1[jB+N_QM])-0.5*(c*c*U_2_diag[A] + d_0*d_0*U_2_diag[jB+N_QM]))/(c*d_0)
        
        self.setTestParticleChargeDipole(c,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0.,0.,0.],[1,2,3])
        for j in range(0,3):
            for A in range(0,N_QM):
                jA = j*N_QM+A
                if not self.use_prelim_mpole:
                    test_positions = [self.qm_positions[A]._value,
                                      self.qm_positions[A]._value+1e-2*n_alphas[j],
                                      self.qm_positions[A]._value+1e-5*n_alphas[0],
                                      self.qm_positions[A]._value+2e-5*n_alphas[0]]*unit.nanometer
                    self.setTestParticleChargeDipole(c,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                    self.setProbePositions(self.positions,test_positions=test_positions)
                else:
                    self.setTestParticlePrelimDipole([c],[n_alphas[j]*d_0],[self.qm_positions[A]._value],[[1,0]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]])
                U = self.getMultipoleEnergy()._value
                U_2_cross[A,jA] = (U - U_0 - (c*U_1[A]+d_0*U_1[jA+N_QM])-0.5*(c*c*U_2_diag[A] + d_0*d_0*U_2_diag[jA+N_QM]))/(c*d_0)
        
        return U_2_cross
    
    def getTestParticleChargeDipoleCrossMultipoleEnergyForceExpansion(self,U_0,U_1,U_2_diag,F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm):
        d_0 = self.test_dipole
        d = [0.,0.,d_0]
        c = self.test_charge
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        N_QM = int(U_1.shape[0]/4)
        N_MM = self.system.getNumParticles()
        self.setTestParticleChargeDipole(0.,d,[N_MM+3,-1,-1],2)
        self.setTestParticleCharge([c,0.,0.],[0,1,3])
        U_2_cross = np.zeros((N_QM,3*N_QM))
        F_2_mm_cross = np.zeros((N_QM,3*N_QM,N_MM,3))
        F_2_qm_cross = np.zeros((N_QM,3*N_QM,N_QM,3))
        
        # A has unit charge, B has unit dipole
        for A in range(0,N_QM):
            for j in range(0,3):
                for B in range(0,N_QM):
                    jB = j*N_QM+B
                    if not A==B:
                        if not self.use_prelim_mpole:
                            test_positions = [self.qm_positions[A]._value,
                                              self.qm_positions[A]._value+1e-5*n_alphas[0],
                                              self.qm_positions[B]._value,
                                              self.qm_positions[B]._value+1e-2*n_alphas[j]]*unit.nanometer
                            self.setTestParticleChargeDipole(0.,d,[N_MM+3,-1,-1],2,thole=self.qm_thole[B],damp=self.qm_damp[B])
                            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                            self.setProbePositions(self.positions,test_positions=test_positions)
                        else:
                            self.setTestParticlePrelimDipole([c,0.],[Vec3(0.,0.,0.),n_alphas[j]*d_0],[self.qm_positions[A]._value,self.qm_positions[B]._value],[[0,1],[3,2]],thole=[self.qm_thole[A],self.qm_thole[B]],damp=[self.qm_damp[A],self.qm_damp[B]])
                    
                        U,F = self.getMultipoleEnergyForces()
                        F_mm = F[0:N_MM,:]+0
                        F_A = F[N_MM,:]+0
                        F_B = F[N_MM+2,:]+F[N_MM+3,:]
                        U_2_cross[A,jB] = (U - U_0 - (c*U_1[A]+d_0*U_1[jB+N_QM])-0.5*(c*c*U_2_diag[A] + d_0*d_0*U_2_diag[jB+N_QM]))/(c*d_0)
                        F_2_mm_cross[A,jB,:,:] = (F_mm - F_0_mm - (c*F_1_mm[A,:,:]+d_0*F_1_mm[jB+N_QM,:,:])-0.5*(c*c*F_2_mm[A,A,:,:] + d_0*d_0*F_2_mm[jB+N_QM,jB+N_QM,:,:]))/(c*d_0)
                        F_2_qm_cross[A,jB,A,:] = (F_A - F_0_qm[A,:] - (c*F_1_qm[A,A,:]+d_0*F_1_qm[jB+N_QM,A,:])-0.5*(c*c*F_2_qm[A,A,A,:] + d_0*d_0*F_2_qm[jB+N_QM,jB+N_QM,A,:]))/(c*d_0)
                        F_2_qm_cross[A,jB,B,:] = (F_B - F_0_qm[B,:] - (c*F_1_qm[A,B,:]+d_0*F_1_qm[jB+N_QM,B,:])-0.5*(c*c*F_2_qm[A,A,B,:] + d_0*d_0*F_2_qm[jB+N_QM,jB+N_QM,B,:]))/(c*d_0)
        
        
        self.setTestParticleChargeDipole(c,d,[N_MM+1,-1,-1],0)
        self.setTestParticleCharge([0.,0.,0.],[1,2,3])
        for j in range(0,3):
            for A in range(0,N_QM):
                jA = j*N_QM+A
                if not self.use_prelim_mpole:
                    test_positions = [self.qm_positions[A]._value,
                                      self.qm_positions[A]._value+1e-2*n_alphas[j],
                                      self.qm_positions[A]._value+1e-5*n_alphas[0],
                                      self.qm_positions[A]._value+2e-5*n_alphas[0]]*unit.nanometer
                    self.setTestParticleChargeDipole(c,d,[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A])
                    self.setProbePositions(self.positions,test_positions=test_positions)
                else:
                    self.setTestParticlePrelimDipole([c],[n_alphas[j]*d_0],[self.qm_positions[A]._value],[[1,0],[3,2]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]])
                    
                U,F = self.getMultipoleEnergyForces()
                F_mm = F[0:N_MM,:]+0
                F_A = F[N_MM,:]+F[N_MM+1,:]
                U_2_cross[A,jA] = (U - U_0 - (c*U_1[A]+d_0*U_1[jA+N_QM])-0.5*(c*c*U_2_diag[A] + d_0*d_0*U_2_diag[jA+N_QM]))/(c*d_0)
                F_2_mm_cross[A,jA,:,:] = (F_mm - F_0_mm - (c*F_1_mm[A,:,:]+d_0*F_1_mm[jA+N_QM,:,:])-0.5*(c*c*F_2_mm[A,A,:,:] + d_0*d_0*F_2_mm[jA+N_QM,jA+N_QM,:,:]))/(c*d_0)
                F_2_qm_cross[A,jA,A,:] = (F_A - F_0_qm[A,:] - (c*F_1_qm[A,A,:]+d_0*F_1_qm[jA+N_QM,A,:])-0.5*(c*c*F_2_qm[A,A,A,:] + d_0*d_0*F_2_qm[jA+N_QM,jA+N_QM,A,:]) )/(c*d_0)
                        
   
        return U_2_cross,F_2_mm_cross,F_2_qm_cross
    
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
    
    def getTestParticleChargeDipoleMultipoleEnergyForcesExpansion(self):
        
        # get the ref, linear and quadratic terms in the energy expansion for the charge part
        U_0,U_1_c,U_2_c,F_0_mm,F_0_qm,F_1_mm_c,F_1_qm_c,F_2_mm_c,F_2_qm_c = self.getTestParticleChargeMultipoleEnergyForcesExpansion()
        U_0,U_1_d,U_2_d,F_0_mm,F_0_qm,F_1_mm_d,F_1_qm_d,F_2_mm_d,F_2_qm_d = self.getTestParticleDipoleMultipoleEnergyForcesExpansion(U_MM=U_0,F_MM_mm=F_0_mm)
        
        
        # full tensors
        U_1 = np.hstack((U_1_c,U_1_d))
        F_1_mm = np.concatenate((F_1_mm_c,F_1_mm_d),axis=0)
        F_1_qm = np.concatenate((F_1_qm_c,F_1_qm_d),axis=0)
        N_QM = U_1_c.shape[0]
        N_MM = F_0_mm.shape[0]
        N_Q = 4*N_QM
        U_2 = np.zeros((N_Q,N_Q))
        U_2[0:N_QM,0:N_QM] = U_2_c
        U_2[N_QM:N_Q,N_QM:N_Q] = U_2_d
        F_2_mm = np.zeros((4*N_QM,4*N_QM,N_MM,3))
        F_2_mm[0:N_QM,0:N_QM,:,:] = F_2_mm_c
        F_2_mm[N_QM:N_Q,N_QM:N_Q,:,:] = F_2_mm_d
        F_2_qm = np.zeros((4*N_QM,4*N_QM,N_QM,3))
        F_2_qm[0:N_QM,0:N_QM,:,:] = F_2_qm_c
        F_2_qm[N_QM:N_Q,N_QM:N_Q,:,:] = F_2_qm_d
        
        U_2_cross,F_2_mm_cross, F_2_qm_cross = self.getTestParticleChargeDipoleCrossMultipoleEnergyForceExpansion(U_0,U_1,np.diag(U_2),F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm)
        U_2[0:N_QM,N_QM:N_Q] = U_2_cross
        U_2[N_QM:N_Q,0:N_QM] = U_2_cross.T
        F_2_mm[0:N_QM,N_QM:N_Q,:,:] += F_2_mm_cross
        F_2_mm[N_QM:N_Q,0:N_QM,:,:] += np.permute_dims(F_2_mm_cross,axes=(1,0,2,3))
        F_2_qm[0:N_QM,N_QM:N_Q,:,:] += F_2_qm_cross
        F_2_qm[N_QM:N_Q,0:N_QM,:,:] += np.permute_dims(F_2_qm_cross,axes=(1,0,2,3))
        
        
        
        return U_0, U_1, U_2, F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm
    
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
            if self.resp_mode == "quadratic":
                U_0,U_1,U_2 = self.getTestParticleChargeMultipoleEnergyExpansion()
            elif self.resp_mode == "linear":
                U_0,U_1,U_2 = self.getChargePolarizationResponseLinearScaling()
        elif multipole_order==1:
            if self.resp_mode == "quadratic":
                U_0,U_1,U_2 = self.getTestParticleChargeDipoleMultipoleEnergyExpansion()
            elif self.resp_mode == "linear":
                U_0,U_1,U_2 = self.getChargeDipolePolarizationResponseLinearScaling()
        else:
            raise Exception("Multipole order currently must be 0 or 1. multipole_order =",multipole_order,"is not allowed.")
        
        if self.damp_perm:
            dU_1 = self.getPermMultipoleDampCorrectionCharge(multipole_order,get_force_corr=False)
            U_1 += dU_1
            U_1_uncorr = U_1 - dU_1
        
        # collect everything together into a dictionary and convert units as needed
        if units_out == "OpenMM":
            pol_resp = {"U_0":U_0,"U_1":U_1,"U_2":U_2,"units":units_out} 
            if self.damp_perm:
                pol_resp["U_1_uncorr"] = U_1_uncorr
        elif units_out in ["AU","au"]:
            Eh = Data.HARTREE_TO_KJMOL # hartree in kJ/mol
            a0 = Data.BOHR_TO_NM # bohr in nm
            conv = np.ones(U_1.shape)
            N_Q = conv.shape[0]
            if multipole_order==1:
                N_QM = int(N_Q/4)
                conv[N_QM:N_Q] = a0 
            pol_resp = {"U_0":U_0/Eh,"U_1":conv*U_1/(Eh) ,"U_2":(np.outer(conv,conv))*U_2/(Eh),"units":units_out} 
            if self.damp_perm:
                pol_resp["U_1_uncorr"] = conv*U_1_uncorr/Eh
        
        
            
        return pol_resp
    
    def getElectrostaticEnergy(self,units_out="AU"):
        
        U_0 = self.getRefMultipoleEnergy()._value
        
        # collect everything together into a dictionary and convert units as needed
        if units_out == "OpenMM":
            pol_resp = {"U_0":U_0,"U_1":U_1,"U_2":U_2,"units":units_out} 
        elif units_out in ["AU","au"]:
            Eh = Data.HARTREE_TO_KJMOL # hartree in kJ/mol
            a0 = Data.BOHR_TO_NM # bohr in nm
            U_0 = U_0 / Eh
            
            
        return U_0
    
    def getElectrostaticEnergyForces(self,units_out="AU"):
        
        U_0,F = self.getRefMultipoleEnergyForces()
        N_MM = self.system.getNumParticles()
        F = F[0:N_MM]
        
        # collect everything together into a dictionary and convert units as needed
        if units_out == "OpenMM":
            pol_resp = {"U_0":U_0,"U_1":U_1,"U_2":U_2,"units":units_out} 
        elif units_out in ["AU","au"]:
            Eh = Data.HARTREE_TO_KJMOL # hartree in kJ/mol
            a0 = Data.BOHR_TO_NM # bohr in nm
            Eh_per_a0 = Eh / a0
            U_0 = U_0 / Eh
            F = F / Eh_per_a0
            
            
            
        return U_0,F
    
    def getPolarizationEnergyForceResp(self,qm_positions,multipole_order,position_units="Bohr",units_out="AU"):
        '''
        Gets a dictionary of the polarization energy response as a dictionary containing the U_n and info about the 
        units used. The assumed expansion form is
        U_pol = U_0 + sum_a q_a U_a + (1/2)sum_ab q_a q_b U_ab 
        '''
        #if not hasattr(self,"resp_mode_force"):
        #    self.resp_mode_force = "linear"
        # set positions
        self.setQMPositions(qm_positions,units_in=position_units)
        
        # get the expansion given the multipole order
        if multipole_order==0:
            if self.resp_mode_force == "quadratic":
                U_0,U_1,U_2,F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm = self.getTestParticleChargeMultipoleEnergyForcesExpansion()
            elif self.resp_mode_force == "linear":
                U_0,U_1,U_2,F_0_mm,F_0_qm,F_1_mm,F_1_qm = self.getChargePolarizationResponseLinearScaling(get_forces=True)
                F_2_mm = F_2_qm = None
        elif multipole_order==1:
            if self.resp_mode_force == "quadratic":
                U_0,U_1,U_2,F_0_mm,F_0_qm,F_1_mm,F_1_qm,F_2_mm,F_2_qm = self.getTestParticleChargeDipoleMultipoleEnergyForcesExpansion()
            elif self.resp_mode_force == "linear":
                U_0,U_1,U_2,F_0_mm,F_0_qm,F_1_mm,F_1_qm = self.getChargeDipolePolarizationResponseLinearScaling(get_forces=True)
                F_2_mm = F_2_qm = None
                run_test = False
                if run_test:
                    U_0x,U_1x,U_2x,F_0_mmx,F_0_qmx,F_1_mmx,F_1_qmx,F_2_mmx,F_2_qmx = self.getTestParticleChargeDipoleMultipoleEnergyForcesExpansion()
                    print("U_0")
                    print(U_0x-U_0)
                    print("U_1")
                    print(U_1-U_1x)
                    print("U_2")
                    print(U_2-U_2x)
                    print(np.max(np.abs(U_2-U_2x)))
                    print("F_0_mm")
                    print(F_0_mm - F_0_mmx)
                    print("F_0_qm")
                    print(F_0_qm - F_0_qmx)
                    print("F_1_mm")
                    print(F_1_mm-F_1_mmx)
                    print("F_1_qm")
                    print(F_1_qm-F_1_qmx)
        else:
            raise Exception("Multipole order currently must be 0 or 1. multipole_order =",multipole_order,"is not allowed.")
        
        
        N_QM = F_1_qm.shape[1]
        
        if self.damp_perm:
            dU_1,dF_1_mm, dF_1_qm = self.getPermMultipoleDampCorrectionCharge(multipole_order,get_force_corr=True)
            #print(U_1)
            #print(dU_1)
            U_1 += dU_1
            #print(U_1)
            
            #print(F_1_mm)
            #print(dF_1_mm)
            F_1_mm += dF_1_mm
            F_1_qm += dF_1_qm
            #print(np.max(np.abs(U_1)), np.max(np.abs(F_1_mm)), np.max(np.abs(F_1_qm)))
            
            U_1_uncorr = U_1 - dU_1
            F_1_mm_uncorr = F_1_mm - dF_1_mm
            F_1_qm_uncorr = F_1_qm - dF_1_qm
            #dU_1,dF_1_mm, dF_1_qm,U_test,F_mm_test,F_qm_test = self.getPermMultipoleDampCorrectionCharge(multipole_order,get_force_corr=True,do_test=True)
            #print("damp perm test")
            #print("U_1")
            #print(U_1_uncorr-U_test)
            #print("F_1_mm")
            #print(F_1_mm_uncorr-F_mm_test)
            #print("F_1_qm")
            #print(F_1_qm_uncorr-F_qm_test)
            
            F_1_mm_uncorr = np.permute_dims(F_1_mm_uncorr,axes=(1,2,0))
            F_1_qm_uncorr = np.permute_dims(F_1_qm_uncorr,axes=(1,2,0))
            
            
        
        # rearrange the force objects to N_atm x 3 x N_Q x N_Q
        F_1_mm = np.permute_dims(F_1_mm,axes=(1,2,0))
        F_1_qm = np.permute_dims(F_1_qm,axes=(1,2,0))
        if F_2_mm is not None:
            F_2_mm = np.permute_dims(F_2_mm,axes=(2,3,0,1))
        if F_2_qm is not None:
            F_2_qm = np.permute_dims(F_2_qm,axes=(2,3,0,1))
        
        # collect everything together into a dictionary and convert units as needed
        if units_out in ["AU","au"]:
            Eh = Data.HARTREE_TO_KJMOL # hartree in kJ/mol
            a0 = Data.BOHR_TO_NM # bohr in nm
            Eh_per_a0 = Eh / a0
            conv = np.ones(U_1.shape)
            N_Q = conv.shape[0]
            if multipole_order==1:
                N_QM = int(N_Q/4)
                conv[N_QM:N_Q] = a0 
            U_0 = U_0/Eh
            U_1 = conv*U_1/(Eh)
            U_2 = (np.outer(conv,conv))*U_2/(Eh)
            F_0_mm = F_0_mm / Eh_per_a0
            F_0_qm = F_0_qm / Eh_per_a0
            F_1_mm = conv.reshape((1,1,N_Q))*F_1_mm /Eh_per_a0
            F_1_qm = conv.reshape((1,1,N_Q))*F_1_qm /Eh_per_a0
            if F_2_mm is not None:
                F_2_mm = (np.outer(conv,conv)).reshape((1,1,N_Q,N_Q))*F_2_mm/Eh_per_a0
            if F_2_qm is not None:
                F_2_qm = (np.outer(conv,conv)).reshape((1,1,N_Q,N_Q))*F_2_qm/Eh_per_a0
            if self.damp_perm:
                U_1_uncorr = conv*U_1_uncorr / Eh
                F_1_mm_uncorr = conv[None,None,:] * F_1_mm_uncorr /Eh_per_a0
                F_1_qm_uncorr = conv[None,None,:] * F_1_qm_uncorr /Eh_per_a0
        
        
        
        pol_resp = {"U_0":U_0,"U_1":U_1,"U_2":U_2,"F_0_mm":F_0_mm,"F_0_qm":F_0_qm,
                    "F_1_mm":F_1_mm,"F_1_qm":F_1_qm,
                    "F_2_mm":F_2_mm,"F_2_qm":F_2_qm,"units":units_out} 
        if self.damp_perm:
            pol_resp["U_1_uncorr"] = U_1_uncorr
            pol_resp["F_1_mm_uncorr"] = F_1_mm_uncorr
            pol_resp["F_1_qm_uncorr"] = F_1_qm_uncorr
        
        
        
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
            conv = 1.0/Data.BOHR_TO_NM
        
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
            energy = self.simulation.context.getState(getEnergy=True,enforcePeriodicBox=True).getPotentialEnergy()
        elif terms == "remainder":
            energy = self.simulation.context.getState(getEnergy=True,enforcePeriodicBox=True,groups=1).getPotentialEnergy()
        
        if units_out in ["AU","au","Hartree","hartree"]:
            Eh = Data.HARTREE_TO_KJMOL # hartree in kJ/mol
            conv = 1.0/Eh
        elif units_out in ["OpenMM","kJ/mol"]:
            conv = 1.0
        
        return energy._value * conv
    
    def getEnergyForces(self,terms=None,units_out="AU"):
        '''
        gets the energy of the MM syste
        '''
        
        if terms is None:
            state = self.simulation.context.getState(getEnergy=True,getForces=True,enforcePeriodicBox=True)       
        elif terms == "remainder":
            state = self.simulation.context.getState(getEnergy=True,getForces=True,enforcePeriodicBox=True,groups=1)
        energy = state.getPotentialEnergy()
        force = state.getForces(asNumpy=True)
        if units_out in ["AU","au"]:
            Eh = Data.HARTREE_TO_KJMOL # hartree in kJ/mol
            a0 = Data.BOHR_TO_NM # bohr in nm
            Eh_per_a0 = Eh / a0
            conv_energy = 1.0/Eh
            conv_force = 1.0/Eh_per_a0
        elif units_out in ["OpenMM"]:
            conv_energy = 1.0
            conv_force = 1.0
        
        return energy._value * conv_energy, force._value * conv_force
    
    def getPBC(self,units="AU"):
        '''
        Gets whether PBC are used and returns pbc box dimensions if they are used
        '''
        is_pbc = self.multipole_force.usesPeriodicBoundaryConditions()
        if not is_pbc:
            return None
        else:
            if units in ["Bohr","bohr","AU","au"]:
                conv = Data.NM_TO_BOHR
            elif units in ["nanometer","NM","nm","openmm","OpenMM"]:
                conv = 1.0
            pbc_dims = self.multipole_topology.getUnitCellDimensions()._value 
            return np.array(pbc_dims) * conv
        
    def createDirectPolSimulation(self,zero_perm=False):
        # create a new system object with the test charges/dipoles
        self.multipole_force_dir = AmoebaMultipoleForce()
        for force in self.multipole_system.getForces():
            if force.getName() == "AmoebaMultipoleForce":
                self.multipole_force_dir = deepcopy(force)
                # set the polarization type to direct
                self.multipole_force_dir.setPolarizationType(self.multipole_force_dir.Direct)
                self.multipole_force_dir.setNonbondedMethod(force.getNonbondedMethod())
                if force.getNonbondedMethod() == force.PME:
                    #print("Setting PME parameters")
                    self.multipole_force_dir.setPMEParameters(*self.multipole_force.getPMEParametersInContext(self.multipole_simulation.context))
                    self.multipole_force_dir.setCutoffDistance(force.getCutoffDistance())
        
        self.multipole_force_dir.setMutualInducedTargetEpsilon(self.induced_dipole_error)
        self.multipole_force_dir.setMutualInducedMaxIterations(self.max_iter_induced)
        # create a new system object for test charges/dipoles and delete all forces
        self.multipole_system_dir = deepcopy(self.multipole_system)
        if self.multipole_force.usesPeriodicBoundaryConditions():
            self.multipole_system_dir.setDefaultPeriodicBoxVectors(*self.multipole_system.getDefaultPeriodicBoxVectors())
        for i in range(0,self.multipole_system_dir.getNumForces()):
            self.multipole_system_dir.removeForce(0)
        
        N_MM = self.multipole_force_dir.getNumMultipoles()-4
        # get the polarizabilities of each site
        self.alpha_MM = np.zeros((N_MM,))
        for k in range(0,N_MM):
            params_k = self.multipole_force_dir.getMultipoleParameters(k)
            #print(params_k[-1])
            self.alpha_MM[k] = params_k[-1]._value+0
            if zero_perm:
                c = 0.0
                d = [0.,0.,0.]
                q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
                axis_type = self.multipole_force_dir.ZOnly
                kz = 0
                kx = 0
                ky = 0
                # THIS PART MIGHT BREAK STUFF! THOLE/DAMP should be the same
                thole = self.thole_default
                damp = self.damp_default
                pol = params_k[-1]._value+0
                self.multipole_force_dir.setMultipoleParameters(k,c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            
        
        self.alpha_MM_inv = np.zeros((N_MM,))
        self.alpha_MM_inv[self.alpha_MM>0.0] = 1.0 / (4.0*np.pi*EPSILON_0_OPENMM*self.alpha_MM[self.alpha_MM>0.0])
        
        # add four particles to the force, initially set with zero charge,dipole etc.
        #for i in range(0,4):
        #    c = 0.0
        #    d = [0.,0.,0.]
        #    q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
        #    axis_type = self.multipole_force_dir.ZOnly
        #    kz = 0
        #    kx = 0
        #    ky = 0
        #    thole = self.thole_default
        #    damp = self.damp_default
        #    pol = 0.0
        #    # add the particle to the force object
        #    self.multipole_force_dir.addMultipole(c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
        #    # add the particle to the system as well
        #    self.multipole_system_dir.addParticle(0.0)
        
        # update the covalent maps for each test particle to remove nans
        for i in range(0,4):
            self.multipole_force_dir.setCovalentMap(N_MM+i,self.multipole_force_dir.Covalent12,[N_MM+j for j in range(0,4) if not j==i])
            self.multipole_force_dir.setCovalentMap(N_MM+i,self.multipole_force_dir.PolarizationCovalent11,[N_MM+j for j in range(0,4) if not j==i])
        
        # add the new multipole force to the system
        self.multipole_system_dir.addForce(self.multipole_force_dir)
        
        # get the extended system topology and add the test particles
        self.multipole_topology_dir = deepcopy(self.multipole_simulation.topology)
     
        # set up the new Simulation object for the multipole force
        integrator = VerletIntegrator(1e-16)
        #try:
        #    platform = simulation.platform
        #except:
        #    platform = None
        platform = Platform.getPlatformByName(self.multipole_simulation.context.getPlatform().getName())
        properties = getPlatformPropertiesInContext(self.multipole_simulation.context)
        self.multipole_simulation_dir = app.Simulation(self.multipole_topology_dir,self.multipole_system_dir,integrator,
                                              platform,platformProperties=properties)
        

        
        #print("Dir PBC?:",self.multipole_force_dir.usesPeriodicBoundaryConditions(),self.multipole_simulation_dir.topology.getPeriodicBoxVectors())
        #print("PME:",self.multipole_force_dir.getPMEParametersInContext(self.multipole_simulation_dir.context))
        #print("Cutoff:",self.multipole_force_dir.getCutoffDistance())
        
        return
    
    def createPolSimulation(self,zero_perm=False):
        # create a new system object with the test charges/dipoles
        self.multipole_force_pol = AmoebaMultipoleForce()
        for force in self.system.getForces():
            if force.getName() == "AmoebaMultipoleForce":
                self.multipole_force_pol = deepcopy(force)
                # set the polarization type to direct
                #self.multipole_force_pol.setPolarizationType(self.multipole_force_pol.Direct)
                self.multipole_force_pol.setNonbondedMethod(force.getNonbondedMethod())
                if force.getNonbondedMethod() == force.PME:
                    self.multipole_force_pol.setPMEParameters(*self.multipole_force.getPMEParametersInContext(self.multipole_simulation.context))
                    self.multipole_force_pol.setCutoffDistance(force.getCutoffDistance())
        
        self.multipole_force_pol.setMutualInducedTargetEpsilon(self.induced_dipole_error)
        self.multipole_force_pol.setMutualInducedMaxIterations(self.max_iter_induced)
        # create a new system object for test charges/dipoles and delete all forces
        self.multipole_system_pol = deepcopy(self.system)
        for i in range(0,self.multipole_system_pol.getNumForces()):
            self.multipole_system_pol.removeForce(0)
        
        N_MM = self.multipole_force_pol.getNumMultipoles()
        # get the polarizabilities of each site
        #self.alpha_MM = np.zeros((N_MM,))
        for k in range(0,N_MM):
            params_k = self.multipole_force_pol.getMultipoleParameters(k)
            #print(params_k[-1])
            #self.alpha_MM[k] = params_k[-1]._value+0
            if zero_perm:
                c = 0.0
                d = [0.,0.,0.]
                q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
                axis_type = self.multipole_force_pol.ZOnly
                kz = 0
                kx = 0
                ky = 0
                # THIS PART MIGHT BREAK STUFF! THOLE/DAMP should be the same
                thole = self.thole_default
                damp = self.damp_default
                pol = 0*params_k[-1]._value+0
                self.multipole_force_pol.setMultipoleParameters(k,c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            
        #print(self.alpha_MM)     
        #self.alpha_MM_inv = np.zeros((N_MM,))
        #self.alpha_MM_inv[self.alpha_MM>0.0] = 1.0 / (4.0*np.pi*EPSILON_0_OPENMM*self.alpha_MM[self.alpha_MM>0.0])
        
        # add four particles to the force, initially set with zero charge,dipole etc.
        for i in range(0,4):
            c = 0.0
            d = [0.,0.,0.]
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = self.multipole_force_pol.ZOnly
            kz = 0
            kx = 0
            ky = 0
            thole = self.thole_default
            damp = self.damp_default
            pol = 0.0
            # add the particle to the force object
            self.multipole_force_pol.addMultipole(c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            # add the particle to the system as well
            self.multipole_system_pol.addParticle(0.0)
            
            
        
        # update the covalent maps for each test particle to remove nans
        for i in range(0,4):
            self.multipole_force_pol.setCovalentMap(N_MM+i,0,[N_MM+j for j in range(0,4) if not j==i])
            self.multipole_force_pol.setCovalentMap(N_MM+i,4,[N_MM+j for j in range(0,4) if not j==i])
        
        # add the new multipole force to the system
        self.multipole_system_pol.addForce(self.multipole_force_pol)
        
        # get the extended system topology and add the test particles
        self.multipole_topology_pol = deepcopy(self.simulation.topology)
        #print("NumAtoms:",self.multipole_topology.getNumAtoms(),"NumRes:",self.multipole_topology.getNumResidues(),
        #     "NumChains:",self.multipole_topology.getNumChains())
        test_chain = self.multipole_topology_pol.addChain()
        test_res = self.multipole_topology_pol.addResidue("PRB",test_chain)
        for i in range(0,4):
            self.multipole_topology_pol.addAtom(str(i+1),app.Element.getByAtomicNumber(2),test_res)
        #print("NumAtoms:",self.multipole_topology.getNumAtoms(),"NumRes:",self.multipole_topology.getNumResidues(),
        #     "NumChains:",self.multipole_topology.getNumChains())
        
        # set up the new Simulation object for the multipole force
        integrator = VerletIntegrator(1e-16)
        
        platform = Platform.getPlatformByName(self.multipole_simulation.context.getPlatform().getName())
        properties = getPlatformPropertiesInContext(self.multipole_simulation.context)
        self.multipole_simulation_pol = app.Simulation(self.multipole_topology_pol,self.multipole_system_pol,integrator,
                                              platform,platformProperties=properties)

        
        return
    
    def createProbeSimulation(self,N_probes):
        # create a new system object with the test charges/dipoles
        self.multipole_force_probe = AmoebaMultipoleForce()
        for force in self.system.getForces():
            if force.getName() == "AmoebaMultipoleForce":
                self.multipole_force_probe = deepcopy(force)
                self.multipole_force_probe.setNonbondedMethod(force.getNonbondedMethod())
                if force.getNonbondedMethod() == force.PME:
                    #print("Setting PME parameters")
                    self.multipole_force_probe.setPMEParameters(*self.multipole_force.getPMEParametersInContext(self.multipole_simulation.context))
                    self.multipole_force_probe.setCutoffDistance(force.getCutoffDistance())
        
        self.multipole_force_probe.setMutualInducedTargetEpsilon(self.induced_dipole_error)
        self.multipole_force_probe.setMutualInducedMaxIterations(self.max_iter_induced)
        # create a new system object for test charges/dipoles and delete all forces
        self.multipole_system_probe = deepcopy(self.system)
        if self.multipole_system_probe.usesPeriodicBoundaryConditions():
            self.multipole_system_probe.setDefaultPeriodicBoxVectors(*self.multipole_system.getDefaultPeriodicBoxVectors())
        for i in range(0,self.multipole_system_probe.getNumForces()):
            self.multipole_system_probe.removeForce(0)
        
        N_MM = self.multipole_force_probe.getNumMultipoles()
        # add N_probes probe particles and N_probes reference particles for defining the probe particle dipole moments
        c = 0.0
        d = [0.]*3
        q = [0.]*9
        axis_type = self.multipole_force_probe.NoAxisType
        kx = ky = -1
        kz = -1
        pol = 0.0
        thole = self.thole_default
        damp = self.damp_default
        # first add the reference particles
        for i in range(0,N_probes):
            # add the particle to the force object
            self.multipole_force_probe.addMultipole(c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            # add the particle to the system as well
            self.multipole_system_probe.addParticle(0.0)
        # second add the "real" probe particles
        axis_type = self.multipole_force_probe.ZOnly
        for i in range(0,N_probes):
            thole = self.qm_thole[i]
            damp = self.qm_damp[i]
            kz = N_MM + i
            # add the particle to the force object
            self.multipole_force_probe.addMultipole(c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            # add the particle to the system as well
            self.multipole_system_probe.addParticle(0.0)
        
        # update the covalent maps for each test particle to remove nans
        for i in range(0,2*N_probes):
            self.multipole_force_probe.setCovalentMap(N_MM+i,self.multipole_force_probe.Covalent12,[N_MM+j for j in range(0,2*N_probes) if not j==i])
            self.multipole_force_probe.setCovalentMap(N_MM+i,self.multipole_force_probe.PolarizationCovalent11,[N_MM+j for j in range(0,2*N_probes) if not j==i])
        
        # add the new multipole force to the system
        self.multipole_system_probe.addForce(self.multipole_force_probe)
        
        # get the extended system topology and add the test particles
        self.multipole_topology_probe = deepcopy(self.simulation.topology)
        test_chain = self.multipole_topology_probe.addChain()
        test_res = self.multipole_topology_probe.addResidue("PRB",test_chain)
        for i in range(0,2*N_probes):
            self.multipole_topology_probe.addAtom(str(i+1),app.Element.getByAtomicNumber(2),test_res)
        atoms = [a for a in self.multipole_topology_probe.atoms()]
        for i in range(0,2*N_probes):
            for j in range(0,2*N_probes):
                if not i==j:
                    self.multipole_topology_probe.addBond(atoms[i+N_MM],atoms[j+N_MM])
        # set up the new Simulation object for the multipole force
        integrator = VerletIntegrator(1e-16)
        #try:
        #    platform = self.simulation.platform
        #except:
        #    platform = None
        platform = Platform.getPlatformByName(self.multipole_simulation.context.getPlatform().getName())
        properties = getPlatformPropertiesInContext(self.multipole_simulation.context)
        self.multipole_simulation_probe = app.Simulation(self.multipole_topology_probe,self.multipole_system_probe,integrator,
                                              platform,platformProperties=properties)
        
        return

    def setProbeForcePositionsAndMultipoles(self,positions,charges,dipoles=None,use_prelim_mpole=False):
        '''
        Sets probe force 
        positions is assumed to be in openmm format
        '''
        N_probes = len(charges)
        N_MM = self.system.getNumParticles()
        positions_ref = deepcopy(positions)
        dx = 1.0e-4
        
        for i in range(0,N_probes):
            c = charges[i]
            
            if dipoles is not None:
                mod_d = np.linalg.norm(np.array(dipoles[i]))
                if mod_d>1.0e-10:
                    d = [0.,0.,mod_d]
                else:
                    d = [0.]*3
            else:
                d = [0.]*3
            kx = ky = -1
            q = [0.]*9
            pol = 0.
            kz = N_MM + i
            axis_type = self.multipole_force_probe.ZOnly
            thole = self.qm_thole[i]
            damp = self.qm_damp[i]
            if not use_prelim_mpole:
                self.multipole_force_probe.setMultipoleParameters(N_MM+N_probes+i,c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
                if dipoles is not None and mod_d>1.0e-10:
                    d_i = dipoles[i]
                    positions_ref[i] = positions_ref[i] + ((dx/d[2])*Vec3(d_i[0],d_i[1],d_i[2])) * positions_ref[i].unit
                else:
                    positions_ref[i] = positions_ref[i] + (Vec3(1.,0.,0.)*dx) * positions_ref[i].unit
            else:
                if dipoles is None:
                    self.multipole_force_probe.setMultipoleParameters(N_MM+N_probes+i,c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
                    positions_ref[i] = positions_ref[i] + (Vec3(1.,0.,0.)*dx) * positions_ref[i].unit
                else:
                    c0,c1,x0,x1 = self.getPrelimDipoleChargePosition(c,dipoles[i],positions[i]._value)
                    
                    self.multipole_force_probe.setMultipoleParameters(N_MM+N_probes+i,c0,[0.]*3,q,self.multipole_force_probe.NoAxisType,-1,-1,-1,thole,damp,pol)
                    self.multipole_force_probe.setMultipoleParameters(N_MM+i,c1,[0.]*3,q,self.multipole_force_probe.NoAxisType,-1,-1,-1,thole,damp,pol)
                    positions[i] = x0*positions[i].unit
                    positions_ref[i] = x1*positions[i].unit
                
        self.multipole_force_probe.updateParametersInContext(self.multipole_simulation_probe.context)
        self.multipole_simulation_probe.context.setPositions(self.positions+positions_ref+positions)
        return
    
    def getProbeForces(self,probe_positions,probe_charges,probe_dipoles=None,use_prelim_mpole=None):
        '''
        Gets forces on full set of probe particles.
        Units in are assumed to be atomic units in numpy arrays
        '''
        if use_prelim_mpole is None:
            use_prelim_mpole = self.use_prelim_mpole

        # first convert to lists and Vec3
        N_probes = probe_positions.shape[0]
        probe_positions =  BOHR_TO_NM * probe_positions
        probe_positions = [Vec3(probe_positions[i,0],probe_positions[i,1],probe_positions[i,2]) for i in range(0,N_probes)] * unit.nanometer
        probe_charges = list(probe_charges)
        if probe_dipoles is not None:
            probe_dipoles = probe_dipoles *  BOHR_TO_NM
            probe_dipoles = [[probe_dipoles[i,0],probe_dipoles[i,1],probe_dipoles[i,2]] for i in range(0,N_probes)]
        
        if not hasattr(self,"multipole_simulation_probe"):
            self.createProbeSimulation(N_probes)
        
        self.setProbeForcePositionsAndMultipoles(probe_positions,probe_charges,dipoles=probe_dipoles,use_prelim_mpole=use_prelim_mpole)
        
        F = self.multipole_simulation_probe.context.getState(getForces=True).getForces(asNumpy=True)._value
        N_MM = self.system.getNumParticles()
        F_MM = F[0:N_MM,:] 
        F_probe = F[N_MM:(N_MM+N_probes),:] + F[(N_MM+N_probes):(N_MM+2*N_probes),:]
        
        F_MM *= ( KJMOL_TO_HARTREE /  NM_TO_BOHR) 
        F_probe *= ( KJMOL_TO_HARTREE /  NM_TO_BOHR) 
        
        return F_probe, F_MM
    
    def getChargePolarizationResponseLinearScaling(self,get_forces=False):
        '''
        Method for getting the polarization response expansion in a linear scaling way
        '''
        if not hasattr(self, 'multipole_simulation_dir'):
            self.createDirectPolSimulation()
        #self.createPolSimulation()
        inc_pol=False
        N_MM = self.system.getNumParticles()
        N_QM = len(self.qm_positions)
        N_Q = N_QM
        c = self.test_charge
        if not get_forces:
            U_MM = self.getRefMultipoleEnergy()._value
        else:
            U_MM,F = self.getRefMultipoleEnergyForces()
            F_0_mm = F[0:N_MM,:]+0.
            F_0_qm = np.zeros((N_QM,3))
            
        self.setTestParticleCharge([0.0,0.0,0.0,0.0],[0,1,2,3],inc_dir=True,inc_pol=inc_pol)
        self.setProbePositions(self.positions,inc_dir=True,inc_pol=inc_pol)
        mu_MM = np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]
        #mu_MM = np.array(self.multipole_force_pol.getInducedDipoles(self.multipole_simulation_pol.context))[0:N_MM,:]
        
        f_MM = self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]
        U_lin = np.zeros((N_QM,))
        U_diag = np.zeros((N_QM,))
        U_plus = np.zeros((N_QM,))
        U_minus = np.zeros((N_QM,))
        mu_QM = np.zeros((N_QM,N_MM,3))
        #mu_QM_minus = np.zeros((N_QM,N_MM,3))
        f_QM = np.zeros((N_QM,N_MM,3))
        #f_QM_minus = np.zeros((N_QM,N_MM,3))
        if get_forces:
            F_plus_qm = np.zeros((N_Q,N_QM,3))
            F_plus_mm = np.zeros((N_Q,N_MM,3))
            F_minus_qm = np.zeros((N_Q,N_QM,3))
            F_minus_mm = np.zeros((N_Q,N_MM,3))
        
        # get energies of system with a test particle energy of +c
        start = timer()

        for A in range(0,N_QM):
            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True,inc_pol=inc_pol)
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-3*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions,inc_dir=True,inc_pol=inc_pol)
            if not get_forces:
                U_plus[A] = self.getMultipoleEnergy()._value
            if get_forces:
                U_plus[A],F = self.getMultipoleEnergyForces()
                F_plus_qm[A,A,:] = F[N_MM,:]+0.
                F_plus_mm[A,0:N_MM,:] = F[0:N_MM,:]+0.
            mu_QM[A,:,:] = np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]-mu_MM
            #mu_QM[A,:,:] = np.array(self.multipole_force_pol.getInducedDipoles(self.multipole_simulation_pol.context))[0:N_MM,:] - mu_MM
            f_QM[A,:,:] = self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]-f_MM
            self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True,inc_pol=inc_pol)
            if not get_forces:
                U_minus[A] = self.getMultipoleEnergy()._value
            if get_forces:
                U_minus[A],F = self.getMultipoleEnergyForces()
                F_minus_qm[A,A,:] = F[N_MM,:]+0.
                F_minus_mm[A,0:N_MM,:] = F[0:N_MM,:]+0.
            
            #f_QM_minus[A,:,:] = (self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]-f_MM)
            #f_QM[A,:,:] = 0.5*(f_QM[A,:,:]-f_QM_minus[A,:,:])
            #
            #mu_QM_minus[A,:,:] = np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]-mu_MM
            #mu_QM[A,:,:] = 0.5  * (mu_QM[A,:,:] - mu_QM_minus[A,:,:])
            
        # get energies of system with a test particle energy of -c
        #self.setTestParticleCharge(-c,0)
        #for A in range(0,N_QM):
        #    self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True)
        #    test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-3*i,0,0) for i in range(1,4)])*unit.nanometer
        #    self.setProbePositions(self.positions,test_positions=test_positions,inc_dir=True)
        #    U_minus[A] = self.getMultipoleEnergy()._value
        #    # Could do averaging but hopefully not necessary
        
        U_2 = -np.einsum('Akx,Bkx->AB',mu_QM,f_QM)/(c*c)
        U_2 = 0.5*(U_2 + U_2.T)
        #print(U_2)
        U_diag = ((U_plus + U_minus) - 2.0*U_MM) / (c*c)

        #print(U_2)
        
        U_1 = 0.5*(U_plus - U_minus)/c
        #U_1 = (U_plus-U_MM-0.5*c*c*np.diag(U_2))/c
        if get_forces:
            F_1_mm = (F_plus_mm - F_minus_mm) * (0.5/c)
            F_1_qm = (F_plus_qm - F_minus_qm) * (0.5/c)
            #F_1_mm = (F_plus_mm - F_0_mm[None,:,:]) * (1.0/c)
            #F_1_qm = (F_plus_qm - F_0_qm[None,:,:]) * (1.0/c)
        
        if self.multipole_force.usesPeriodicBoundaryConditions():
            U_self = self.getSelfInteractionEnergy(multipole_order=0)
            U_2 += U_self
            
        
                
        if not get_forces:
            return U_MM,U_1,U_2
        else:
            return U_MM,U_1,U_2,F_0_mm,F_0_qm,F_1_mm,F_1_qm
    
    def getChargeDipolePolarizationResponseLinearScaling(self,get_forces=False):
        '''
        Method for getting the polarization response expansion in a linear scaling way
        if get_forces is True then the forces for the zeroth and first order terms are also returned
        '''
        
        if not hasattr(self, 'multipole_simulation_dir'):
            self.createDirectPolSimulation()
        #self.createPolSimulation()
        inc_pol = False
        N_MM = self.system.getNumParticles()
        N_QM = len(self.qm_positions)
        N_Q = 4 * N_QM
        c = self.test_charge
        
        if not get_forces:
            U_MM = self.getRefMultipoleEnergy()._value
        else:
            U_MM,F = self.getRefMultipoleEnergyForces()
            F_0_mm = F[0:N_MM,:]+0.
            F_0_qm = np.zeros((N_QM,3))
        self.setTestParticleCharge([0.0,0.0,0.0,0.0],[0,1,2,3],inc_dir=True,inc_pol=inc_pol)
        self.setProbePositions(self.positions,inc_dir=True,inc_pol=inc_pol)
        #mu_MM = np.array(self.multipole_force_pol.getInducedDipoles(self.multipole_simulation_pol.context))[0:N_MM,:]
        mu_MM = np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]
        f_MM = self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]
        U_plus = np.zeros((4*N_QM,))
        U_minus = np.zeros((4*N_QM,))
        U_diag = np.zeros((4*N_QM,))
        mu_QM = np.zeros((4*N_QM,N_MM,3))
        #mu_QM_minus = np.zeros((4*N_QM,N_MM,3))
        f_QM = np.zeros((4*N_QM,N_MM,3))
        #f_QM_minus = np.zeros((4*N_QM,N_MM,3))
        if get_forces:
            F_plus_qm = np.zeros((N_Q,N_QM,3))
            F_plus_mm = np.zeros((N_Q,N_MM,3))
            F_minus_qm = np.zeros((N_Q,N_QM,3))
            F_minus_mm = np.zeros((N_Q,N_MM,3))
        
        # get energies of system with a test particle energy of +c
        start = timer()
        self.setTestParticleCharge([0.0,0.0,0.0,0.0],[0,1,2,3],inc_dir=True,inc_pol=inc_pol)
        self.setProbePositions(self.positions,inc_dir=True,inc_pol=inc_pol)
        for A in range(0,N_QM):
            
            self.setTestParticleCharge(c,0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True,inc_pol=inc_pol)
            test_positions = ([self.qm_positions[A]._value]+[self.qm_positions[A]._value+Vec3(1e-3*i,0,0) for i in range(1,4)])*unit.nanometer
            self.setProbePositions(self.positions,test_positions=test_positions,inc_dir=True,inc_pol=inc_pol)
            
            
            if not get_forces:
                U_plus[A] = self.getMultipoleEnergy()._value
            if get_forces:
                U_plus[A],F = self.getMultipoleEnergyForces()
                F_plus_qm[A,A,:] = F[N_MM,:]+0.
                F_plus_mm[A,0:N_MM,:] = F[0:N_MM,:]+0.
            #mu_QM[A,:,:] = (np.array(self.multipole_force_pol.getInducedDipoles(self.multipole_simulation_pol.context))[0:N_MM,:]-mu_MM)/c
            
            mu_QM[A,:,:] = (np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]-mu_MM)/c
            
            f_QM[A,:,:] = (self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]-f_MM)/c
            self.setTestParticleCharge(-c,0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True,inc_pol=inc_pol)
            if not get_forces:
                U_minus[A] = self.getMultipoleEnergy()._value
            if get_forces:
                U_minus[A],F = self.getMultipoleEnergyForces()
                F_minus_qm[A,A,:] = F[N_MM,:]+0.
                F_minus_mm[A,0:N_MM,:] = F[0:N_MM,:]+0.
            
            #f_QM_minus[A,:,:] = (self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]-f_MM)/c
            #f_QM[A,:,:] = 0.5*(f_QM[A,:,:]-f_QM_minus[A,:,:])
            #mu_QM_minus[A,:,:] = (np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]-mu_MM)/c
            #mu_QM[A,:,:] = 0.5  * (mu_QM[A,:,:] - mu_QM_minus[A,:,:])
            
        n_alphas = [Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        d_0 = self.test_dipole
        self.setTestParticleCharge([0.0,0.0,0.0,0.0],[0,1,2,3],inc_dir=True,inc_pol=inc_pol)
        self.setProbePositions(self.positions,inc_dir=True,inc_pol=inc_pol)
        for alpha in range(0,3):
            n_alpha = n_alphas[alpha]
            for A in range(0,N_QM):
                alphaA = N_QM*(alpha+1) + A 
                if not self.use_prelim_mpole:
                    self.setTestParticleChargeDipole(0.0,[0,0,+d_0],[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True,inc_pol=inc_pol)
                    test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alpha*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                    self.setProbePositions(self.positions,test_positions=test_positions,inc_dir=True,inc_pol=inc_pol)
                else:
                    self.setTestParticlePrelimDipole([0.],[n_alpha*(d_0)],[self.qm_positions[A]._value],[[1,0]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]],inc_dir=True)
                    
                if not get_forces:
                    U_plus[alphaA] = self.getMultipoleEnergy()._value
                if get_forces:
                    U_plus[alphaA],F = self.getMultipoleEnergyForces()
                    F_plus_qm[alphaA,A,:] = F[N_MM,:]+F[N_MM+1,:]
                    F_plus_mm[alphaA,0:N_MM,:] = F[0:N_MM,:]+0.
                #mu_QM[alphaA,:,:] = (np.array(self.multipole_force_pol.getInducedDipoles(self.multipole_simulation_pol.context))[0:N_MM,:]-mu_MM)/d_0
                mu_QM[alphaA,:,:] = (np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]-mu_MM)/d_0
            
                f_QM[alphaA,:,:] = (self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]-f_MM)/d_0
                if not self.use_prelim_mpole:
                    self.setTestParticleChargeDipole(0.0,[0,0,-d_0],[N_MM+1,-1,-1],0,thole=self.qm_thole[A],damp=self.qm_damp[A],inc_dir=True,inc_pol=inc_pol)
                else:
                    self.setTestParticlePrelimDipole([0.],[n_alpha*(-d_0)],[self.qm_positions[A]._value],[[1,0]],thole=[self.qm_thole[A]],damp=[self.qm_damp[A]],inc_dir=True)
                    
                if not get_forces:
                    U_minus[alphaA] = self.getMultipoleEnergy()._value
                if get_forces:
                    U_minus[alphaA],F = self.getMultipoleEnergyForces()
                    F_minus_qm[alphaA,A,:] = F[N_MM,:]+F[N_MM+1,:]
                    F_minus_mm[alphaA,0:N_MM,:] = F[0:N_MM,:]+0.
                
                #f_QM_minus[alphaA,:,:] = (self.alpha_MM_inv[:,None] * np.array(self.multipole_force_dir.getInducedDipoles(self.multipole_simulation_dir.context))[0:N_MM,:]-f_MM)/d_0
                #f_QM[alphaA,:,:] = 0.5*(f_QM[alphaA,:,:]-f_QM_minus[alphaA,:,:])
                #mu_QM_minus[alphaA,:,:] = (np.array(self.multipole_force.getInducedDipoles(self.multipole_simulation.context))[0:N_MM,:]-mu_MM)/d_0
                #mu_QM[alphaA,:,:] = 0.5  * (mu_QM[alphaA,:,:] - mu_QM_minus[alphaA,:,:])

        U_2 = -np.einsum('Akx,Bkx->AB',mu_QM,f_QM)
        U_2 = 0.5*(U_2+U_2.T)
        
        U_diag[0:N_QM] = ((U_plus[0:N_QM] + U_minus[0:N_QM]) - 2.0*U_MM) / (c*c)
        U_diag[N_QM:] = ((U_plus[N_QM:] + U_minus[N_QM:]) - 2.0*U_MM) / (d_0*d_0)
        #np.einsum('aa->a',U_2)[:] = U_diag
        #print(U_2)
        #np.einsum('aa->a',U_2)[:] = U_diag
        U_1 = np.zeros((4*N_QM,))
        
        U_1[0:N_QM] = 0.5*(U_plus[0:N_QM] - U_minus[0:N_QM])/c
        U_1[N_QM:] = 0.5*(U_plus[N_QM:] - U_minus[N_QM:])/d_0
        #U_2_diag = np.diag(U_2)
        #U_1[0:N_QM] = (U_plus[0:N_QM] - U_MM - 0.5 * c*c*U_2_diag[0:N_QM])/c
        #U_1[N_QM:] = (U_plus[N_QM:] - U_MM- 0.5 * d_0*d_0*U_2_diag[N_QM:])/d_0
        if get_forces:
            F_1_qm = np.zeros((N_Q,N_QM,3))
            F_1_mm = np.zeros((N_Q,N_MM,3))
            F_1_mm[0:N_QM,:,:] = (F_plus_mm[0:N_QM,:,:] - F_minus_mm[0:N_QM,:,:]) * (0.5/c)
            F_1_qm[0:N_QM,:,:] = (F_plus_qm[0:N_QM,:,:] - F_minus_qm[0:N_QM,:,:]) * (0.5/c)
            F_1_mm[N_QM:,:,:] = (F_plus_mm[N_QM:,:,:] - F_minus_mm[N_QM:,:,:]) * (0.5/d_0)
            F_1_qm[N_QM:,:,:] = (F_plus_qm[N_QM:,:,:] - F_minus_qm[N_QM:,:,:]) * (0.5/d_0)
            #F_1_mm[0:N_QM,:,:] = (F_plus_mm[0:N_QM,:,:] - F_0_mm[None,:,:]) * (1.0/c)
            #F_1_qm[0:N_QM,:,:] = (F_plus_qm[0:N_QM,:,:] - F_0_qm[None,:,:]) * (1.0/c)
            #F_1_mm[N_QM:,:,:] = (F_plus_mm[N_QM:,:,:] - F_0_mm[None,:,:]) * (1.0/d_0)
            #F_1_qm[N_QM:,:,:] = (F_plus_qm[N_QM:,:,:] - F_0_qm[None,:,:]) * (1.0/d_0)
        #U_1[0:N_QM] = (U_plus[0:N_QM] - U_MM - (0.5*c*c)*np.diag(U_2)[0:N_QM])/c
        #U_1[N_QM:] = (U_plus[N_QM:] - U_MM - (0.5*c*c)*np.diag(U_2)[N_QM:])/d_0
        
        if self.multipole_force.usesPeriodicBoundaryConditions():
            U_self = self.getSelfInteractionEnergy(multipole_order=1)
            U_2 += U_self

        if not get_forces:
            return U_MM,U_1,U_2
        else:
            return U_MM,U_1,U_2,F_0_mm,F_0_qm,F_1_mm,F_1_qm
    
    def createSelfIntSimulation(self):
        self.multipole_force_self = AmoebaMultipoleForce()
        for force in self.system.getForces():
            if force.getName() == "AmoebaMultipoleForce":
                # set the polarization type to direct
                self.multipole_force_self.setPolarizationType(self.multipole_force_self.Direct)
                self.multipole_force_self.setNonbondedMethod(self.multipole_force.getNonbondedMethod())
                if force.getNonbondedMethod() == force.PME:
                    self.multipole_force_self.setPMEParameters(*self.multipole_force.getPMEParametersInContext(self.multipole_simulation.context))
                    self.multipole_force_self.setCutoffDistance(force.getCutoffDistance())
                    #print("PME params:",self.multipole_force_self.getPMEParameters())
                    #print("cutoff:",self.multipole_force_self.getCutoffDistance())
                    
        self.multipole_system_self = System()
        self.multipole_system_self.setDefaultPeriodicBoxVectors(*self.multipole_system.getDefaultPeriodicBoxVectors())
        
        # add four particles to the force, initially set with zero charge,dipole etc.
        for i in range(0,4):
            c = 0.0
            d = [0.,0.,0.]
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = self.multipole_force_self.NoAxisType
            kz = -1
            kx = -1
            ky = -1
            thole = self.thole_default
            damp = self.damp_default
            pol = 0.0
            # add the particle to the force object
            self.multipole_force_self.addMultipole(c,d,q,axis_type,kz,kx,ky,thole,damp,pol)
            # add the particle to the system as well
            self.multipole_system_self.addParticle(0.0)

        # update the covalent maps for each test particle to remove nans
        for i in range(0,4):
            self.multipole_force_self.setCovalentMap(i,self.multipole_force_self.Covalent12,[+j for j in range(0,4) if not j==i])
            self.multipole_force_self.setCovalentMap(i,self.multipole_force_self.PolarizationCovalent11,[+j for j in range(0,4) if not j==i])
        
        # add the new multipole force to the system
        self.multipole_system_self.addForce(self.multipole_force_self)
        
        # get the extended system topology and add the test particles
        self.multipole_topology_self = app.Topology()
        self.multipole_topology_self.setUnitCellDimensions(self.simulation.topology.getUnitCellDimensions())
        #print("NumAtoms:",self.multipole_topology.getNumAtoms(),"NumRes:",self.multipole_topology.getNumResidues(),
        #     "NumChains:",self.multipole_topology.getNumChains())
        test_chain = self.multipole_topology_self.addChain()
        test_res = self.multipole_topology_self.addResidue("PRB",test_chain)
        for i in range(0,4):
            self.multipole_topology_self.addAtom(str(i+1),app.Element.getByAtomicNumber(2),test_res)
        for i in range(0,4):
            for j in range(0,4):
                if not i==j:
                    self.multipole_topology_self.addBond(i,j)

        # set up the new Simulation object for the multipole force
        integrator = VerletIntegrator(1e-16)
        #try:
        #    platform = simulation.platform
        #except:
        #    platform = None
        platform = Platform.getPlatformByName(self.multipole_simulation.context.getPlatform().getName())
        properties = getPlatformPropertiesInContext(self.multipole_simulation.context)
        self.multipole_simulation_self = app.Simulation(self.multipole_topology_self,self.multipole_system_self,integrator,
                                              platform,platformProperties=properties)
        
        return
    
    
    def setTestParticleChargeDipoleSelf(self,c,d,k,ind,thole=None,damp=None):
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
            index = i
            q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
            axis_type = self.multipole_force.ZOnly
            kz = k_i[0]
            kx = k_i[1]
            ky = k_i[2]
            pol = 0.0
            # set the particle parameters
            self.multipole_force_self.setMultipoleParameters(index,c_i,d_i,q,axis_type,kz,kx,ky,thole_i,damp_i,pol)
            
        
        self.multipole_force_self.updateParametersInContext(self.multipole_simulation_self.context)
        
        return
    
    def getSelfInteractionEnergy(self,multipole_order=0):
        if not hasattr(self, 'multipole_simulation_self'):
            self.createSelfIntSimulation()
        
        #for force in self.multipole_simulation_self.context._system.getForces():
        #    print(force)
        #    print(force.getPMEParametersInContext(self.multipole_simulation_self.context))
        #    print(force.getCutoffDistance())
        #print(self.multipole_simulation_self.topology.getUnitCellDimensions())
        if multipole_order == 0 :
            charge = [self.test_charge]
            dipole = [[0.,0.,0.]]
            multipole = [self.test_charge]
            q_max = 1
        elif multipole_order == 1 :
            charge = [self.test_charge,0.,0.,0.]
            dipole = [[0.,0.,0.],[0.,0.,self.test_dipole],[0.,0.,self.test_dipole],[0.,0.,self.test_dipole]]
            multipole = [self.test_charge,self.test_dipole,self.test_dipole,self.test_dipole]
            q_max = 4
        n_alphas = [Vec3(1,0,0),Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)]
        k_nodip = [0,0,0]
        N_QM = len(self.qm_positions)
        N_Q = N_QM * len(multipole)
        U_self = np.zeros((N_Q,N_Q))
        damp = [self.damp_default]*4
        # diagonal terms
        #epsilon_grid = 0.001 
        #grid = []
        #n_grid = [Vec3(1,0,0),Vec3(-1,0,0),Vec3(0,1,0),Vec3(0,-1,0),Vec3(0,0,1),Vec3(0,0,-1)]
        #for A in range(0,N_QM):
        #    for v in n_grid:
        #        grid.append(epsilon_grid*v)
        #potential_grid = np.zeros((N_Q,len(grid)))
        for q in range(0,len(charge)):
            for A in range(0,N_QM):
                self.setTestParticleChargeDipoleSelf([charge[q],0.,0.,0.],[dipole[q],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],[[1,0,0],k_nodip,k_nodip,k_nodip],[0,1,2,3],damp=damp)
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+n_alphas[q]*1e-2]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                self.multipole_simulation_self.context.setPositions(test_positions)
                qA = N_QM*q +A
                U_self[qA,qA] = 2.0*self.multipole_simulation_self.context.getState(getEnergy=True).getPotentialEnergy()._value / (multipole[q]*multipole[q])
                #potential_grid[qA,:] = np.array(self.multipole_force_self.getElectrostaticPotential(grid,self.multipole_simulation_self.context))
                
                
        # charge-charge terms
        q = 0
        for A in range(0,N_QM):
            for B in range(0,A):
                self.setTestParticleChargeDipoleSelf([charge[q],charge[q],0.,0.],[dipole[q],dipole[q],[0.,0.,0.],[0.,0.,0.]],[[1,-1,-1],k_nodip,k_nodip,k_nodip],[0,1,2,3],damp=damp)
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[B]._value]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                self.multipole_simulation_self.context.setPositions(test_positions)
                qA = N_QM*q +A
                qB = N_QM*q +B
                U = self.multipole_simulation_self.context.getState(getEnergy=True).getPotentialEnergy()._value
                U_self[qA,qB] = (U - 0.5*(multipole[q]*multipole[q])*(U_self[qA,qA]+U_self[qB,qB]))/(multipole[q]*multipole[q])
                U_self[qB,qA] = U_self[qA,qB] + 0.
        
        # dipole-dipole terms        
        for q1 in range(1,q_max):
            for q2 in range(1,q_max):
                for A in range(0,N_QM):
                    for B in range(0,N_QM):
                        if not A==B:
                            self.setTestParticleChargeDipoleSelf([charge[q1],charge[q2],0.,0.],[dipole[q1],dipole[q2],[0.,0.,0.],[0.,0.,0.]],[[2,-1,-1],[3,-1,-1],k_nodip,k_nodip],[0,1,2,3],damp=damp)
                            test_positions = ([self.qm_positions[A]._value,self.qm_positions[B]._value,self.qm_positions[A]._value+1e-2*n_alphas[q1],self.qm_positions[B]._value+1e-2*n_alphas[q2]])*unit.nanometer
                            self.multipole_simulation_self.context.setPositions(test_positions)
                            q1A = N_QM*q1 +A
                            q2B = N_QM*q2 +B
                            U = self.multipole_simulation_self.context.getState(getEnergy=True).getPotentialEnergy()._value
                            U_self[q1A,q2B] = (U-0.5*(multipole[q1]*multipole[q1])*U_self[q1A,q1A] -0.5*(multipole[q2]*multipole[q2])*U_self[q2B,q2B])/(multipole[q1]*multipole[q2])
                            U_self[q2B,q1A] = U_self[q1A,q2B]+0.
        for q1 in range(1,q_max):
            for q2 in range(1,q1):
                for A in range(0,N_QM):
                    dipole_12 = [0.,0.,np.sqrt(2.0)*self.test_dipole]
                    self.setTestParticleChargeDipoleSelf([0.,0.,0.,0.],[dipole_12,[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],[[1,-1,-1],k_nodip,k_nodip,k_nodip],[0,1,2,3],damp=damp)
                    test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+1e-2*n_alphas[q1]+1e-2*n_alphas[q2]]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                    self.multipole_simulation_self.context.setPositions(test_positions)
                    q1A = q1*N_QM + A
                    q2A = q2*N_QM + A
                    U = self.multipole_simulation_self.context.getState(getEnergy=True).getPotentialEnergy()._value
                    U_self[q1A,q2A] = (U-0.5*(multipole[q1]*multipole[q1])*U_self[q1A,q1A] -0.5*(multipole[q2]*multipole[q2])*U_self[q2A,q2A])/(multipole[q1]*multipole[q2])
                    U_self[q2A,q1A] = U_self[q1A,q2A]+0.
        # charge-dipole term
        q1 = 0
        for q2 in range(1,q_max):
            for A in range(0,N_QM):
                for B in range(0,N_QM):
                    if not A==B:
                        self.setTestParticleChargeDipoleSelf([charge[q1],charge[q2],0.,0.],[dipole[q1],dipole[q2],[0.,0.,0.],[0.,0.,0.]],[[2,-1,-1],[3,-1,-1],k_nodip,k_nodip],[0,1,2,3],damp=damp)
                        test_positions = ([self.qm_positions[A]._value,self.qm_positions[B]._value,self.qm_positions[A]._value+1e-2*n_alphas[q1],self.qm_positions[B]._value+1e-2*n_alphas[q2]])*unit.nanometer
                        self.multipole_simulation_self.context.setPositions(test_positions)
                        q1A = N_QM*q1 +A
                        q2B = N_QM*q2 +B
                        U = self.multipole_simulation_self.context.getState(getEnergy=True).getPotentialEnergy()._value
                        U_self[q1A,q2B] = (U-0.5*(multipole[q1]*multipole[q1])*U_self[q1A,q1A] -0.5*(multipole[q2]*multipole[q2])*U_self[q2B,q2B])/(multipole[q1]*multipole[q2])
                        U_self[q2B,q1A] = U_self[q1A,q2B]+0.
        for q2 in range(1,q_max):
            for A in range(0,N_QM):
                dipole_12 = [0.,0.,self.test_dipole]
                self.setTestParticleChargeDipoleSelf([charge[q1],0.,0.,0.],[dipole_12,[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]],[[1,-1,-1],k_nodip,k_nodip,k_nodip],[0,1,2,3],damp=damp)
                test_positions = ([self.qm_positions[A]._value,self.qm_positions[A]._value+1e-2*n_alphas[q2]]+[self.qm_positions[A]._value+Vec3(1e-5*i,0,0) for i in range(2,4)])*unit.nanometer
                self.multipole_simulation_self.context.setPositions(test_positions)
                q1A = q1*N_QM + A
                q2A = q2*N_QM + A
                U = self.multipole_simulation_self.context.getState(getEnergy=True).getPotentialEnergy()._value
                U_self[q1A,q2A] = (U-0.5*(multipole[q1]*multipole[q1])*U_self[q1A,q1A] -0.5*(multipole[q2]*multipole[q2])*U_self[q2A,q2A])/(multipole[q1]*multipole[q2])
                U_self[q2A,q1A] = U_self[q1A,q2A]+0.
                            
        return U_self
    
    def setTestParticlePrelimDipole(self,q,d,x,inds,thole=None,damp=None,inc_dir=False):
        '''
        Sets up pre-limit form of dipole(s), d[j], with total charges q[j] at positions x[j]
        '''
        if thole is None:
            thole = [self.thole_default]*len(q)
        if damp is None:
            damp = [self.damp_default]*len(q)
        N_MM = self.system.getNumParticles()
        zero_Q = [0.]*9
        zero_D = [0.]*3
        zero_pol = 0.
        axis_type = self.multipole_force.NoAxisType
        kz = kx = ky = -1
        test_positions = [x[0]+Vec3(1.0e-5*(i+1),0.,0.) for i in range(0,4)]
        for j in range(0,len(q)):
            x_j = x[j]
            d_j = np.array(d[j])
            mod_d_j = np.linalg.norm(d_j)
            if mod_d_j >0:
                n_j = d_j / mod_d_j
                q_j = q[j]
                x_A = x_j + (self.prelim_dr*0.5) * n_j 
                x_B = x_j - (self.prelim_dr*0.5) * n_j 
                q_A = 0.5*q_j + mod_d_j / self.prelim_dr
                q_B = 0.5*q_j - mod_d_j / self.prelim_dr
            else:
                q_A = q[j]
                q_B = 0.
                x_A = x_j*1.
                x_B = test_positions[inds[j][1]]
                
            thole_j = thole[j]
            damp_j = damp[j]
            A = inds[j][0]+N_MM
            B = inds[j][1]+N_MM
            
            self.multipole_force.setMultipoleParameters(A,q_A,zero_D,zero_Q,axis_type,kz,kx,ky,thole_j,damp_j,zero_pol)
            self.multipole_force.setMultipoleParameters(B,q_B,zero_D,zero_Q,axis_type,kz,kx,ky,thole_j,damp_j,zero_pol)
            test_positions[inds[j][0]] = x_A*1.
            test_positions[inds[j][1]] = x_B*1.
            if inc_dir:
                self.multipole_force_dir.setMultipoleParameters(A,q_A,zero_D,zero_Q,axis_type,kz,kx,ky,thole_j,damp_j,zero_pol)
                self.multipole_force_dir.setMultipoleParameters(B,q_B,zero_D,zero_Q,axis_type,kz,kx,ky,thole_j,damp_j,zero_pol)
        inds_set = [set[0] for set in inds] + [set[1] for set in inds]
        inds_reset = [i+N_MM for i in range(0,4) if i not in inds_set]
        for J in inds_reset:
            self.multipole_force.setMultipoleParameters(J,0.,zero_D,zero_Q,axis_type,kz,kx,ky,self.thole_default,self.damp_default,zero_pol)
        if inc_dir:
            for J in inds_reset:
                self.multipole_force_dir.setMultipoleParameters(J,0.,zero_D,zero_Q,axis_type,kz,kx,ky,self.thole_default,self.damp_default,zero_pol)
            
            
        self.multipole_force.updateParametersInContext(self.multipole_simulation.context)
        if inc_dir:
            self.multipole_force_dir.updateParametersInContext(self.multipole_simulation_dir.context)
            
        self.setProbePositions(self.positions,test_positions=test_positions*unit.nanometer,inc_dir=inc_dir)
        
    
        
        return
    
    def getPrelimDipoleChargePosition(self,q,d,x):
        
        x_j = x
        d_j = np.array(d)
        q_j = q 
        mod_d_j = np.linalg.norm(d_j)
        if mod_d_j >0:
            n_j = d_j / mod_d_j
            q_j = q
            x_A = x_j + (self.prelim_dr*0.5) * n_j 
            x_B = x_j - (self.prelim_dr*0.5) * n_j 
            q_A = 0.5*q_j + mod_d_j / self.prelim_dr
            q_B = 0.5*q_j - mod_d_j / self.prelim_dr
        else:
            q_A = q
            q_B = 0.
            x_A = x_j*1.
            x_B = x+Vec3(1.0e-5*(1),0.,0.)
        
        return q_A,q_B,x_A,x_B

    
    def getPermMultipoleDampCorrectionCharge(self,multipole_order,get_force_corr=False,do_test=False):
        
        pbc_dims = self.getPBC(units="nm")
        
        R_QM = np.array([[v.x,v.y,v.z] for v in self.qm_positions])
        R_MM = np.array([[v.x,v.y,v.z] for v in self.positions])

        N_QM = R_QM.shape[0]
        N_MM = R_MM.shape[0]
        if multipole_order == 0 :
            N_Q = N_QM
        else:
            N_Q = 4 * N_QM
            
        dU = np.zeros((N_Q,))
        U_test = np.zeros((N_Q,))
        if get_force_corr:
            dF_qm = np.zeros((N_Q,N_QM,3))
            dF_mm = np.zeros((N_Q,N_MM,3))
            F_qm_test = np.zeros((N_Q,N_QM,3))
            F_mm_test = np.zeros((N_Q,N_MM,3))
        
        if not hasattr(self,"tol_damp_corr_exp"):
            self.tol_damp_corr_exp = 1.0e-8
        tol = -np.log(self.tol_damp_corr_exp)
        #n_added = 0
        for A in range(0,N_QM):
            # get parameters
            rA = R_QM[A,:]
            tholeA = self.qm_thole[A]
            dampA = self.qm_damp[A]
            for B in range(0,N_MM):
                #get parameters
                rB = R_MM[B,:]
                params = self.multipole_force.getMultipoleParameters(B)
                tholeB = params[7]
                dampB = params[8]
                
                # determine if the A-B correction should be added
                
                if dampB < 1.0e-10:
                    damp = (dampA)**6
                    damp = tholeA / (damp)
                else:
                    damp = (dampA*dampB)**3
                    damp = min(tholeA,tholeB) / (damp)
                rBA = extra.getNearestImages(rB,rA,pbc=pbc_dims)
                r = np.linalg.norm(rBA)
                rB = rBA + rA
                add_corr = ((damp*r*r*r) < tol)
                #add_corr = True
                if add_corr:
                    #n_added += 1
                    cA = 1.0
                    cB = params[0]._value
                    dA = np.zeros((3,))
                    dB = np.array(params[1]._value)
                    qA = np.zeros((3,3))
                    qB = np.array(params[2]._value).reshape((3,3))
                    # need to convert dB and qB to the labframe
                    axis_type = params[3]
                    Z = params[4]
                    X = params[5]
                    Y = params[6]
                    
                    if params[4] >= 0: # Z atom
                        rZ = extra.getNearestImages(np.array(self.positions[Z]._value),R_MM[B,:])+rB
                    else:
                        rZ = None
                    if params[5] >= 0: # X atom
                        rX = extra.getNearestImages(np.array(self.positions[X]._value),R_MM[B,:])+rB
                    else:
                        rX = None
                    if params[6] >= 0: # Y atom
                        rY = extra.getNearestImages(np.array(self.positions[Y]._value),R_MM[B,:])+rB
                    else:
                        rY = None
                    dB, qB = extra.getLabFrameMultipoles(dB,qB,rB, axis_type,rZ=rZ,rX=rX,rY=rY)
                    if self.damp_chargedipole_only:
                        qB *= 0.0
                    if self.damp_charge_only:
                        dB *= 0.0
                        qB *= 0.0
                    if not get_force_corr:
                        du = - extra.getMultipolePairEnergy(rA,rB,cA,cB,dA,dB,qA,qB)
                        U_test[A] -= du 
                        du += extra.getMultipolePairTholeEnergy(rA,rB,cA,cB,dA,dB,qA,qB,damp)
                        #print(du)
                        dU[A] += du
                        
                    else:
                        du,dfA,dfB,dtauA,dtauB = extra.getMultipolePairEnergyForce(rA,rB,cA,cB,dA,dB,qA,qB)
                        dudamp,dfAdamp,dfBdamp,dtauAdamp,dtauBdamp = extra.getMultipolePairTholeEnergyForce(rA,rB,cA,cB,dA,dB,qA,qB,damp)
                        dU[A] += (dudamp - du)
                        dF_qm[A,A,:] += (dfAdamp - dfA)
                        dF_mm[A,B,:] += (dfBdamp - dfB)
                        # torque forces on the mm atoms
                        df_torque = extra.computeTorqueForces(dtauB, rB, rZ, posX=rX, posW=rY, axisType=axis_type)
                        df_torque_damp = extra.computeTorqueForces(dtauBdamp, rB, rZ, posX=rX, posW=rY, axisType=axis_type)
                        dF_mm[A,B,:] += df_torque_damp["forceA"]-df_torque["forceA"]
                        if rZ is not None:
                            dF_mm[A,Z,:] += df_torque_damp["forceZ"]-df_torque["forceZ"]
                        if rX is not None:
                            dF_mm[A,X,:] += df_torque_damp["forceX"]-df_torque["forceX"]
                        if rY is not None:
                            dF_mm[A,Y,:] += df_torque_damp["forceW"]-df_torque["forceW"]
                        
                        U_test[A] += du 
                        F_qm_test[A,A,:] += dfA
                        F_mm_test[A,B,:] += dfB
                        if rZ is not None:
                            F_mm_test[A,Z,:] += df_torque["forceZ"]
                        if rX is not None:
                            F_mm_test[A,X,:] += df_torque["forceX"]
                        if rY is not None:
                            F_mm_test[A,Y,:] += df_torque["forceW"]
                        
                    if multipole_order > 0 :
                        cA = 0.
                        for alpha in range(0,3):
                            alphaA = N_QM*(1+alpha)+A
                            dA = np.zeros((3,))
                            dA[alpha] = 1.0
                            if not get_force_corr:
                                if not self.use_prelim_mpole:
                                    du = - extra.getMultipolePairEnergy(rA,rB,cA,cB,dA,dB,qA,qB)
                                    U_test[A] -= du 
                                    du += extra.getMultipolePairTholeEnergy(rA,rB,cA,cB,dA,dB,qA,qB,damp)
                                else:
                                    c0,c1,x0,x1 = self.getPrelimDipoleChargePosition(0.,dA,rA)
                                    du = - extra.getMultipolePairEnergy(x0,rB,c0,cB,np.zeros((3,)),dB,qA,qB)
                                    du -= extra.getMultipolePairEnergy(x1,rB,c1,cB,np.zeros((3,)),dB,qA,qB)
                                    U_test[A] -= du 
                                    du += extra.getMultipolePairTholeEnergy(x0,rB,c0,cB,np.zeros((3,)),dB,qA,qB,damp)
                                    du += extra.getMultipolePairTholeEnergy(x1,rB,c1,cB,np.zeros((3,)),dB,qA,qB,damp)
                                    
                                #print(du)
                                dU[alphaA] += du
                            else:
                                if not self.use_prelim_mpole:
                                    du,dfA,dfB,dtauA,dtauB = extra.getMultipolePairEnergyForce(rA,rB,cA,cB,dA,dB,qA,qB)
                                    dudamp,dfAdamp,dfBdamp,dtauAdamp,dtauBdamp = extra.getMultipolePairTholeEnergyForce(rA,rB,cA,cB,dA,dB,qA,qB,damp)
                                    dU[alphaA] += (dudamp - du)
                                    dF_qm[alphaA,A,:] += (dfAdamp - dfA)
                                    dF_mm[alphaA,B,:] += (dfBdamp - dfB)
                                    # torque forces on the mm atoms
                                    df_torque = extra.computeTorqueForces(dtauB, rB, rZ, posX=rX, posW=rY, axisType=axis_type)
                                    df_torque_damp = extra.computeTorqueForces(dtauBdamp, rB, rZ, posX=rX, posW=rY, axisType=axis_type)
                                else:
                                    c0,c1,x0,x1 = self.getPrelimDipoleChargePosition(0.,dA,rA)
                                    du0,dfA0,dfB0,dtauA0,dtauB0 = extra.getMultipolePairEnergyForce(x0,rB,c0,cB,np.zeros((3,)),dB,qA,qB)
                                    du1,dfA1,dfB1,dtauA1,dtauB1 = extra.getMultipolePairEnergyForce(x1,rB,c1,cB,np.zeros((3,)),dB,qA,qB)
                                    du0damp,dfA0damp,dfB0damp,dtauA0damp,dtauB0damp = extra.getMultipolePairTholeEnergyForce(x0,rB,c0,cB,np.zeros((3,)),dB,qA,qB,damp)
                                    du1damp,dfA1damp,dfB1damp,dtauA1damp,dtauB1damp = extra.getMultipolePairTholeEnergyForce(x1,rB,c1,cB,np.zeros((3,)),dB,qA,qB,damp)
                                    
                                    dU[alphaA] += (du0damp+du1damp - (du0+du1))
                                    dF_qm[alphaA,A,:] += (dfA0damp+dfA1damp - (dfA0+dfA1))
                                    dF_mm[alphaA,B,:] += (dfB0damp+dfB1damp - (dfB0+dfB1))
                                    # torque forces on the mm atoms
                                    df_torque = extra.computeTorqueForces(dtauB0+dtauB1, rB, rZ, posX=rX, posW=rY, axisType=axis_type)
                                    df_torque_damp = extra.computeTorqueForces(dtauB0damp+dtauB1damp, rB, rZ, posX=rX, posW=rY, axisType=axis_type)
                                    
                                dF_mm[alphaA,B,:] += df_torque_damp["forceA"]-df_torque["forceA"]
                                if rZ is not None:
                                    dF_mm[alphaA,Z,:] += df_torque_damp["forceZ"]-df_torque["forceZ"]
                                if rX is not None:
                                    dF_mm[alphaA,X,:] += df_torque_damp["forceX"]-df_torque["forceX"]
                                if rY is not None:
                                    dF_mm[alphaA,Y,:] += df_torque_damp["forceW"]-df_torque["forceW"]
                                
                                U_test[alphaA] += du 
                                F_qm_test[alphaA,A,:] += dfA
                                F_mm_test[alphaA,B,:] += dfB
                                if rZ is not None:
                                    F_mm_test[alphaA,Z,:] += df_torque["forceZ"]
                                if rX is not None:
                                    F_mm_test[alphaA,X,:] += df_torque["forceX"]
                                if rY is not None:
                                    F_mm_test[alphaA,Y,:] += df_torque["forceW"]
                            
                            
        #print(dU)
        #print("Number of damping corrections added:", n_added)
        if do_test:
            if not get_force_corr:
                return dU,U_test
            else:
                return dU, dF_mm, dF_qm,U_test,F_mm_test,F_qm_test
        if not get_force_corr:
            return dU
        else:
            return dU, dF_mm, dF_qm
        
    def getCPRepulsion(self,get_force=False,units_out="AU"):
        pbc_dims = self.getPBC(units="nm")
        
        R_QM = np.array([[v.x,v.y,v.z] for v in self.qm_positions])
        R_MM = np.array([[v.x,v.y,v.z] for v in self.positions])

        N_QM = R_QM.shape[0]
        N_MM = R_MM.shape[0]

        U = 0.0
        if get_force:
            F_qm = np.zeros((N_QM,3))
            F_mm = np.zeros((N_MM,3))
        
        if not hasattr(self,"tol_damp_corr_exp"):
            self.tol_damp_corr_exp = 1.0e-8
        tol = -np.log(self.tol_damp_corr_exp)
        #n_added = 0
        for A in range(0,N_QM):
            # get parameters
            rA = R_QM[A,:]
            tholeA = self.qm_thole[A]
            dampA = self.qm_damp[A]
            for B in range(0,N_MM):
                #get parameters
                rB = R_MM[B,:]
                params = self.multipole_force.getMultipoleParameters(B)
                tholeB = params[7]
                dampB = params[8]
                
                # determine if the A-B correction should be added
                if dampB < 1.0e-10:
                    damp = (dampA)**6
                    damp = tholeA / (damp)
                else:
                    damp = (dampA*dampB)**3
                    damp = min(tholeA,tholeB) / (damp)
                rBA = extra.getNearestImages(rB,rA,pbc=pbc_dims)
                r = np.linalg.norm(rBA)
                rB = rBA + rA
                add_corr = ((damp*r*r*r) < tol)
                #add_corr = True
                if add_corr:
                    ZA = self.Z_QM[A]
                    ZB = self.Z_MM[B]
                    if get_force:
                        U_AB, fA, fB = extra.getDampedCoreRepEnergyForce(rA,rB,ZA,ZB,damp,do_force=True)
                        U += U_AB
                        F_qm[A,:] += fA 
                        F_mm[B,:] += fB
                    else:
                        U_AB = extra.getDampedCoreRepEnergyForce(rA,rB,ZA,ZB,damp,do_force=False)
                        U += U_AB
        if units_out in ["AU","au"]:
            conv_U = Data.KJMOL_TO_HARTREE
            conv_F = Data.KJMOL_TO_HARTREE / Data.NM_TO_BOHR
        elif units_out in ["OpenMM","openmm"]:
            conv_U = 1.0
            conv_F = 1.0 
        
        if get_force:
            return conv_U*U, conv_F*F_qm, conv_F*F_mm
        else:
            return conv_U*U
                    
    
        
        

            