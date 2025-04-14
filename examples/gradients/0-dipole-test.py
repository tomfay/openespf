# import OpenMM for setting up the MM part of the calculation
from openmm.app import *
from openmm import *
from openmm.unit import *

#

system = System()

c = 0.4
pol = 0.001 
damp = pol**1.0/6.0
thole = 0.39 
dip = 0.01
d = [0.,0.,dip] 
q = [0.0]*9
dx = 0.0001
c_dip = dip / dx 

force = AmoebaMultipoleForce()
force.setNonbondedMethod(force.NoCutoff)


# add a reference particle
force.addMultipole(0.0,[0.,0.,0.],q,force.NoAxisType,-1,-1,-1,thole,damp,0.0)
# add the particle with a dipole
force.addMultipole(0.0,d,q,force.ZOnly,0,-1,-1,thole,damp,0.0)

# add the charged particle
force.addMultipole(c,[0.,0.,0.],q,force.NoAxisType,-1,-1,-1,thole,damp,pol)
force.addMultipole(c,[0.,0.,0.],q,force.NoAxisType,-1,-1,-1,thole,damp,pol)
force.addMultipole(-2*c,[0.,0.,0.],q,force.NoAxisType,-1,-1,-1,thole,damp,pol)

# set covalent maps
force.setCovalentMap(0,force.Covalent12,[1])
force.setCovalentMap(0,force.PolarizationCovalent11,[1])
force.setCovalentMap(1,force.Covalent12,[0])
force.setCovalentMap(1,force.PolarizationCovalent11,[0])

force.setCovalentMap(2,force.Covalent12,[4])
force.setCovalentMap(2,force.Covalent13,[3])
force.setCovalentMap(2,force.PolarizationCovalent11,[3,4])
force.setCovalentMap(3,force.Covalent12,[4])
force.setCovalentMap(3,force.Covalent13,[2])
force.setCovalentMap(3,force.PolarizationCovalent11,[2,4])
force.setCovalentMap(4,force.Covalent12,[2,3])
force.setCovalentMap(4,force.PolarizationCovalent11,[2,3])

# system 
system = System()
for i in range(0,5):
    system.addParticle(1.0)

system.addForce(force)    
    
# topology
topology = Topology()
chain = topology.addChain()
ion = topology.addResidue("ION",chain)
dipole = topology.addResidue("DIP",chain)
topology.addAtom("D1",Element.getBySymbol("He"),dipole)
topology.addAtom("D2",Element.getBySymbol("He"),dipole)
topology.addAtom("M1",Element.getBySymbol("He"),ion)
topology.addAtom("M2",Element.getBySymbol("He"),ion)
topology.addAtom("M3",Element.getBySymbol("He"),ion)
atoms = [a for a in topology.atoms()]
topology.addBond(atoms[0],atoms[1])
topology.addBond(atoms[2],atoms[4])
topology.addBond(atoms[3],atoms[4])

# simulation
platform = Platform.getPlatformByName("Reference")
integrator = VerletIntegrator(1.0e-3)
simulation = Simulation(topology,system,integrator,platform)

# positions
positions = [Vec3(-0.1,0.,0.),Vec3(0.,0.,0.),Vec3(0.25,0.,0.),Vec3(0.25+0.12,0.,0.),Vec3(0.25+0.15,0.1,0.)]*nanometer
simulation.context.setPositions(positions)
state = simulation.context.getState(getEnergy=True,getForces=True)
energy = state.getPotentialEnergy()
forces = state.getForces(asNumpy=True)
print(energy)
print(forces)
force.setMultipoleParameters(0,-c_dip,[0.,0.,0.],q,force.NoAxisType,-1,-1,-1,thole,damp,0.0)
force.setMultipoleParameters(1,c_dip,[0.,0.,0.],q,force.NoAxisType,-1,-1,-1,thole,damp,0.0)
force.updateParametersInContext(simulation.context)
positions = [Vec3(0.5*dx,0.,0.),Vec3(-0.5*dx,0.,0.),Vec3(0.25,0.,0.),Vec3(0.25+0.12,0.,0.),Vec3(0.25+0.15,0.1,0.)]*nanometer
simulation.context.setPositions(positions)
state = simulation.context.getState(getEnergy=True,getForces=True)
energy = state.getPotentialEnergy()
forces = state.getForces(asNumpy=True)
print(energy)
print(forces)
print(forces[0,:]+forces[1,:])