import numpy as np
from scipy.special import erfc, gamma, gammaincc
from openmm import AmoebaMultipoleForce
from .Data import EPSILON_0_OPENMM

ONETHIRD = 1.0/3.0
GAMMA_TWOTHIRDS = gamma(2.0/3.0)
# prefactor for  electrostatics in OpenMM units, charges in e, distance in nm and energy in kJ/mol
PREFACTOR = 1.0 / (4.0*np.pi*EPSILON_0_OPENMM) 

def getDist(x_A_set,x_B,pbc=None):
    if pbc is None:
        dx = (x_A_set - x_B[None,:])
        return np.linalg.norm(dx,axis=1), dx
    else:
        dx = x_A_set - x_B[None,:]
        dx_ni = dx - pbc[None,:]*np.round(dx/pbc[None,:])
        return np.linalg.norm(dx_ni,axis=1), dx_ni

def getNearestImages(x_A_set,x_B,pbc=None):
    '''
    Returns the set of nearest image displacement vectors dx_AB = x_A,nearesttoB - x_B
    If pbc=None then it just returns x_A-x_B
    If pbc = np.array([Lx,Ly,Lz]) it assumes periodic boundary conditions.
    '''
    if pbc is None:
        if len(x_A_set.shape)>1:
            dx = (x_A_set - x_B[None,:])
            return dx
        else:
            dx = (x_A_set - x_B)
            return dx
    else:
        if len(x_A_set.shape)>1:
            dx = x_A_set - x_B[None,:]
            dx_ni = dx - pbc[None,:]*np.round(dx/pbc[None,:])
            return dx_ni
        else:
            dx = x_A_set - x_B
            dx_ni = dx - pbc*np.round(dx/pbc)
            return dx_ni

def getChargeChargeEnergy(q1,q2,r1,r2):
    '''
    Gets coulomb energy of two charges interacting
    U = q1 q2 / r
    r21 = r2 - r1 vector from 1 to 2
    '''
    r = np.linalg.norm(r2-r1)
    return PREFACTOR*q1*q2/r

def getChargeChargeDirectEnergy(q1,q2,r1,r2,alpha):
    '''
    Gets coulomb energy of two charges interacting with driect space ewald interaction
    U = q1 q2 erfc(alpha r) / r
    r21 = r2 - r1 vector from 1 to 2
    '''
    r = np.linalg.norm(r2-r1)
    alpha_r = alpha*r 
    return PREFACTOR*q1*q2 *erfc(alpha_r)/r

def getChargeChargeTholeEnergy(q1,q2,r1,r2,damp):
    '''
    U = lambda_cc q1 q2 / r
    r21 = r2 - r1 vector from 1 to 2
    In correcting energies, we never need to actually worry about the ewald damped part because thole factors are never applied
    damp = a / (alpha1 * alpha2)^1/6
    The damping we use is consistent with thole's exp(- damp * r^3) charge distirbution model
    '''
    r = np.linalg.norm(r2-r1)
    au3 = damp * r * r * r
    Gamma_incc = GAMMA_TWOTHIRDS * gammaincc(2.0/3.0,au3)
    f_thole = 1.0 + ((au3)**(1./3.)) * (Gamma_incc - ((au3)**(-1.0/3.0)) * np.exp(-au3))
    return (PREFACTOR*q1*q2*f_thole /r)

def getChargeChargeEnergyForce(q1,q2,r1,r2):
    '''
    U = lambda_cc q1 q2 / r
    r21 = r2 - r1 vector from 1 to 2
    In correcting energies, we never need to actually worry about the ewald damped part because thole factors are never applied
    damp = a / (alpha1 * alpha2)^1/6
    The damping we use is consistent with thole's exp(- damp * r^3) charge distirbution model
    '''
    r21 = r2-r1
    r = np.linalg.norm(r21)
    u = PREFACTOR*q1*q2 /r
    # force on 2
    f2 = PREFACTOR*((q1*q2)/(r*r*r)) * r21 # field at 2 * q2
    return u,-f2,f2 # energy, force on 1, force on 2

def getChargeChargeTholeEnergyForce(q1,q2,r1,r2,damp):
    '''
    U = lambda_cc q1 q2 / r
    r21 = r2 - r1 vector from 1 to 2
    In correcting energies, we never need to actually worry about the ewald damped part because thole factors are never applied
    damp = a_thole / (alpha1^1/6 * alpha2^1/6)^3
    The damping we use is consistent with thole's exp(- damp * r^3) charge distirbution model
    '''
    r21 = r2-r1
    r = np.linalg.norm(r21)
    au3 = damp * r * r * r
    # energy
    Gamma_incc = GAMMA_TWOTHIRDS * gammaincc(2.0/3.0,au3)
    f_thole = 1.0 + ((au3)**(1./3.)) * (Gamma_incc - ((au3)**(-1.0/3.0)) * np.exp(-au3))
    # energy
    u = PREFACTOR*q1*q2*f_thole /r
    # force on 2
    f2 = PREFACTOR*(q1*q2*(1.0-np.exp(-au3))/(r*r*r)) * r21 # field at 2 * q2
    return u,-f2,f2 # energy, force on 1, force on 2

def getChargeDipoleEnergy(q1,d2,r1,r2):
    '''
    returns the charge dipole energy
    U = -E1 . d2 = -(q1/r^2) (r21/r).d2
    '''
    r21 = r2-r1
    r = np.linalg.norm(r21)
    return PREFACTOR*(-q1/(r*r*r)) * r21.dot(d2)

def getChargeDipoleEnergyThole(q1,d2,r1,r2):
    '''
    returns the charge dipole energy
    U = -E1 . d2 = -(q1/r^2) (r21/r).d2
    '''
    r21 = r2-r1
    r = np.linalg.norm(r21)
    return PREFACTOR*(-q1/(r*r*r)) * r21.dot(d2)

def getMultipolePairTholeEnergy(rA,rB,qA,qB,dA,dB,QA,QB,damp,do_damp=True):
    return getMultipolePairTholeEnergyForce(rA,rB,qA,qB,dA,dB,QA,QB,damp,do_damp=do_damp,do_force=False)

def getMultipolePairTholeEnergyOld(rA,rB,qA,qB,dA,dB,QA,QB,damp,do_damp=True):
    
    # separation and distance
    rBA = rB - rA
    rAB = - rBA
    r = np.linalg.norm(rAB)
    
    # array of rinv values
    rinv = np.ones((9,))
    rinv[1] = 1.0/r
    for n in range(2,len(rinv)):
        rinv[n] = rinv[n-1]*rinv[1]
    
    # damping values
    if do_damp:
        au3 = damp * r * r * r
        exp_au3 = np.exp(-au3)
        Gamma_incc = GAMMA_TWOTHIRDS * gammaincc(2.0/3.0,au3)
        f = np.zeros((4,))
        f[0] = 1.0 + ((au3)**(1./3.)) * (Gamma_incc - ((au3)**(-1.0/3.0)) * exp_au3)
        f[1] = 1.0 - exp_au3
        f[2] = 1.0 - (1.0+au3)*exp_au3
        f[3] = 1.0 - (1.0+au3 + 0.6 * au3 * au3) *exp_au3
        #f[4] = 1.0 - (1.0+au3 + (18.0*au3*au3 + 9.0*au3*au3*au3)/35.)*exp_au3
    
    # B values
    B = np.zeros((4,))
    B[0] = rinv[1] # B_l = 
    for l in range(1,len(B)):
        B[l] = rinv[2] * (2*l-1) * B[l-1]
        
        
    # G values
    RAB = np.outer(rAB,rAB)
    DAB = np.outer(dA,rBA)
    DBA = np.outer(dB,rAB)
    G = np.zeros((4,))
    G[0] = qA*qB
    G[1] = qA * rAB.dot(dB) - dA.dot(rAB) * qB + dA.dot(dB)
    G[2] = - (dA.dot(rAB)) * (dB.dot(rAB)) + qA * np.sum(QB*RAB) + qB * np.sum(QA*RAB) \
        -2.0 * np.sum(QB * DAB) - 2.0 * np.sum(QA * DBA) # QA QB terms ignores
    G[3] = (dB.dot(rAB)) * np.sum(QA*RAB) - (dA.dot(rAB)) * np.sum(QB*RAB) # QA QB terms ignored
    
    # pair energy is a sum of these terms
    if do_damp:
        u = PREFACTOR*np.sum(f*B*G)
    else:
        u = PREFACTOR*np.sum(B*G)
        
    return u

def getMultipolePairEnergy(rA,rB,qA,qB,dA,dB,QA,QB):
    return getMultipolePairTholeEnergy(rA,rB,qA,qB,dA,dB,QA,QB,None,do_damp=False)

def getMultipolePairEnergyOld(rA,rB,qA,qB,dA,dB,QA,QB):
    
    # separation and distance
    rBA = rB - rA
    rAB = - rBA
    r = np.linalg.norm(rAB)
    
    # array of rinv values
    rinv = np.ones((9,))
    rinv[1] = 1.0/r
    for n in range(2,len(rinv)):
        rinv[n] = rinv[n-1]*rinv[1]
    
    # B values
    B = np.zeros((4,))
    B[0] = rinv[1]
    for l in range(1,len(B)):
        B[l] = rinv[2] * (2*l-1) * B[l-1]
        
    # G values
    RAB = np.outer(rAB,rAB)
    DAB = np.outer(dA,rBA)
    DBA = np.outer(dB,rAB)
    G = np.zeros((4,))
    G[0] = qA*qB
    G[1] = (qA * rAB.dot(dB)) - (dA.dot(rAB) * qB) + (dA.dot(dB))
    G[2] = - (dA.dot(rAB)) * (dB.dot(rAB)) + qA * np.sum(QB*RAB) + qB * np.sum(QA*RAB) \
        -2.0 * np.sum(QB * DAB) - 2.0 * np.sum(QA * DBA) # QA QB terms ignores
    G[3] = (dB.dot(rAB)) * np.sum(QA*RAB) - (dA.dot(rAB)) * np.sum(QB*RAB) # QA QB terms ignored
    
    # pair energy is a sum of these terms
    u = PREFACTOR*np.sum(B*G)
        
    return u

def getMultipolePairTholeEnergyForce(rA,rB,qA,qB,dA,dB,QA,QB,damp,do_damp=True,do_force=True):
    
    
    # separation and distance
    rBA = rB - rA
    rAB = - rBA
    r = np.linalg.norm(rAB)
    
    # array of rinv values
    rinv = np.ones((9,))
    rinv[1] = 1.0/r
    for n in range(2,len(rinv)):
        rinv[n] = rinv[n-1]*rinv[1]
    
    # B values
    B = np.zeros((5,))
    B[0] = rinv[1]
    for l in range(1,len(B)):
        B[l] = rinv[2] * (2*l-1) * B[l-1]
        
    # damping values
    if do_damp:
        au3 = damp * r * r * r
        exp_au3 = np.exp(-au3)
        Gamma_incc = GAMMA_TWOTHIRDS * gammaincc(2.0/3.0,au3) # Gamma(2/3,au^3)
        f = np.zeros((4,))
        f[0] = 1.0 + ((au3)**(1./3.)) * (Gamma_incc - ((au3)**(-1.0/3.0)) * exp_au3)
        f[1] = 1.0 - exp_au3
        f[2] = 1.0 - (1.0+au3)*exp_au3
        f[3] = 1.0 - (1.0 + au3 + 0.6 * au3 * au3) *exp_au3
        #f[4] = 1.0 - (1.0+au3 + (18.0*au3*au3 + 9.0*au3*au3*au3)/35.)*exp_au3
        # derivatives of the damping terms wrt au3
        df = np.zeros((4,))
        #df[0] = (Gamma_incc +( 1.0 - au3**(2./3.) )*exp_au3)
        df[0] = (Gamma_incc) / (3. * (au3)**(2./3.) )
        df[1] = exp_au3
        df[2] = au3*exp_au3 
        df[3] = (-0.2*au3 + 0.6 * au3*au3) * exp_au3
        # chain rule to get df/dr
        dau3_r = 3.0*damp*r # (1/r) * d(au^3)/dr
        df *= dau3_r 
        
    # G values
    RAB = np.outer(rAB,rAB)
    RBA = np.outer(rBA,rBA)
    DAB = np.outer(dA,rBA)
    DBA = np.outer(dB,rAB)
    G = np.zeros((4,))
    G[0] = qA*qB
    G[1] = (qA * rAB.dot(dB)) - (qB * dA.dot(rAB) ) + (dA.dot(dB))
    G[2] = qA * np.sum(QB*RAB) + qB * np.sum(QA*RAB) +2.0 * np.sum(QB * DAB) + 2.0 * np.sum(QA * DBA) \
        - (dA.dot(rAB)) * (dB.dot(rAB)) #+ 2.0 * np.sum(QA*QB)
    #G[2] = - (dA.dot(rAB)) * (dB.dot(rAB)) + qA * np.sum(QB*RAB) + qB * np.sum(QA*RAB) \
    #    -2.0 * np.sum(QB * DAB) - 2.0 * np.sum(QA * DBA) # QA QB terms ignores
    G[3] = (dB.dot(rAB)) * np.sum(QA*RAB) - (dA.dot(rAB)) * np.sum(QB*RAB) # QA QB terms ignored
    # derivatives of G values wrt rB
    dG = np.zeros((4,3))
    dG[1,:] = qB * dA - qA * dB
    dG[2,:] = - 2.0*qB * QA.dot(rAB) -2.0*qA * QB.dot(rAB)  \
        + 2.0 * QB.dot(dA) - 2.0* QA.dot(dB)  \
        + (dA.dot(rAB)) * dB +  (dB.dot(rAB)) * dA
    dG[3,:] = - np.sum(QA*RAB) * dB - 2.0*(dB.dot(rAB)) * QA.dot(rAB) \
        + np.sum(QB*RAB) * dA + 2.0 *(dA.dot(rAB)) * QB.dot(rAB) 
    
    #G[3] *= 0.
    #dG[3,:] *= 0.
    
    # pair energy is a sum of these terms
    if do_damp:
        u = PREFACTOR*np.sum(f*B[0:4]*G)
    else:
        u = PREFACTOR*np.sum(B[0:4]*G)
    # - (sum_l=1 B_l dG_l [dG_0 = 0] + sum B_{l+1} G_l rAB)
    if do_damp:
        fB = - np.sum(f[1:4,None]*B[1:4,None]*dG[1:4,:],axis=0) - np.sum(f[0:4,None] * B[1:5,None] * G[0:4,None] * rAB[None,:],axis=0) 
        # thole part
        fB +=  np.sum(df[0:4,None] * B[0:4,None] * G[0:4,None] * rAB[None,:],axis=0 )  
    else:
        fB = - np.sum(B[1:4,None]*dG[1:4,:],axis=0) - np.sum( B[1:5,None] * G[0:4,None] * rAB[None,:],axis=0) 
    fB *= PREFACTOR
     #torques on A and B are sum_l B_l Gt_l 
    # The relevant expressions are in Smith CCP5
    GtB = np.zeros((4,3))
    GtB[0,:] = 0.0 # GtA_0 is zero - included for readability
    GtB[1,:] = qA * np.cross(rAB,dB) + np.cross(dA,dB)
    # original
    GtB[2,:] = (2.0*qA)*cross_matrix(RAB,QB) \
        - (dA.dot(rAB)) * np.cross(rAB,dB) \
        + 2.0 * cross_matrix(DAB,QB) \
        + 2.0 * cross_matrix(DAB.T,QB) \
        - 2.0 * np.cross(QA.dot(rAB),dB)
    GtB[3,:] = np.sum(QA*RAB) * np.cross(rAB,dB) \
        - (2.0*dA.dot(rAB)) * cross_matrix(RAB,QB) 
    #GtB[3,:] *= 0.
    # for A
    GtA = np.zeros((4,3))
    GtA[0,:] = 0.0  # GtB_0 is zero - included for readability
    GtA[1,:] = qB * np.cross(rBA, dA) + np.cross(dB, dA)
    GtA[2,:] = (2.0*qB)*cross_matrix(RBA,QA) \
        - (dB.dot(rBA)) * np.cross(rBA,dA) \
        + 2.0 * cross_matrix(DBA,QA) \
        + 2.0 * cross_matrix(DBA.T,QA) \
        - 2.0 * np.cross(QB.dot(rBA),dA)
    GtA[3,:] = np.sum(QA * RAB) * np.cross(rAB, dB) \
        - (1.0 * dA.dot(rAB)) * cross_matrix(RAB, QB)
    if do_damp:
        tauA = PREFACTOR * np.sum(GtA[1:,:]*B[1:4,None]*f[1:,None],axis=0)
        tauB = PREFACTOR * np.sum(GtB[1:,:]*B[1:4,None]*f[1:,None],axis=0)
    else:
        tauA = PREFACTOR * np.sum(GtA[1:,:]*B[1:4,None],axis=0)
        tauB = PREFACTOR * np.sum(GtB[1:,:]*B[1:4,None],axis=0)
    
    if do_force:
        return u, -fB, fB, tauA, tauB
    else:
        return u

def getMultipolePairEnergyForce(rA,rB,qA,qB,dA,dB,QA,QB):
    return getMultipolePairTholeEnergyForce(rA,rB,qA,qB,dA,dB,QA,QB,None,do_damp=False)

def getMultipolePairEnergyForceOld(rA,rB,qA,qB,dA,dB,QA,QB):
    
    
    # separation and distance
    rBA = rB - rA
    rAB = - rBA
    r = np.linalg.norm(rAB)
    
    # array of rinv values
    rinv = np.ones((9,))
    rinv[1] = 1.0/r
    for n in range(2,len(rinv)):
        rinv[n] = rinv[n-1]*rinv[1]
    
    # B values
    B = np.zeros((5,))
    B[0] = rinv[1]
    for l in range(1,len(B)):
        B[l] = rinv[2] * (2*l-1) * B[l-1]
        
    # G values
    RAB = np.outer(rAB,rAB)
    RBA = np.outer(rBA,rBA)
    DAB = np.outer(dA,rBA)
    DBA = np.outer(dB,rAB)
    G = np.zeros((4,))
    G[0] = qA*qB
    G[1] = (qA * rAB.dot(dB)) - (dA.dot(rAB) * qB) + (dA.dot(dB))
    G[2] = - (dA.dot(rAB)) * (dB.dot(rAB)) + qA * np.sum(QB*RAB) + qB * np.sum(QA*RAB) \
        -2.0 * np.sum(QB * DAB) - 2.0 * np.sum(QA * DBA) # QA QB terms ignores
    G[3] = (dB.dot(rAB)) * np.sum(QA*RAB) - (dA.dot(rAB)) * np.sum(QB*RAB) # QA QB terms ignored
    # derivatives of G values wrt rB
    dG = np.zeros((4,3))
    dG[1,:] = - qB * dA + qA * dB
    dG[2,:] = -2.0*qA * QB.dot(rAB) - 2.0*qB * QA.dot(rAB) + 2.0* QA.dot(dB) - 2.0 * QB.dot(dA) \
        + (dA.dot(rAB)) * dB +  (dB.dot(rAB)) * dA
    dG[3,:] = - 2.0*(dB.dot(rAB)) * QA.dot(rAB) + 2.0 *(dA.dot(rAB)) * QB.dot(rAB) \
        + np.sum(QB*RAB) * dA - np.sum(QA*RAB) * dB
    
    # pair energy is a sum of these terms
    u = PREFACTOR*np.sum(B[0:4]*G)
    # - (sum_l=1 B_l dG_l [dG_0 = 0] + sum B_{l+1} G_l rAB)
    fB = - np.sum(B[1:4,None]*dG[1:4,:],axis=0) - np.sum(B[1:5,None] * G[0:4,None] * rAB[None,:],axis=0) 
    fB *= PREFACTOR
        
     #torques on A and B are sum_l B_l Gt_l 
    # The relevant expressions are in 
    GtB = np.zeros((4,3))
    GtB[0,:] = 0.0 # GtA_0 is zero - included for readability
    GtB[1,:] = qA * np.cross(rAB,dB) + np.cross(dA,dB)
    GtB[2,:] = (2.0*qA)*cross_matrix(RAB,QB) - 2.0* cross_matrix(np.outer(dA,rAB)+np.outer(rAB,dA),QB) \
        - 2.0 * np.cross(QA.dot(rAB),dB) - (dA.dot(rAB)) * np.cross(rAB,dB)
    GtB[3,:] = np.sum(QA*RAB) * np.cross(rAB,dB) - (2.0*dA.dot(rAB)) * cross_matrix(RAB,QB)
    # for A
    GtA = np.zeros((4,3))
    GtA[0,:] = 0.0  # GtB_0 is zero - included for readability
    GtA[1,:] = qB * np.cross(rBA, dA) + np.cross(dB, dA)
    GtA[2,:] = (2.0 * qB) * cross_matrix(RBA, QA) - 2.0 * cross_matrix(np.outer(dB, rBA) + np.outer(rBA, dB), QA) \
             - 2.0 * np.cross(QB.dot(rBA), dA) - (dB.dot(rAB)) * np.cross(rBA,dA)
    GtA[3,:] = np.sum(QB * RBA) * np.cross(rBA, dA) - (2.0 * dB.dot(rBA)) * cross_matrix(RBA, QA)
    
    tauA = PREFACTOR * np.sum(GtA[1:4,:]*B[1:4,None],axis=0)
    tauB = PREFACTOR * np.sum(GtB[1:4,:]*B[1:4,None],axis=0)
    return u, -fB, fB, tauA, tauB

def buildRotationMatrix(axisType, rA, rZ, rX, rY):
    """
    Build a rotation matrix from molecular frame to lab frame based on axisType.
    """
    if axisType == AmoebaMultipoleForce.ZOnly:
        z = normalize(rZ - rA)
        x_guess = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(z, x_guess)) > 0.99:
            x_guess = np.array([0.0, 1.0, 0.0])
        x = normalize(np.cross(x_guess, z))
        y = normalize(np.cross(z, x))
    elif axisType == AmoebaMultipoleForce.ZThenX:
        z = normalize(rZ - rA)
        x = normalize(rX - rA)
        y = normalize(np.cross(z, x))
        x = normalize(np.cross(y, z))  # re-orthogonalize

    elif axisType == AmoebaMultipoleForce.Bisector:
        v1 = normalize(rZ - rA)
        v2 = normalize(rX - rA)
        z = normalize(v1 + v2)
        x = normalize(v1 - v2)
        y = normalize(np.cross(z, x))
        x = normalize(np.cross(y, z))

    elif axisType == AmoebaMultipoleForce.ZBisect:
        v1 = normalize(rZ - rA)
        v2 = normalize(rX - rA)
        v3 = normalize(rY - rA)
        z = normalize(v1)
        x = normalize(v2 + v3)
        y = normalize(np.cross(z, x))
        x = normalize(np.cross(y, z))

    elif axisType == AmoebaMultipoleForce.ThreeFold:
        v1 = normalize(rZ - rA)
        v2 = normalize(rX - rA)
        v3 = normalize(rY - rA)
        z = normalize(v1 + v2 + v3)
        # original generated
        #x = normalize(v1 - v3)
        #y = normalize(np.cross(z, x))
        #x = normalize(np.cross(y, z))
        # new - from openmm
        x = normalize(v2 - z * z.dot(v2))
        y = normalize(np.cross(z, x))
    
    elif axisType == AmoebaMultipoleForce.NoAxisType:
        x, y, z = list(np.eye(3))
    
    else:
        raise ValueError(f"Unsupported axisType: {axisType}")

    return np.column_stack((x, y, z))  # 3x3 rotation matrix

def getLabFrameMultipoles(dipole_mol, quad_mol, rA, axisType, rZ=None, rX=None, rY=None):
    """
    Get lab-frame dipole and quadrupole for atom i from AmoebaMultipoleForce.
    
    Args:
        i: Atom index
        multipole_force: AmoebaMultipoleForce object
        positions: list of Vec3 atomic positions in nm
        
    Returns:
        dipole_lab (np.array shape [3]), quadrupole_lab (np.array shape [3, 3])
    """
    
    dipole_mol = np.array(dipole_mol)
    #quad_mol = unpack_quadrupole(quad_mol)


    R = buildRotationMatrix(axisType, rA, rZ, rX, rY)
    # original
    dipole_lab = R @ dipole_mol
    quad_lab = R @ quad_mol @ R.T
    ## testing
    #dipole_lab = R.T @ dipole_mol
    #quad_lab = R.T @ quad_mol @ R

    return dipole_lab, quad_lab

def cross_matrix(A,B):
    return np.sum(np.cross(A,B,axisa=0,axisb=0),axis=0)


# need to check this vs c++
def computeFrameVectorsOld(posA, posZ, posX=None, posW=None, axisType=AmoebaMultipoleForce.ZThenX):
    U = normalize(posZ - posA)        # Z-axis
    normU = np.linalg.norm(posZ - posA)

    if axisType in (AmoebaMultipoleForce.ZThenX, AmoebaMultipoleForce.Bisector, AmoebaMultipoleForce.ZOnly):
        V = normalize(posX - posA) if posX is not None else np.zeros(3)
    elif axisType in (AmoebaMultipoleForce.ZBisect, AmoebaMultipoleForce.ThreeFold):
        V = normalize(posX - posA)
        W = normalize(posW - posA)
    else:
        return None

    normV = np.linalg.norm(posX - posA) if posX is not None else 1.0

    if axisType in (AmoebaMultipoleForce.ZThenX, AmoebaMultipoleForce.Bisector):
        W = normalize(np.cross(U, V))  # complete frame
    elif axisType in (AmoebaMultipoleForce.ZBisect, AmoebaMultipoleForce.ThreeFold):
        pass  # W already defined
    elif axisType == AmoebaMultipoleForce.ZOnly:
        V = np.zeros(3)
        W = np.zeros(3)
    else:
        return None

    UV = normalize(np.cross(V, U))
    UW = normalize(np.cross(W, U))
    VW = normalize(np.cross(W, V))

    #angles = {
    #    'UV': (np.dot(U, V), np.linalg.norm(np.cross(U, V))),
    #    'UW': (np.dot(U, W), np.linalg.norm(np.cross(U, W))),
    #    'VW': (np.dot(V, W), np.linalg.norm(np.cross(V, W))),
    #}
    angles = {
        'UV': (np.dot(U, V), np.sqrt(1.0-np.dot(U, V)*np.dot(U, V))),
        'UW': (np.dot(U, W), np.sqrt(1.0-np.dot(U, W)*np.dot(U, W))),
        'VW': (np.dot(V, W), np.sqrt(1.0-np.dot(V,W)*np.dot(V,W))),
    }

    return U, V, W, UV, UW, VW, angles, normU, normV

def computeFrameVectors(posA, posZ, posX=None, posW=None, axisType=None):
    # Vector U (Z axis)
    vectorU = posZ - posA
    normU = np.linalg.norm(vectorU)
    U = normalize(vectorU)

    # Vector V (X axis)
    vectorV = posX - posA if posX is not None else np.zeros(3)
    normV = np.linalg.norm(vectorV)
    V = normalize(vectorV)

    # Vector W
    if axisType in (AmoebaMultipoleForce.ZBisect, AmoebaMultipoleForce.ThreeFold) and posW is not None:
        vectorW = posW - posA
    else:
        vectorW = np.cross(vectorU, vectorV)
    normW = np.linalg.norm(vectorW)
    W = normalize(vectorW)

    # Cross products for reference vectors
    vectorUV = np.cross(vectorV, vectorU)
    normUV = np.linalg.norm(vectorUV)
    UV = normalize(vectorUV)

    vectorUW = np.cross(vectorW, vectorU)
    normUW = np.linalg.norm(vectorUW)
    UW = normalize(vectorUW)

    vectorVW = np.cross(vectorW, vectorV)
    normVW = np.linalg.norm(vectorVW)
    VW = normalize(vectorVW)


    angles = {
        'UV': compute_cos_sin(U, V),
        'UW': compute_cos_sin(U, W),
        'VW': compute_cos_sin(V, W),
    }

    return U, V, W, UV, UW, VW, angles, normU, normV

# Cosine and sine of angles
def compute_cos_sin(a, b):
    cos = np.dot(a, b)
    sin = np.sqrt(max(0.0, 1.0 - cos * cos))  # Clamp for safety
    return (cos, sin)

def computeTorqueForcesOld(torque, posA, posZ, posX=None, posW=None, axisType=AmoebaMultipoleForce.ZOnly):
    if axisType == AmoebaMultipoleForce.NoAxisType:
        return {'forceZ': np.zeros(3), 'forceX': np.zeros(3), 'forceW': np.zeros(3), 'forceA': np.zeros(3)}

    result = computeFrameVectors(posA, posZ, posX, posW, axisType)
    if result is None:
        raise ValueError("Invalid axis configuration.")
    
    U, V, W, UV, UW, VW, angles, normU, normV = result

    # Project torque into local frame
    dphi = np.array([
        -np.dot(U, torque),
        -np.dot(V, torque),
        -np.dot(W, torque)
    ])

    sinUV = angles['UV'][1] if angles['UV'][1] > 1e-6 else 1.0  # prevent divide by zero
    half = 0.5

    # Initialize all forces
    forceZ = forceX = forceW = np.zeros(3)

    if axisType == AmoebaMultipoleForce.ZThenX:
        factor1 = dphi[1] / (normU * sinUV)
        factor2 = dphi[2] / normU
        factor3 = -dphi[0] / (normV * sinUV)
        forceZ = factor1 * UV + factor2 * UW
        forceX = factor3 * UV

    elif axisType == AmoebaMultipoleForce.Bisector:
        factor1 = dphi[1] / (normU * sinUV)
        factor2 = half * dphi[2] / normU
        factor3 = -dphi[0] / (normV * sinUV)
        factor4 = half * dphi[2] / normV
        forceZ = (factor1 * UV + factor2 * UW)
        forceX = (factor3 * UV + factor4 * VW)

    elif axisType == AmoebaMultipoleForce.ZOnly:
        forceZ = dphi[2] * UW
        forceX = np.zeros(3)

    elif axisType == AmoebaMultipoleForce.ZBisect:
        factor1 = half * dphi[1] / (normU * sinUV)
        factor2 = dphi[2] / normU
        factor3 = half * dphi[1] / (normV * sinUV)
        factor4 = dphi[2] / normV
        forceZ = factor1 * UV + factor2 * UW
        forceX = factor3 * UV + factor4 * VW

    elif axisType == AmoebaMultipoleForce.ThreeFold:
        factor1 = dphi[1] / (3.0 * normU * sinUV)
        factor2 = dphi[2] / (3.0 * normU)
        factor3 = dphi[1] / (3.0 * normV * sinUV)
        factor4 = dphi[2] / (3.0 * normV)
        forceZ = factor1 * UV + factor2 * UW
        forceX = factor3 * UV + factor4 * VW
        factor5 = dphi[1] / (3.0 * np.linalg.norm(posW - posA) * angles['VW'][1])
        factor6 = dphi[2] / (3.0 * np.linalg.norm(posW - posA))
        forceW = factor5 * UV + factor6 * VW

    # Net reaction on atom A
    forceA = -(forceZ + forceX + forceW)

    return {
        'forceZ': -forceZ,
        'forceX': -forceX,
        'forceW': -forceW,
        'forceA': -forceA
    }

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("Zero-length vector in axis definition.")
    return v / norm


def computeTorqueForcesNew(torque, posA, posZ, posX=None, posW=None, axisType=AmoebaMultipoleForce.ZOnly):
    if axisType == AmoebaMultipoleForce.NoAxisType:
        return {'forceZ': np.zeros(3), 'forceX': np.zeros(3), 'forceW': np.zeros(3), 'forceA': np.zeros(3)}

    result = computeFrameVectors(posA, posZ, posX, posW, axisType)
    if result is None:
        raise ValueError("Invalid axis configuration.")
    
    vectorU, vectorV, vectorW, vectorUV, vectorUW, vectorVW, angles, normU, normV = result

    vectorU = posZ - posA
    vectorV = posX - posA if posX is not None else np.zeros(3)
    vectorW = posW - posA if (axisType in [AmoebaMultipoleForce.ZBisect, AmoebaMultipoleForce.ThreeFold] and posW is not None) else np.cross(vectorU, vectorV)

    dphi = np.array([
        np.dot(vectorU, torque),
        np.dot(vectorV, torque),
        np.dot(vectorW, torque)
    ]) * -1.0

    forceZ = forceX = forceW = np.zeros(3)

    if axisType in (AmoebaMultipoleForce.ZThenX, AmoebaMultipoleForce.Bisector):
        sinUV = angles['UV'][1] if angles['UV'][1] > 1e-10 else 1.0
        half = 0.5

        factor1 = dphi[1] / (normU * sinUV)
        factor2 = dphi[2] / normU
        factor3 = -dphi[0] / (normV * sinUV)

        if axisType == AmoebaMultipoleForce.Bisector:
            factor2 *= half
            factor4 = half * dphi[2] / normV
        else:
            factor4 = 0.0

        forceZ = vectorUV * factor1 + vectorUW * factor2
        forceX = vectorUV * factor3 + vectorVW * factor4

    elif axisType == AmoebaMultipoleForce.ZOnly:
        forceZ = vectorUV * dphi[1] / (normU * angles['UV'][1]) + vectorUW * dphi[2] / normU
        
        
    elif axisType == AmoebaMultipoleForce.ZBisect:
        vectorR = vectorV + vectorW
        vectorS = np.cross(vectorU, vectorR)

        normR = np.linalg.norm(vectorR)
        normS = np.linalg.norm(vectorS)
        vectorR /= normR
        vectorS /= normS

        vectorUR = np.cross(vectorR, vectorU)
        vectorUS = np.cross(vectorS, vectorU)
        vectorVS = np.cross(vectorS, vectorV)
        vectorWS = np.cross(vectorS, vectorW)

        dotUR = np.dot(vectorU, vectorR)
        dotUS = np.dot(vectorU, vectorS)
        dotVS = np.dot(vectorV, vectorS)
        dotWS = np.dot(vectorW, vectorS)

        sinUR = np.sqrt(max(0.0, 1.0 - dotUR * dotUR))
        sinUS = np.sqrt(max(0.0, 1.0 - dotUS * dotUS))
        sinVS = np.sqrt(max(0.0, 1.0 - dotVS * dotVS))
        sinWS = np.sqrt(max(0.0, 1.0 - dotWS * dotWS))

        t1 = vectorV - vectorS * dotVS
        t2 = vectorW - vectorS * dotWS

        t1 /= np.linalg.norm(t1)
        t2 /= np.linalg.norm(t2)

        ut1cos = np.dot(vectorU, t1)
        ut2cos = np.dot(vectorU, t2)

        ut1sin = np.sqrt(max(0.0, 1.0 - ut1cos * ut1cos))
        ut2sin = np.sqrt(max(0.0, 1.0 - ut2cos * ut2cos))

        dphiR = np.dot(vectorR, torque) * -1.0
        dphiS = np.dot(vectorS, torque) * -1.0

        factor1 = dphiR / (normU * sinUR)
        factor2 = dphiS / normU
        denom = ut1sin + ut2sin
        factor3 = dphi[0] / (normV * denom)
        factor4 = dphi[0] / (np.linalg.norm(posW - posA) * denom)

        forceZ = vectorUR * factor1 + vectorUS * factor2
        forceX = (vectorS * sinVS - t1 * dotVS) * factor3
        forceW = (vectorS * sinWS - t2 * dotWS) * factor4

    elif axisType == AmoebaMultipoleForce.ThreeFold:
        sinUV = angles['UV'][1]
        sinUW = angles['UW'][1]
        sinVW = angles['VW'][1]

        normW = np.linalg.norm(vectorW)

        du = (vectorUW * dphi[2] / (normU * sinUW)
            + vectorUV * dphi[1] / (normU * sinUV)
            - vectorUW * dphi[0] / (normU * sinUW)
            - vectorUV * dphi[0] / (normU * sinUV)) / 3.0

        dv = (vectorVW * dphi[2] / (normV * sinVW)
            - vectorUV * dphi[0] / (normV * sinUV)
            - vectorVW * dphi[1] / (normV * sinVW)
            + vectorUV * dphi[1] / (normV * sinUV)) / 3.0

        dw = (-vectorUW * dphi[0] / (normW * sinUW)
            - vectorVW * dphi[1] / (normW * sinVW)
            + vectorUW * dphi[2] / (normW * sinUW)
            + vectorVW * dphi[2] / (normW * sinVW)) / 3.0

        forceZ = du
        forceX = dv
        forceW = dw

    # Net reaction on atom A (central particle)
    forceA = -(forceZ + forceX + forceW)

    return {
        'forceZ': -forceZ,
        'forceX': -forceX,
        'forceW': -forceW,
        'forceA': -forceA
    }
    
computeTorqueForces = computeTorqueForcesOld