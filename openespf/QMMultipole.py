import numpy as np
from scipy.linalg import sqrtm, inv, solve
from scipy.optimize import fsolve
from copy import deepcopy, copy
from pyscf import gto, scf, ao2mo, dft, lib
from pyscf.data import radii

from timeit import default_timer as timer

'''
This module contains stand alone implementations of the multipole operators.
This includes Mulliken and ESPF operators and their derivatives.

The QMMultipole class wraps a lot of these methods for ease of use.
'''

def getNumMultipolePerAtom(l_max):
    if l_max == 0:
        return 1
    elif l_max == 1:
        return 4 
    else:
        print("l_max = ", l_max, "not implemented!")
        return None

def getMullikenMultipoleOperators(mol,l_max=0,S=None,r_int=None):
    '''
    gets Mulliken multipole operators for the mol object.
    '''
    # get AO and mol info needed for constructing Mulliken operators
    ao_info = mol.aoslice_by_atom()
    N_atm = mol.natm
    N_Q = N_atm * getNumMultipolePerAtom(l_max)
    N_AO = mol.nao
    
    # get overlap matrix if not already available
    if S is None:
        S = mol.intor('int1e_ovlp')
    
    # get empty Q operators
    Q = np.zeros((N_Q,N_AO,N_AO))
    
    # charge part
    # Q_A = -(1/2)(P_A S + S P_A)
    for A in range(0,N_atm):
        ao_end = ao_info[A][3]
        ao_start = ao_info[A][2]
        Q[A,:,ao_start:ao_end] = -0.5 * S[:,ao_start:ao_end]
        Q[A,ao_start:ao_end,:] = Q[A,ao_start:ao_end,:] -0.5 * S[ao_start:ao_end,:]
    
    # dipole part
    # mu_Ax = -(1/2)(P_A r + r P_A) - R_Ax Q_A
    if l_max > 0:
        ao_info = mol.aoslice_by_atom()
        if r_int is None:
            r_int = mol.intor('int1e_r')
        R = mol.atom_coords()
        for x in range(0,3):
            for A in range(0,N_atm):
                ao_end = ao_info[A][3]
                ao_start = ao_info[A][2]
                Q[A+(x+1)*N_atm,:,ao_start:ao_end] = -0.5 * r_int[x,:,ao_start:ao_end] + (0.5*R[A,x])*S[:,ao_start:ao_end]
                Q[A+(x+1)*N_atm,ao_start:ao_end,:] = Q[A+(x+1)*N_atm,ao_start:ao_end,:] - 0.5 * r_int[x,ao_start:ao_end,:] + (0.5*R[A,x])*S[ao_start:ao_end,:]
    return Q

def getMullikenMultipoleOperatorsAtmDeriv(mol,J,l_max=0,S=None,r_int=None,ip=None,irp=None):
    '''
    gets Mulliken multipole operator derivatives wrt atom J coordinates.
    '''
    # get AO and mol info needed for constructing Mulliken operators
    ao_info = mol.aoslice_by_atom()
    N_atm = mol.natm
    N_Q = N_atm * getNumMultipolePerAtom(l_max)
    N_AO = mol.nao
    N_bas = mol.nbas
    
    # empty grad operator
    gradJ_Q  = np.zeros((3,N_Q,N_AO,N_AO))
    
    # get the <nabla_x chi_n | chi_m> integrals basis functions n in J
    # <nabla_Jx chi_n | chi_m> = -<nabla_x chi_n | chi_m> for n in J and 0 otherwise
    if ip is None:
        bas_start = ao_info[J][0]
        bas_end = ao_info[J][1]
        ip = mol.intor('int1e_ipovlp',shls_slice=(bas_start,bas_end,0,N_bas))
    
    # construct the derivative operator
    ao_start_J = ao_start = ao_info[J][2]
    ao_end_J = ao_info[J][3]
    for x in range(0,3):
        for A in range(0,N_atm):
            ao_start = ao_info[A][2]
            ao_end = ao_info[A][3]
            if not A == J:
                gradJ_Q[x,A,ao_start_J:ao_end_J,ao_start:ao_end] = 0.5 * ip[x,:,ao_start:ao_end]
                gradJ_Q[x,A,ao_start:ao_end,ao_start_J:ao_end_J] = 0.5 * ip[x,:,ao_start:ao_end].T
            else:
                gradJ_Q[x,J,ao_start_J:ao_end_J,:] = 0.5 * ip[x,:,:]
                gradJ_Q[x,J,:,ao_start_J:ao_end_J] = gradJ_Q[x,J,:,ao_start_J:ao_end_J] + 0.5 * ip[x,:,:].T
    
    if l_max>0:
        # get the < chi_n | r_y nabla_x | chi_m> integrals basis functions n in J
        # < chi_n |r_y|nabla_Jx chi_m> = -< chi_n |r_y| nabla_x chi_m> for n in J and 0 otherwise
        # these integrals are stored 0: r_0 ip_0, 1: r_0 ip_1, 2: r_0 ip_2, 3: r_1, ip_0, ... etc.
        if irp is None:
            bas_start = ao_info[J][0]
            bas_end = ao_info[J][1]
            irp = mol.intor('int1e_irp',shls_slice=(0,N_bas,bas_start,bas_end))
        if S is None:
            bas_start = ao_info[J][0]
            bas_end = ao_info[J][1]
            S = mol.intor('int1e_ovlp',shls_slice=(0,N_bas,bas_start,bas_end))
        
        R = mol.atom_coords() 
        
        for x in range(0,3):
            Ay = 0
            for y in range(0,3):
                ao_info = mol.aoslice_by_atom()
                for A in range(0,N_atm): 
                    Ay = (A + ((1+y)*N_atm))
                    yx = (3*y) + x
                    ao_start = ao_info[A][2]
                    ao_end = ao_info[A][3]
                    if (not A==J):
                        Ay = (A + ((1+y)*N_atm))
                        yx = (3*y) + x
                        gradJ_Q[x,Ay,ao_start:ao_end,ao_start_J:ao_end_J] = 0.5 * irp[yx,ao_start:ao_end,:] -  0.5 * R[A,y] * ip[x,:,ao_start:ao_end].T
                        gradJ_Q[x,Ay,ao_start_J:ao_end_J,ao_start:ao_end] = 0.5 * irp[yx,ao_start:ao_end,:].T -  0.5 * R[A,y] * ip[x,:,ao_start:ao_end]
                    else:
                        Ay = (A + ((1+y)*N_atm))
                        yx = (3*y) + x
                        gradJ_Q[x,Ay,:,ao_start_J:ao_end_J] = 0.5 * irp[yx,:,:] - 0.5 * R[A,y] *  ip[x,:,:].T
                        gradJ_Q[x,Ay,ao_start_J:ao_end_J,:] = gradJ_Q[x,Ay,ao_start_J:ao_end_J,:] + 0.5 * irp[yx,:,:].T  - 0.5 * R[A,y] *  ip[x,:,:]
                        gradJ_Q[x,Ay,ao_start_J:ao_end_J,ao_start_J:ao_end_J] += 0.5*irp[yx,ao_start_J:ao_end_J,:] + 0.5*irp[yx,ao_start_J:ao_end_J,:].T
                        if x==y:
                            gradJ_Q[x,Ay,:,ao_start_J:ao_end_J] = gradJ_Q[x,Ay,:,ao_start_J:ao_end_J] + 0.5 * S
                            gradJ_Q[x,Ay,ao_start_J:ao_end_J,:] = gradJ_Q[x,Ay,ao_start_J:ao_end_J,:] +  0.5 * S.T
    
    return gradJ_Q

def getMullikenMultipoleOperatorsAtmDerivNum(mol,J,l_max=0,dx=0.001):
    '''
    gets Mulliken multipole operator derivatives wrt atom J coordinates by central finite difference. 
    This is used for testing only!
    '''
    
    # get AO and mol info needed for constructing Mulliken operators
    ao_info = mol.aoslice_by_atom()
    N_atm = mol.natm
    N_Q = N_atm * getNumMultipolePerAtom(l_max)
    N_AO = mol.nao
    N_bas = mol.nbas
    
    # empty grad operator
    gradJ_Q  = np.zeros((3,N_Q,N_AO,N_AO))
    
    R = mol.atom_coords() 
    
    for x in range(0,3):
        # forward difference
        R_new = R+0.
        R_new[J,x] = R_new[J,x] + dx
        mol_new = mol.copy()
        mol_new.set_geom_(R_new,unit="Bohr")
        Q_f = getMullikenMultipoleOperators(mol_new,l_max=l_max)
        # backward difference
        R_new = R+0.
        R_new[J,x] = R_new[J,x] - dx 
        mol_new = mol.copy()
        mol_new.set_geom_(R_new,unit="Bohr")
        Q_b = getMullikenMultipoleOperators(mol_new,l_max=l_max)
        # central finite difference derivatives
        gradJ_Q[x,:,:,:] = (Q_f - Q_b) * (0.5/dx)
        
    
    return gradJ_Q

def calcNumGradIntor(mol,J,int_label,dx=0.001):
    '''
    gets Mulliken multipole operator derivatives wrt atom J coordinates by central finite difference. 
    This is used for testing only!
    '''
    
    # get AO and mol info needed for constructing Mulliken operators
    ao_info = mol.aoslice_by_atom()
    N_atm = mol.natm
    N_AO = mol.nao
    N_bas = mol.nbas
    
    molint = mol.intor(int_label)
    size = molint.shape
    # empty grad operator
    gradJ_Q  = []
    
    R = mol.atom_coords() 
    
    for x in range(0,3):
        # forward difference
        R_new = R+0.
        R_new[J,x] = R_new[J,x] + dx
        mol_new = mol.copy()
        mol_new.set_geom_(R_new,unit="Bohr")
        Q_f = mol_new.intor(int_label)
        # backward difference
        R_new = R+0.
        R_new[J,x] = R_new[J,x] - dx 
        mol_new = mol.copy()
        mol_new.set_geom_(R_new,unit="Bohr")
        Q_b = mol_new.intor(int_label)
        # central finite difference derivatives
        gradJ_Q.append( (Q_f - Q_b) * (0.5/dx))
    
        
    
    return np.array(gradJ_Q)

def getESPFGrid(mol,n_ang,n_rad,rad_method="vdw",ang_method="lebedev",rad_scal=1.0):
    '''
    gets a grid of points for ESPF operator fitting.
    '''
    
    # get info about the molecule for which the grid is being getd
    Z = mol.atom_charges()
    R = mol.atom_coords()
    N_atm = len(Z)
    
    # form the angular grid
    if ang_method == "lebedev":
        mol_0 = gto.M(atom=[["H",[0,0,0]]],spin=1)
        grids = dft.gen_grid.Grids(mol_0)
        grids.atom_grid = (1,n_ang)
        grids.kernel()
        # get a single shell of lebedev grid points 
        grid_ang = grids.coords 
        grid_ang = np.array([grid_ang[n,:] for n in range(0,grid_ang.shape[0]) if np.linalg.norm(grid_ang[n,:])>1.0e-2])
    else:
        raise Exception("Angular grid method ",ang_method," not recognised.")
    
    # form the radial grid
    if rad_method == "vdw":
        # vdw radii of the atoms
        r_vdw = radii.VDW[Z]
        r_grid = rad_scal * np.arange(1,n_rad+1)
        grid_rad = r_vdw.reshape((N_atm,1)) * r_grid.reshape((1,n_rad))
    else:
        raise Exception("Radial grid method ",rad_method," not recognised.")
        
    # combine the radial and angular grids
    N_grid_atm = n_ang * n_rad 
    N_grid = N_grid_atm * N_atm
    grid_coords = np.zeros((N_grid,3))
    grid_atms = np.ones((N_grid,))*(N_atm+1)
    for A in range(0,N_atm):
        grid_atm = np.zeros((N_grid_atm,3))
        for k in range(0,n_rad):
            start = k*n_ang
            end = start+n_ang
            grid_atm[start:end,:] = (grid_ang * grid_rad[A,k]) + R[A,:]
        start = A*N_grid_atm
        end = start + N_grid_atm
        grid_coords[start:end,:] = grid_atm+0.
        grid_atms[start:end] = A
        
    return grid_coords, grid_atms


def getESPMat(mol,grid_coords,block_size=16):
    N_grid = grid_coords.shape[0]
    N_AO = mol.nao
    N_block = int(np.ceil(N_grid/block_size))
    esp = np.zeros((N_grid,N_AO,N_AO))
    for i in range(0,N_block):
        start = i*block_size
        end = min(start+block_size,N_grid)
        esp[start:end,:,:] = -mol.intor('int1e_grids',grids=grid_coords[start:end,:])
    return esp

def getIPESPMat(mol,grid_coords,block_size=16):
    N_grid = grid_coords.shape[0]
    N_AO = mol.nao
    N_block = int(np.ceil(N_grid/block_size))
    ipesp = np.zeros((3,N_grid,N_AO,N_AO))
    for i in range(0,N_block):
        start = i*block_size
        end = min(start+block_size,N_grid)
        ipesp[:,start:end,:,:] = -mol.intor('cint1e_grids_ip_sph',grids=grid_coords[start:end,:])
    return ipesp

def getESPFMultipoleOperators(mol,grid_coords,weights,l_max=0,block_size=16,add_correction=True,return_fit_vars=False,
                              S=None,r_int=None,reg_type="none",reg_param=0.0):
    '''
    gets ESPF charge operators.
    '''
    # get coordinates of 
    R = mol.atom_coords(unit="Bohr")
    
    # general info
    N_grid = grid_coords.shape[0]
    N = R.shape[0]
    N_Q_atm = getNumMultipolePerAtom(l_max)
    N_Q = N * N_Q_atm
    N_AO = mol.nao
    
    # get the ESP matrix D_ka
    D = getESPMatrix(grid_coords,R,l_max)
    
    # weights of grid points
    w = weights
    
    # get the ESP potential operators at grid points
    esp = getESPMat(mol,grid_coords,block_size) 
    
    #reg = 1.0e-7
    
    # matrices for fitting
    w_D = w.reshape((N_grid,1)) * D
    A_fit = (D.T).dot( w_D )
    A_fit0_diag = np.diag(A_fit)+0.
    if reg_type in ["mulliken","magnitude"]:
        A_fit[np.diag_indices_from(A_fit)] *= (1.0+reg_param)
    A_fit_inv = inv(A_fit)
    b_fit = np.einsum('ka,knm->anm',w_D,esp,optimize=True)
    #D_esp = np.einsum('ka,knm',D,esp)
    if reg_type == "mulliken":
        Q_mulliken = getMullikenMultipoleOperators(mol,l_max=l_max,S=S,r_int=r_int)
        reg_b_fit = reg_param * A_fit0_diag[:,None,None] * Q_mulliken
        b_fit += reg_b_fit
    
    # get the uncorrect Q operators
    Q = np.einsum('ab,bnm->anm',A_fit_inv,b_fit)
    
    # add correction 
    if add_correction:
        if S is None:
            Q_tot = -mol.intor('int1e_ovlp')
        else:
            Q_tot = -S
        Q[0:N,:,:] = Q[0:N,:,:] + (1./N)*(Q_tot - np.einsum('Anm->nm',Q[0:N,:,:]))
        if l_max>0:
            if r_int is None:
                mu_tot = -mol.intor('int1e_r')
            else:
                mu_tot = -r_int
            for alpha in range(0,3):
                start = (alpha+1)*N
                end = start + N
                Q[start:end,:,:] = Q[start:end,:,:] + (1./N)*(mu_tot[alpha,:,:] - np.einsum('Anm->nm',Q[start:end,:,:])-np.einsum('A,Anm->nm',R[:,alpha],Q[0:N,:,:]))
    
    
    if return_fit_vars:
        fit_vars = {"A_fit":A_fit,"A_fit_inv":A_fit_inv,"b_fit":b_fit,"D":D,"w":w,"esp":esp,"w_D":w_D,"A_fit0_diag":A_fit0_diag}
        if reg_type == "mulliken":
            fit_vars["Q_mulliken"] = Q_mulliken
        return Q, fit_vars
    else:
        return Q
    
def getESPMatrix(grid_coords,R,l_max):
    '''
    gets the ESP matrix of ESP at grid coords getd by point multipoles at R.
    '''
    N = R.shape[0]
    N_Q_atm = getNumMultipolePerAtom(l_max)
    N_Q = N * N_Q_atm
    N_grid = grid_coords.shape[0]
    D = np.zeros((N_grid,N_Q))
    
    # N_grid x N x 3
    x = grid_coords.reshape((N_grid,1,3)) - R.reshape((1,N,3))
    r = np.linalg.norm(x,axis=2)
    
    # charge part
    r_inv = 1./r
    D[:,0:N] = r_inv
    
    # dipole part
    if l_max>0 : 
        r3_inv = r_inv * r_inv * r_inv
        for alpha in range(0,3):
            start = (alpha+1)*N
            end = start + N
            D[:,start:end] = x[:,:,alpha] * r3_inv
        
    return D

def getGridWeights(mol,grid_coords,weight_mode="hard",hard_cut_vdw_scal=1.0,sigma=0.2,smooth_func="sin"):
    '''
    gets weights of grid points dependent on scheme used
    '''
    R = mol.atom_coords()
    N = R.shape[0]
    N_grid = grid_coords.shape[0]
    
    if weight_mode =="hard":
        # N_grid x N x 3
        x = grid_coords.reshape((N_grid,1,3)) - R.reshape((1,N,3))
        r = np.linalg.norm(x,axis=2)
        w = np.ones((N_grid,))
        Z = mol.atom_charges()
        r_vdw = radii.VDW[Z]
        r_cut = hard_cut_vdw_scal * r_vdw
        for A in range(0,N):
            cutoff = (r[:,A]<r_cut[A])
            w[cutoff] = 0.0
        
    elif weight_mode == "none":
        w = np.ones((N_grid,))
    
    elif weight_mode == "smooth":
        x = grid_coords.reshape((N_grid,1,3)) - R.reshape((1,N,3))
        r = np.linalg.norm(x,axis=2)
        w = np.ones((N_grid,))
        Z = mol.atom_charges()
        r_vdw = radii.VDW[Z]
        r_cut = hard_cut_vdw_scal * r_vdw
        y = (r-sigma-r_cut.reshape((1,N)))/sigma
        h = softStep(y,func=smooth_func)
        w = np.prod(h,axis=1)
    
    return w

def getESPFMultipoleOperatorsAtmDerivNum(mol,J,l_max,multipole_fnc,grid_fnc,weight_fnc,dx=0.001):
    '''
    Calculates numerical derivatives of the ESPF charge operators
    '''
    
    # get AO and mol info needed for constructing Mulliken operators
    ao_info = mol.aoslice_by_atom()
    N_atm = mol.natm
    N_Q = N_atm * getNumMultipolePerAtom(l_max)
    N_AO = mol.nao
    N_bas = mol.nbas
    
    # empty grad operator
    gradJ_Q  = np.zeros((3,N_Q,N_AO,N_AO))
    
    R = mol.atom_coords() 
    
    for x in range(0,3):
        # forward difference
        R_new = R+0.
        R_new[J,x] = R_new[J,x] + dx
        mol_new = mol.copy()
        mol_new.set_geom_(R_new,unit="Bohr")
        grid_coords,grid_atms = grid_fnc(mol_new)
        weights = weight_fnc(mol_new,grid_coords)
        Q_f = multipole_fnc(mol_new,grid_coords,weights,l_max=l_max)
        # backward difference
        R_new = R+0.
        R_new[J,x] = R_new[J,x] - dx 
        mol_new = mol.copy()
        mol_new.set_geom_(R_new,unit="Bohr")
        grid_coords,grid_atms = grid_fnc(mol_new)
        weights = weight_fnc(mol_new,grid_coords)
        Q_b = multipole_fnc(mol_new,grid_coords,weights,l_max=l_max)
        # central finite difference derivatives
        gradJ_Q[x,:,:,:] = (Q_f - Q_b) * (0.5/dx)
    
    return gradJ_Q


def calculateGridWeightDerivs(mol,J,grid_coords,w,weight_mode="hard",hard_cut_vdw_scal=1.0,sigma=0.2,grid_atms=None,smooth_func="sin"):
    '''
    Calculate derivatives of the grid weights for atom J
    '''
    if weight_mode in ["hard","none"]:
        N_grid = grid_coords.shape[0]
        gradJ_w = np.zeros((3,N_grid))
    elif weight_mode == "smooth":
        R = mol.atom_coords()
        N = R.shape[0]
        N_grid = grid_coords.shape[0]
        gradJ_w = np.zeros((3,N_grid))
        x = grid_coords.reshape((N_grid,1,3)) - R.reshape((1,N,3))
        r = np.linalg.norm(x,axis=2)
        Z = mol.atom_charges()
        r_vdw = radii.VDW[Z]
        r_cut = hard_cut_vdw_scal * r_vdw
        y = (r-sigma-r_cut.reshape((1,N)))/sigma
        # indices of grid points belonging to J
        gridJ_inds = np.array([k for k in range(0,N_grid) if grid_atms[k]==J])
        N_gridJ = len(gridJ_inds)
        for alpha in range(0,3):
            gradJ_w[alpha,gridJ_inds] += (1./sigma)*w[gridJ_inds]*np.sum((x[gridJ_inds,:,alpha]/r[gridJ_inds,:])*derivLogSoftStep(y[gridJ_inds,:],func=smooth_func),axis=1)
            gradJ_w[alpha,:] -= (1./sigma)* w * (x[:,J,alpha]/r[:,J])*derivLogSoftStep(y[:,J],func=smooth_func)
        print("non-zero gradJ_w : ",np.sum(np.abs(gradJ_w)>1.0e-14),"of",(np.prod(np.array(gradJ_w.shape))))
    return gradJ_w

def calculateGridWeightDerivsOpt(mol,J,grid_coords,w,weight_mode="hard",hard_cut_vdw_scal=1.0,sigma=0.2,grid_atms=None,smooth_func="sin"):
    '''
    Calculate derivatives of the grid weights for atom J
    '''
    if weight_mode in ["hard","none"]:
        N_grid = grid_coords.shape[0]
        gradJ_w = np.zeros((3,N_grid))
    elif weight_mode == "smooth":
        R = mol.atom_coords()
        N = R.shape[0]
        N_grid = grid_coords.shape[0]
        gradJ_w = np.zeros((3,N_grid))
        x = grid_coords.reshape((N_grid,1,3)) - R.reshape((1,N,3))
        r = np.linalg.norm(x,axis=2)
        Z = mol.atom_charges()
        r_vdw = radii.VDW[Z]
        r_cut = hard_cut_vdw_scal * r_vdw
        y = (r-sigma-r_cut.reshape((1,N)))/sigma
        # indices of grid points belonging to J
        gridJ_inds = np.array([k for k in range(0,N_grid) if grid_atms[k]==J])
        N_gridJ = len(gridJ_inds)
        for alpha in range(0,3):
            gradJ_w[alpha,gridJ_inds] += (1./sigma)*w[gridJ_inds]*np.sum((x[gridJ_inds,:,alpha]/r[gridJ_inds,:])*derivLogSoftStep(y[gridJ_inds,:],func=smooth_func),axis=1)
            gradJ_w[alpha,:] -= (1./sigma)* w * (x[:,J,alpha]/r[:,J])*derivLogSoftStep(y[:,J],func=smooth_func)
        inds = np.where(np.linalg.norm(gradJ_w,axis=0)>1.0e-14)[0]
        gradJ_w = {"grid":gradJ_w[:,inds],"grid_inds":inds}
    return gradJ_w

def getESPFMultipoleOperatorAtmDeriv(mol,J,grid_coords,fit_vars,gradJ_w,grad_D,ipesp,grid_atms,l_max=0,
                                     block_size=16,add_correction=True,Q=None,approx_esp=False,
                                    reg_type="none",reg_param=0.0):
    
    print_info = False
    # get coordinates of 
    R = mol.atom_coords()
    
    # general info
    N_grid = grid_coords.shape[0]
    N = R.shape[0]
    N_Q_atm = getNumMultipolePerAtom(l_max)
    N_Q = N * N_Q_atm
    N_AO = mol.nao
    
    # get fitting parameters
    D = fit_vars["D"]
    w = fit_vars["w"]
    esp = fit_vars["esp"]
    b_fit = fit_vars["b_fit"]
    A_fit = fit_vars["A_fit"]
    A_fit_inv = fit_vars["A_fit_inv"]
    w_D = fit_vars["w_D"]
    A_fit0_diag = fit_vars["A_fit0_diag"]
    # empty array for grad
    gradJ_Q = np.zeros((3,N_Q,N_AO,N_AO))
    
    start = timer()
    # first get grad_J D
    #gradJ_D = getGradJDESP(J,grad_D,grid_atms,l_max)
    gradJ_D_opt = getGradJDESPOpt(J,grad_D,grid_atms,l_max)
    
    if print_info: print("gradJ_D time:",timer()-start,"s")
    # gradJ A_fit
    start = timer()
    #gradJ_A_fit = getGradAfitESP(gradJ_D,gradJ_w,D,w,l_max)
    gradJ_A_fit = getGradAfitESPOpt(gradJ_D_opt,gradJ_w,D,w,w_D)
    if reg_type in ["mulliken","magnitude"]:
        gradJ_A_fit0_diag = np.einsum('xaa->xa',gradJ_A_fit)
        np.einsum('xaa->xa',gradJ_A_fit)[:,:] += reg_param * gradJ_A_fit0_diag
    if print_info: print("gradJ_A_fit time:",timer()-start,"s")
    # the derivative consists of two parts:
    # grad_J (A^-1 b) = (grad_J A^-1 ) b + A^-1 (grad_J b)
    # first the (grad_J A^-1) b part
    start = timer()
    gradJ_A_fit_inv = np.einsum('xab,bc->xac',gradJ_A_fit,-A_fit_inv,optimize=True)
    gradJ_A_fit_inv = np.einsum('ab,xbc->xac',A_fit_inv,gradJ_A_fit_inv,optimize=True)
    gradJ_Q += np.einsum('xab,bnm->xanm',gradJ_A_fit_inv,b_fit,optimize=True)
    if print_info: print("gradJ_Q gradJ_A_fit time:",timer()-start,"s")
    # get gradJ_esp
    start = timer()
    #gradJ_esp = getGradESPGrid(J,mol,ipesp,grid_atms)
    gradJ_esp_opt = getGradESPGridOpt(J,mol,ipesp,grid_atms)
    if print_info: print("gradJ_esp time:",timer()-start,"s")
    # get gradJ b_fit
    start = timer()
    #gradJ_b_fit = getGradbfitESP(esp,w,D,gradJ_D,gradJ_w,gradJ_esp)
    #gradJ_D_opt = getGradJDESPOpt(J,grad_D,grid_atms,l_max)
    
    # this line is introduced just to test the quality of using an approximate multipole expression 
    # - this could enable a faster implementation of the QM gradients
    if approx_esp:
        esp_approx = np.einsum('ka,anm->knm',D,Q,optimize=True)
        gradJ_b_fit = getGradbfitESPOpt(esp_approx,w,D,gradJ_D_opt,gradJ_w,gradJ_esp_opt,w_D)
    else:
        gradJ_b_fit = getGradbfitESPOpt(esp,w,D,gradJ_D_opt,gradJ_w,gradJ_esp_opt,w_D)
    
    if reg_type == "mulliken":
        Q_mulliken = fit_vars["Q_mulliken"]
        gradJ_b_fit += reg_param * np.einsum('xa,anm->xanm',gradJ_A_fit0_diag,Q_mulliken,optimize=True)
        gradJ_Q_mulliken = getMullikenMultipoleOperatorsAtmDeriv(mol,J,l_max=l_max,S=None,r_int=None,ip=None,irp=None)
        gradJ_b_fit += reg_param * np.einsum('a,xanm->xanm',A_fit0_diag,gradJ_Q_mulliken,optimize=True)
    
    # second the A^-1 (grad_J b) part
    
    gradJ_Q += np.einsum('ab,xbnm->xanm',A_fit_inv,gradJ_b_fit,optimize=True)
    if print_info: print("gradJ_b_fit time:",timer()-start,"s")
    start = timer() 
    if add_correction:
        gradJ_Q_tot = getGradQtot(mol,J)
        gradJ_Q[:,0:N,:,:] += (1./N) * (gradJ_Q_tot - np.sum(gradJ_Q[:,0:N,:,:],axis=1)).reshape((3,1,N_AO,N_AO))
        
        if l_max>0:
            gradJ_mu_tot = getGradmutot(mol,J)    
            for alpha in range(0,3):
                start = (alpha+1)*N
                end = start+N
                #gradJ_Q[:,start:end,:,:] += (1./N) * (gradJ_mu_tot[:,alpha,:,:] - np.sum(gradJ_Q[:,start:end,:,:],axis=1)).reshape((3,1,N_AO,N_AO))
                #gradJ_Q[:,start:end,:,:] += (-1./N) * (np.einsum('xanm,a->xnm',gradJ_Q[:,0:N,:,:],R[:,alpha])).reshape((3,1,N_AO,N_AO))
                #gradJ_Q[alpha,start:end,:,:] += (-1./N) * Q[J,:,:].reshape((1,N_AO,N_AO))
                gradJ_Q[:,start:end,:,:] += (1./N) * (gradJ_mu_tot[:,alpha,:,:] - np.sum(gradJ_Q[:,start:end,:,:],axis=1))[:,None,:,:]
                gradJ_Q[:,start:end,:,:] += (-1./N) * (np.einsum('xanm,a->xnm',gradJ_Q[:,0:N,:,:],R[:,alpha]))[:,None,:,:]
                gradJ_Q[alpha,start:end,:,:] += (-1./N) * Q[J,:,:]
    
    if print_info: print("correction grad time:",timer()-start,"s")
    return gradJ_Q

def getGradbfitESPOpt(esp,w,D,gradJ_D,gradJ_w,gradJ_esp,w_D):
    '''
    An optimised version of below
    '''

    # Compute gradJ_wD more efficiently
    #gradJ_wD = np.einsum('xk,ka->xka', gradJ_w, D, optimize=True)
    dD_row = gradJ_D["rows"]
    row_inds = gradJ_D["rowinds"]
    dD_col = gradJ_D["cols"]
    col_inds = gradJ_D["colinds"]
    
    N_grid = esp.shape[0]
    N_AO = esp.shape[1]
    N_Q = D.shape[1]
    # Compute gradJ_b_fit using optimized einsum
    desp_ao = gradJ_esp["ao"]
    aos = gradJ_esp["ao_inds"]
    gradJ_b_fit = np.zeros((3,N_Q,N_AO,N_AO))
    gradJ_b_fit_ao = np.einsum('ka,xknm->xanm', w_D, desp_ao, optimize="optimal")
    gradJ_b_fit[:,:,aos,:] += gradJ_b_fit_ao
    gradJ_b_fit[:,:,:,aos] += gradJ_b_fit_ao.transpose(0,1,3,2)
    desp_grid = gradJ_esp["grid"]
    grids = gradJ_esp["grid_inds"]
    gradJ_b_fit_grid = np.einsum('ka,xknm->xanm', w_D[grids,:], desp_grid,optimize="optimal")
    gradJ_b_fit += gradJ_b_fit_grid + gradJ_b_fit_grid.transpose(0,1,3,2)
    
    dD_row = dD_row * w[None,row_inds,None]
    dD_col = dD_col * w[None,:,None]
    gradJ_b_fit += np.einsum('xka,knm->xanm',dD_row,esp[row_inds,:,:],optimize="optimal")
    gradJ_b_fit[:,col_inds] += np.einsum('xka,knm->xanm',dD_col,esp[:,:],optimize="optimal")
    if isinstance(gradJ_w,dict):
        dw = gradJ_w["grid"]
        inds = gradJ_w["grid_inds"]
        dw_D = np.einsum('xk,ka->xka',dw,D[inds,:],optimize="optimal")
        gradJ_b_fit += np.einsum('knm,xka->xanm',esp[inds,:,:],dw_D,optimize="optimal")
    else:
        gradJ_w_D = np.einsum('xk,ka->xka',gradJ_w,D,optimize="optimal")
        gradJ_b_fit += np.einsum('knm,xka->xanm',esp,gradJ_w_D,optimize="optimal")
    #gradJ_b_fit += np.einsum()
    
    # Compute gradJ_b_fit using optimized einsum
    #gradJ_b_fit = np.einsum('ka,xknm->xanm', w_D, gradJ_esp, optimize=True)
    #gradJ_b_fit += np.einsum('xka,knm->xanm', gradJ_wD, esp, optimize=True)

    return gradJ_b_fit

def getGradbfitESP(esp,w,D,gradJ_D,gradJ_w,gradJ_esp):
    # Compute weighted D
    w_D = w[:, None] * D  # Equivalent to w.reshape((N_grid,1)) * D

    # Compute gradJ_wD more efficiently
    gradJ_wD = np.einsum('xk,ka->xka', gradJ_w, D, optimize=True)

    gradJ_wD += np.einsum('k,xka->xka', w, gradJ_D, optimize=True)

    # Compute gradJ_b_fit using optimized einsum
    gradJ_b_fit = np.einsum('ka,xknm->xanm', w_D, gradJ_esp, optimize=True)
    gradJ_b_fit += np.einsum('xka,knm->xanm', gradJ_wD, esp, optimize=True)

    return gradJ_b_fit

def getGradQtot(mol,J):
    bas_start,bas_end,ao_start,ao_end = mol.aoslice_by_atom()[J]
    aoJ_inds = np.arange(ao_start,ao_end)
    N_bas = mol.nbas
    N_AO = mol.nao
    ip = mol.intor('int1e_ipovlp',shls_slice=(bas_start,bas_end,0,N_bas))
    gradJ_Q_tot = np.zeros((3,N_AO,N_AO))
    gradJ_Q_tot[:,aoJ_inds,:] += ip
    gradJ_Q_tot[:,:,aoJ_inds] += np.swapaxes(ip,1,2)
    #gradJ_Q_tot[:,aoJ_inds,aoJ_inds] = 0.
    return gradJ_Q_tot

def getGradmutot(mol,J):
    bas_start,bas_end,ao_start,ao_end = mol.aoslice_by_atom()[J]
    aoJ_inds = np.arange(ao_start,ao_end)
    N_bas = mol.nbas
    N_AO = mol.nao
    irp = mol.intor('int1e_irp',shls_slice=(0,N_bas,bas_start,bas_end))
    irp_J = np.zeros((3,3,N_AO,irp.shape[2]))
    for beta in range(0,3):
        # <d_beta chi_nu|r_alpha|ch_mu>
        irp_J[:,beta,:,:] = irp[(3*beta):(3*beta+3),:,:]
    gradJ_mu_tot = np.zeros((3,3,N_AO,N_AO))
    gradJ_mu_tot[:,:,:,aoJ_inds] += irp_J[:,:,:,:]
    gradJ_mu_tot[:,:,aoJ_inds,:] += np.swapaxes(irp_J[:,:,:,:],2,3)
    #gradJ_mu_tot[:,aoJ_inds,aoJ_inds] = 0.
    return gradJ_mu_tot

def getGradESPGrid(J,mol,ipesp,grid_atms):
    
    N_grid = ipesp.shape[1]
    # indices of grid points belonging to J
    #gridJ_inds = np.array([k for k in range(0,N_grid) if grid_atms[k]==J])
    gridJ_inds = np.where(grid_atms == J)[0]
    # atomic orbital indices for atom J
    bas_start,bas_end,ao_start,ao_end = mol.aoslice_by_atom()[J]
    aoJ_inds = np.arange(ao_start,ao_end)
    N_AO = mol.nao
    
    gradJ_esp = np.zeros((3,N_grid,N_AO,N_AO))

    gradJ_esp[:,0:N_grid,aoJ_inds,:] -= ipesp[:,0:N_grid,aoJ_inds,:]
    gradJ_esp[:,0:N_grid,:,aoJ_inds] -= np.swapaxes(ipesp[:,0:N_grid,aoJ_inds,:],2,3)
    gradJ_esp[:,gridJ_inds,:,:] += ipesp[:,gridJ_inds,:,:]
    gradJ_esp[:,gridJ_inds,:,:] += np.swapaxes(ipesp[:,gridJ_inds,:,:],2,3)
    
    return gradJ_esp

def getGradESPGridOpt(J,mol,ipesp,grid_atms):
    
    N_grid = ipesp.shape[1]
    # indices of grid points belonging to J
    #gridJ_inds = np.array([k for k in range(0,N_grid) if grid_atms[k]==J])
    gridJ_inds = np.where(grid_atms == J)[0]
    # atomic orbital indices for atom J
    bas_start,bas_end,ao_start,ao_end = mol.aoslice_by_atom()[J]
    aoJ_inds = np.arange(ao_start,ao_end)
    N_AO = mol.nao
    
    gradJ_esp = dict()
    gradJ_esp["ao"] = -ipesp[:,0:N_grid,aoJ_inds,:]
    gradJ_esp["ao_inds"] = aoJ_inds
    gradJ_esp["grid"] = ipesp[:,gridJ_inds,:,:]
    gradJ_esp["grid_inds"] = gridJ_inds

    
    return gradJ_esp

def getGradJDESP(J,grad_D,grid_atms,l_max):
    
    gradJ_D = np.zeros(grad_D.shape)
    N_Q = grad_D.shape[2]
    N_Q_atm = getNumMultipolePerAtom(l_max)
    N = int(N_Q/N_Q_atm)
    N_grid = grad_D.shape[1]
    # indices of grid points belonging to J
    #gridJ_inds = np.array([k for k in range(0,N_grid) if grid_atms[k]==J])
    gridJ_inds = np.where(grid_atms == J)[0]
    # indices of the multipole operators corresponding to atom J
    if l_max == 0:
        aJ_inds = np.array([J])
    elif l_max == 1:
        aJ_inds = np.array([J+N*x for x in range(0,4)])
    gradJ_D[:,gridJ_inds,:] += grad_D[:,gridJ_inds,:] 
    gradJ_D[:,:,aJ_inds] -= grad_D[:,:,aJ_inds]
    
    return gradJ_D

def getGradJDESPOpt(J,grad_D,grid_atms,l_max):
    
    gradJ_D = np.zeros(grad_D.shape)
    N_Q = grad_D.shape[2]
    N_Q_atm = getNumMultipolePerAtom(l_max)
    N = int(N_Q/N_Q_atm)
    N_grid = grad_D.shape[1]
    # indices of grid points belonging to J
    #gridJ_inds = np.array([k for k in range(0,N_grid) if grid_atms[k]==J])
    gridJ_inds = np.where(grid_atms == J)[0]
    # indices of the multipole operators corresponding to atom J
    if l_max == 0:
        aJ_inds = np.array([J])
    elif l_max == 1:
        aJ_inds = np.array([J+N*x for x in range(0,4)])
    gradJ_D = dict()
    gradJ_D["rows"] = grad_D[:,gridJ_inds,:] 
    gradJ_D["rowinds"] = gridJ_inds 
    gradJ_D["cols"] = -grad_D[:,:,aJ_inds]
    gradJ_D["colinds"] = aJ_inds
    
    return gradJ_D

def getGradAfitESP(gradJ_D,gradJ_w,D,w,l_max):
    N_Q = D.shape[1]
    N_grid = D.shape[0]
    gradJ_A_fit = np.zeros((3,N_Q,N_Q))
    w_D = w.reshape((N_grid,1)) * D
    for alpha in range(0,3):
        gradJ_A_fit[alpha,:,:] = (gradJ_D[alpha,:,:].T).dot(w_D) + (D.T).dot(gradJ_w[alpha,:][:,None] * D) + (w_D.T).dot(gradJ_D[alpha,:,:])
    
    return gradJ_A_fit 

def getGradAfitESPOpt(gradJ_D,gradJ_w,D,w,w_D):
    
    N_Q = D.shape[1]
    N_grid = D.shape[0]
    gradJ_A_fit = np.zeros((3,N_Q,N_Q))
    
    dD_row = gradJ_D["rows"]
    row_inds = gradJ_D["rowinds"]
    dD_col = gradJ_D["cols"]
    col_inds = gradJ_D["colinds"]
   
    #for alpha in range(0,3):
    #    gradJ_A_fit[alpha,:,:] += (dD_row[alpha].T).dot(w_D[row_inds,:]) 
    #    gradJ_A_fit[alpha,:,:] += (w_D[row_inds,:].T).dot(dD_row[alpha,:,:])
    #    gradJ_A_fit[alpha,col_inds,:] += (dD_col[alpha,:,:].T).dot(w_D[:,col_inds]) 
    #    gradJ_A_fit[alpha,:,col_inds] += (w_D[:,col_inds].T).dot(dD_col[alpha,:,:]) 
    
    gradJ_A_fit = np.einsum('xka,kb->xab',dD_row,w_D[row_inds,:]) 
    gradJ_A_fit[:,col_inds,:] += np.einsum('xka,kb->xab',dD_col,w_D) 
    #gradJ_A_fit += np.einsum('ka,xkb->xab',w_D[row_inds,:],dD_row) 
    #gradJ_A_fit += np.einsum('ka,xkb->xab',w_D[:,col_inds],dD_col) 
    gradJ_A_fit += gradJ_A_fit.transpose(0,2,1)
    if isinstance(gradJ_w,dict):
        dw = gradJ_w["grid"]
        inds = gradJ_w["grid_inds"]
        gradJ_A_fit += np.einsum('ka,xkb->xab',D[inds,:],dw[:,:,None]*D[None,inds,:])
    else:
        gradJ_A_fit += np.einsum('ka,xkb->xab',D,gradJ_w[:,:,None]*D[None,:,:])
    #for alpha in range(0,3):
    #    gradJ_A_fit[alpha,:,:] += (D.T).dot(gradJ_w[alpha,:][:,None] * D)
    
    #for alpha in range(0,3):
    #    gradJ_A_fit[alpha,:,:] = (gradJ_D[alpha,:,:].T).dot(w_D) + (D.T).dot(gradJ_w[alpha,:][:,None] * D) + (w_D.T).dot(gradJ_D[alpha,:,:])

    return gradJ_A_fit 



def getGradESPMatrix(grid_coords,R,l_max):
    '''
    Returns grad D_ka wrt R_ka,alpha 3 x N_grid x N_Q
    '''
    N = R.shape[0]
    N_Q_atm = getNumMultipolePerAtom(l_max)
    N_Q = N * N_Q_atm
    N_grid = grid_coords.shape[0]
    D = np.zeros((N_grid,N_Q))
    
    # N_grid x N x 3
    x = grid_coords.reshape((N_grid,1,3)) - R.reshape((1,N,3))
    r = np.linalg.norm(x,axis=2)
    
    # charge part
    r_inv = 1./r
    r3_inv = r_inv * r_inv * r_inv
    grad_D = np.zeros((3,N_grid,N_Q))
    
    for alpha in range(0,3):
        grad_D[alpha,:,0:N] = -x[:,:,alpha] * r3_inv
    
    # dipole part
    if l_max>0 : 
        r5_inv = r3_inv * r_inv * r_inv
        for alpha in range(0,3):
            for beta in range(0,3):
                start = (beta+1)*N
                end = start + N
                grad_D[alpha,:,start:end] = -3.0 * x[:,:,alpha] * x[:,:,beta] * r5_inv
            start = (alpha+1)*N
            end = start + N
            grad_D[alpha,:,start:end] += r3_inv
            
    return grad_D

def softStep(x,func="sin"):
    if func == "sin":
        return softStepSin(x)
    elif func == "poly":
        return softStepPoly(x)

def derivSoftStep(x,func="sin"):
    if func == "sin":
        return derivSoftStepSin(x)
    elif func == "poly":
        return derivSoftStepPoly(x)

def derivLogSoftStep(x,func="sin"):
    if func == "sin":
        return derivLogSoftStepSin(x)
    elif func == "poly":
        return derivLogSoftStepPoly(x)

def softStepSin(x):
    '''
    Returns the soft step function smoothly goes from 0 to 1 between -1 and 1
    '''
    # step function part x>1 f=1, x<-1 f=0
    f = np.zeros(x.shape)
    f[x>1.0] = 1.0 
    # soft part -1<x<1
    inds = (np.abs(x)<1.0)
    f[inds] = 0.5 + 0.5*x[inds] +(0.5/np.pi)*np.sin(np.pi*x[inds])
    return f

def softStepPoly(x):
    '''
    Returns the soft step function smoothly goes from 0 to 1 between -1 and 1
    defined as a fifth order polynomial with continuous first and second derivs
    '''
    # step function part x>1 f=1, x<-1 f=0
    f = np.zeros(x.shape)
    f[x>1.0] = 1.0 
    # soft part -1<x<1
    inds = (np.abs(x)<(1.0-1.0e-5))
    x1 = x[inds]
    x2 = x1 * x1
    x3 = x1 * x2
    x5 = x2 * x3
    f[inds] = 0.5 + 0.9375 * x1 - 0.625 * x3 + 0.1875 * x5
    return f

def derivSoftStepPoly(x):
    '''
    Returns the soft step function smoothly goes from 0 to 1 between -1 and 1
    defined as a fifth order polynomial with continuous first and second derivs
    '''
    # step function part x>1 f=1, x<-1 f=0
    df = np.zeros(x.shape)
    # soft part -1<x<1
    inds = (np.abs(x)<(1.0-1.0e-5))
    x1 = x[inds]
    x2 = x1 * x1
    x4 = x2 * x2
    df[inds] = 0.9375 - 1.875 * x2 + 0.9375 * x4
    return df

def derivLogSoftStepPoly(x):
    # step function part |x|>1 df/dx = 0
    dlogf = np.zeros(x.shape)
    # soft part -1<x<1
    inds = (np.abs(x)<(1.0-1.0e-5)) # need to cut off before 0 or divide by zero error can occur
    x1 = x[inds]
    x2 = x1 * x1
    x3 = x1 * x2
    x4 = x2 * x2
    x5 = x1 * x4
    dlogf[inds] = (0.9375 - 1.875 * x2 + 0.9375 * x4)/(0.5 + 0.9375 * x1 - 0.625 * x3 + 0.1875 * x5)
    return dlogf

def derivSoftStepSin(x):
    '''
    Returns the derivative of the soft-step function
    '''
    # step function part |x|>1 df/dx = 0
    df = np.zeros(x.shape)
    # soft part -1<x<1
    inds = (np.abs(x)<1.0)
    df[inds] = 0.5 +(0.5)*np.cos(np.pi*x[inds])
    return df 

def derivLogSoftStepSin(x):
    '''
    Returns the derivative of the log of soft-step function
    d log(h(x))/dx = h'(x)/h(x)
    '''
    # step function part |x|>1 df/dx = 0
    dlogf = np.zeros(x.shape)
    # soft part -1<x<1
    inds = (np.abs(x)<1.0)
    dlogf[inds] = ( 0.5 +(0.5)*np.cos(np.pi*x[inds]) )/(0.5 + 0.5*x[inds] +(0.5/np.pi)*np.sin(np.pi*x[inds]))
    return dlogf 


def calculateNumGrad(func,J,R,dx=1e-3):
    
    f0 = func(R)
    gradJ_func = []
    for alpha in range(0,3):
        R_new = R+0.
        R_new[J,alpha] = R_new[J,alpha] + dx
        f_f = func(R_new)
        R_new = R+0.
        R_new[J,alpha] = R_new[J,alpha] - dx
        f_b = func(R_new)
        gradJ_func.append((f_f-f_b)/(2.0*dx))

    return np.array(gradJ_func)




class QMMultipole:
    '''
    The QMMultipole class wraps the atom-centred multipole operator functionality, giving a simple way to generate multipole
    operators and their derivatives for a given qm system.
    '''    

    def __init__(self,mol=None,multipole_order=0,multipole_method="espf"):
        self.mol = mol 
        self.multipole_order = multipole_order
        self.multipole_method = multipole_method
        self.Q = None
        self.espf_grid = None
        self.espf_grid_weights = None
        self.espf_grid_atms = None
        self.ang_grid_method = "lebedev"
        self.rad_grid_method = "vdw"
        self.n_ang = 38
        self.n_rad = 8
        self.rad_scal = 1.25
        self.weight_mode = "smooth"
        self.hard_cut_scal = 1.5
        self.weight_smooth_func = "poly"
        self.smooth_sigma = 0.2
        self.corr_espf = True
        self.grid_block_size = 16
        self.grad_D = None
        self.ipesp = None
        
        # espf regularisation
        self.espf_reg_type = None
        self.espf_reg_param = 0.0
        return
    
    def reset(self):
        self.Q = None
        self.espf_grid = None
        self.espf_grid_weights = None
        self.espf_grid_atms = None
        self.grad_D = None
        self.ipesp = None
        return
    
    def getMultipoleOperators(self,mol=None):
        '''
        Generates the set of atom-centred multipole operators
        '''
        
        if self.multipole_method == "espf":
            self.getESPFMultipoleOperators(mol=mol)
        elif self.multipole_method == "mulliken":
            self.getMullikenMultipoleOperators(mol=mol)
        else:
            raise Exception("Error:",self.multipole_method,"is not a recognised multipole method.")
        
        return self.Q
    
    def getGradMultipoleOperators(self,A):
        '''
        gets nabla_A Q_a,nm as 3 x N_Q x N_AO x N_AO numpy array
        '''
        if self.multipole_method == "espf":
            gradA_Q = self.getGradESPFMultipoleOperators(A)
        elif self.multipole_method == "mulliken":
            gradA_Q = self.getGradMullikenMultipoleOperators(A)
        else:
            raise Exception("Error:",self.multipole_method,"is not a recognised multipole method.")
            
        return gradA_Q
    
    def getGradMullikenMultipoleOperators(self,A):
        mol = self.mol
        gradA_Q = getMullikenMultipoleOperatorsAtmDeriv(mol,A,l_max=self.multipole_order,S=None,r_int=None,ip=None,irp=None)
        return gradA_Q
    
    def getGradESPFMultipoleOperators(self,A):
        mol = self.mol
        weights = self.espf_grid_weights
        hard_cut_vdw_scal = self.hard_cut_scal
        grid_sigma = self.smooth_sigma
        grid_atms = self.espf_grid_atms
        #gradA_w = calculateGridWeightDerivs(mol,A,self.espf_grid,weights,weight_mode=self.weight_mode,hard_cut_vdw_scal=hard_cut_vdw_scal,sigma=grid_sigma,grid_atms=grid_atms,smooth_func=self.weight_smooth_func)
        gradA_w = calculateGridWeightDerivsOpt(mol,A,self.espf_grid,weights,weight_mode=self.weight_mode,
                                               hard_cut_vdw_scal=hard_cut_vdw_scal,sigma=grid_sigma,grid_atms=grid_atms,
                                               smooth_func=self.weight_smooth_func)
        l_max = self.multipole_order
        if self.ipesp is None:
            self.ipesp = getIPESPMat(mol,self.espf_grid,block_size=self.grid_block_size)
        if self.grad_D is None:
            self.grad_D = getGradESPMatrix(self.espf_grid,mol.atom_coords(),l_max)
        gradA_Q = getESPFMultipoleOperatorAtmDeriv(mol,A,self.espf_grid,self.espf_fit_vars,gradA_w,self.grad_D,self.ipesp,
                                                   grid_atms,l_max=l_max,block_size=self.grid_block_size,
                                                   add_correction=self.corr_espf,Q=self.Q,reg_type=self.espf_reg_type,
                                                  reg_param=self.espf_reg_param)
        return gradA_Q
    
    def getESPFMultipoleOperators(self,mol=None):
        '''
        Generates the ESPF multipole operators
        '''
        if mol is None:
            mol = self.mol
        grid_coords,grid_atms = getESPFGrid(mol,self.n_ang,self.n_rad,rad_scal=self.rad_scal,rad_method=self.rad_grid_method,ang_method=self.ang_grid_method)
        weights = getGridWeights(mol,grid_coords,weight_mode=self.weight_mode,hard_cut_vdw_scal=self.hard_cut_scal,sigma=self.smooth_sigma,smooth_func=self.weight_smooth_func)
        self.espf_grid_weights = weights
        self.espf_grid = grid_coords
        self.espf_grid_atms = grid_atms
        self.evaluateMultipoleIntegrals(mol=mol)
        Q,fit_vars = getESPFMultipoleOperators(mol,grid_coords,weights,l_max=self.multipole_order,
                                               add_correction=self.corr_espf,return_fit_vars=True,S=self.S,r_int=self.r_int,
                                               block_size=self.grid_block_size,reg_type=self.espf_reg_type,reg_param=self.espf_reg_param)
        self.espf_fit_vars = fit_vars
        self.Q = Q 
        return 
    
    def evaluateMultipoleIntegrals(self,mol=None):
    
        if mol is None:
            mol = self.mol
        self.S = mol.intor('int1e_ovlp')
        if self.multipole_order>0:
            self.r_int = mol.intor('int1e_r')
        else:
            self.r_int = None
        return
    
    def getMullikenMultipoleOperators(self,mol=None):
        '''
        Generates Mulliken type multipole operators
        '''
        if mol is None:
            mol = self.mol
        self.evaluateMultipoleIntegrals(mol=mol)
        self.Q = getMullikenMultipoleOperators(mol,l_max=self.multipole_order,S=self.S,r_int=self.r_int)
        return 
    
    def getMultipoleOperatorDerivatives(self):
        '''
        Generates the derivatives of the atom-centred multipole operators N_QM x 3 x N_QM x N_AO x N_AO
        '''
    
        return deriv_Q