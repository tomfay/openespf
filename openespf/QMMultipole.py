import numpy as np
from scipy.linalg import sqrtm, inv, solve
from scipy.optimize import fsolve
from copy import deepcopy, copy
from pyscf import gto, scf, ao2mo, dft, lib, df
from pyscf.data import radii, elements
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer
import os
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    import pytblis
    # Assign it to a custom, short name like 'teinsum' (tensor einsum)
    np.einsum = pytblis.einsum 
except ImportError:
    # Fallback to standard numpy
    np.einsum = np.einsum



def setup_default_min_basis(add_pol=False):
    min_basis_ref = MODULE_DIR + "/data/scalmini.nw"
    min_basis_mod = {}
    
    # 1. Eagerly construct the dictionary for elements up to Radon (Z=86)
    for Z in range(1, 87):
        symb = elements.ELEMENTS[Z]
        try:
            # Try to load the primary basis
            min_basis_mod[symb] = gto.basis.load(min_basis_ref, symb)
        except (RuntimeError, KeyError):
            try:
                # Fallback to sto-3g if the element is missing
                min_basis_mod[symb] = gto.basis.load('sto-3g', symb)
            except (RuntimeError, KeyError):
                pass # Skip if the element is missing in both bases

    # 2. Append polarization functions if requested
    if add_pol:
        if 'H' in min_basis_mod:
            min_basis_mod['H'] = min_basis_mod['H'] + [[1, [0.727, 1]]]
        if 'He' in min_basis_mod:
            min_basis_mod['He'] = min_basis_mod['He'] + [[1, [1.275, 1.0]]]

    return min_basis_mod
    
def solve_kkt_tensor_reshaped(A, b, C, d):
    """
    Solves the KKT constrained minimization problem over the last dimension 
    of a tensor and moves the parameter dimension to the front.
    """
    N = A.shape[0]
    M = C.shape[1]
    
    top_row = np.hstack((A, C))
    bottom_row = np.hstack((C.T, np.zeros((M, M))))
    kkt_mat = np.vstack((top_row, bottom_row))
    
    rhs = np.concatenate((b, d), axis=-1)
    
    kkt_inv = inv(kkt_mat)
    solution = np.einsum('ij,...j->...i', kkt_inv, rhs)
    
    x = solution[..., :N]
    y = np.moveaxis(x, source=-1, destination=0)
    constraint_vals = x @ C 
    # Calculate the maximum absolute error across all dimensions
    max_error = np.max(np.abs(constraint_vals - d))
    #print(f"Max constraint violation |C^T x - d|: {max_error:.4e}")

    return y    
    
quad_map = [
    (0, 0, 0, 1.0), # xx
    (0, 1, 1, 2.0), # xy (weight 2 for off-diagonal rotational invariance)
    (0, 2, 2, 2.0), # xz
    (1, 1, 4, 1.0), # yy
    (1, 2, 5, 2.0), # yz
    (2, 2, 8, 1.0)  # zz
]    
    
def getDMFitChargeOperatorSphericalOld(mol, min_basis='minao', reg=1e-12, reg_r2=0.0, reg_S=0.0, 
                                    w_df=1.0, w_mu=0.0, w_ovlp=0.0, w_quad=0.0, aux_basis='weigend-jfit', 
                                    return_fit=False, constrain_dipole=False, constrain_quad=False):
    """
    Fits the orbital pair density to a minimal basis atom-centered spherically
    symmetric density, and extracts the charge operator Q_{A, nm}.
    """

    
    # =========================================================================
    # Build Minimal Basis & Pre-Compute Global Integrals
    # =========================================================================
    atom_mols = []
    min_ao_ranges = []
    ao_offset = 0
    
    for i in range(mol.natm):
        atm_mol = gto.Mole()
        atm_mol.atom = [[mol.atom_symbol(i), mol.atom_coord(i)]] 
        atm_mol.unit = 'Bohr'
        atm_mol.basis = min_basis
        atm_mol.spin = 0
        atm_mol.charge = mol.atom_charge(i)
        atm_mol.build()
        atom_mols.append(atm_mol)
        
        min_ao_ranges.append((ao_offset, ao_offset + atm_mol.nao))
        ao_offset += atm_mol.nao

    # Out-of-place addition to prevent shallow copy mutation bugs
    min_mol = atom_mols[0]
    for atm in atom_mols[1:]:
        min_mol = min_mol + atm

    Np = mol.nao
    nshl_parent, nshl_min = mol.nbas, min_mol.nbas
    
    mol_comb = mol + min_mol
    shls_slice = (0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_min, nshl_parent, nshl_parent + nshl_min)
    
    eri_parent_min_full = mol_comb.intor('int2e', shls_slice=shls_slice)
    eri_min_min_full = min_mol.intor('int2e')

    S_parent = mol.intor('int1e_ovlp')
    mu_parent = -mol.intor('int1e_r') 
    rr_parent = mol.intor('int1e_rr')

    # =========================================================================
    # Construct Spherically Averaged Basis Pairs & Local Vectors
    # =========================================================================
    spherical_pairs = []
    S_atomic, mu_atomic_shifted, r_atomic, rr_atomic, r2_atomic = [], [], [], [], []

    for A, mol_A in enumerate(atom_mols):
        R_A = mol.atom_coord(A) 
        
        mol_A.set_common_orig(R_A)
        mu_atomic_shifted.append(-mol_A.intor('int1e_r'))
        r_atomic.append(mol_A.intor('int1e_r'))
        rr_atomic.append(mol_A.intor('int1e_rr'))
        r2_atomic.append(mol_A.intor('int1e_r2'))
        mol_A.set_common_orig((0.0, 0.0, 0.0))
        S_atomic.append(mol_A.intor('int1e_ovlp'))

        ao_loc = mol_A.ao_loc_nr()
        n_shells = mol_A.nbas
        
        for i in range(n_shells):
            l_i = mol_A.bas_angular(i)
            for j in range(i + 1):
                if l_i == mol_A.bas_angular(j):
                    a_idx, b_idx = np.arange(ao_loc[i], ao_loc[i+1]), np.arange(ao_loc[j], ao_loc[j+1])
                    spherical_pairs.append({
                        'atom': A,
                        'ab_pairs': np.column_stack((a_idx, b_idx)),
                        'min_ao_a': min_ao_ranges[A][0] + a_idx,
                        'min_ao_b': min_ao_ranges[A][0] + b_idx
                    })

    M_pairs = len(spherical_pairs)

    # Pre-compute Global Contributions
    V_pairs = np.zeros((M_pairs, 3))
    V_quad_pairs = np.zeros((M_pairs, 6))
    
    for I, pI in enumerate(spherical_pairs):
        A_idx = pI['atom']
        R_A = mol.atom_coord(A_idx)
        a, b = pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]
        
        sum_mu = np.sum(mu_atomic_shifted[A_idx][:, a, b], axis=1)
        sum_S = np.sum(S_atomic[A_idx][a, b])
        V_pairs[I] = sum_mu - sum_S * R_A
            
        if w_quad > 0.0 or constrain_quad:
            for q, (i, j, f, w) in enumerate(quad_map):
                V_quad_pairs[I, q] = (np.sum(rr_atomic[A_idx][f, a, b]) + 
                                      R_A[j] * np.sum(r_atomic[A_idx][i, a, b]) + 
                                      R_A[i] * np.sum(r_atomic[A_idx][j, a, b]) + 
                                      R_A[i] * R_A[j] * sum_S)

    # =========================================================================
    # Build Auxiliary Basis Metric (L2 Overlap)
    # =========================================================================
    if w_ovlp > 0.0 and aux_basis is not None:
        auxmol = df.make_auxmol(mol, aux_basis)
        N_aux, nshl_aux = auxmol.nao, auxmol.nbas
        
        S_aux_inv = np.linalg.pinv(auxmol.intor('int1e_ovlp'), rcond=1e-8)
        mol_parent_aux = mol + auxmol
        ovlp3c_parent = mol_parent_aux.intor('int3c1e', shls_slice=(0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_aux))
        
        ovlp3c_min_aux = (min_mol + auxmol).intor('int3c1e', shls_slice=(0, nshl_min, 0, nshl_min, nshl_min, nshl_min + nshl_aux))
        ovlp3c_I = np.array([np.sum(ovlp3c_min_aux[pI['min_ao_a'], pI['min_ao_b'], :], axis=0) for pI in spherical_pairs])
        ovlp3c_I_proj = ovlp3c_I @ S_aux_inv

    # =========================================================================
    # Build Vectorized KKT Matrices
    # =========================================================================
    n_constraints = 1 + (3 if constrain_dipole else 0) + (6 if constrain_quad else 0)
    C_mat = np.zeros((M_pairs, n_constraints))
    b_tensor_df = np.zeros((Np, Np, M_pairs))
    A_mat_df = np.zeros((M_pairs, M_pairs))

    for I, pI in enumerate(spherical_pairs):
        a_I, b_I = pI['min_ao_a'], pI['min_ao_b']
        C_mat[I, 0] = np.sum(S_atomic[pI['atom']][pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]])
        b_tensor_df[:, :, I] = np.sum(eri_parent_min_full[:, :, a_I, b_I], axis=-1)
        
        eri_slice = eri_min_min_full[a_I, b_I, :, :]
        for J, pJ in enumerate(spherical_pairs):
            A_mat_df[I, J] = np.sum(eri_slice[:, pJ['min_ao_a'], pJ['min_ao_b']])

    # Apply Weights and Penalities
    b_tensor = w_df * b_tensor_df.copy()
    A_mat = w_df * A_mat_df.copy()

    if reg_S > 0.0:
        b_tensor += reg_S * np.einsum('i,jk->jki', C_mat[:, 0], S_parent)
        A_mat += reg_S * np.outer(C_mat[:, 0], C_mat[:, 0])
        
    if w_mu > 0.0:
        for x in range(3):
            b_tensor += w_mu * np.einsum('i,jk->jki', V_pairs[:, x], mu_parent[x])
        A_mat += w_mu * (V_pairs @ V_pairs.T)
        
    if w_quad > 0.0:
        W_vec = np.array([w for (_, _, _, w) in quad_map])
        for q, (i, j, f, w) in enumerate(quad_map):
            b_tensor += w_quad * w * np.einsum('i,jk->jki', V_quad_pairs[:, q], rr_parent[f])
        A_mat += w_quad * (V_quad_pairs * W_vec) @ V_quad_pairs.T
        
    if reg_r2 > 0.0:
        R_vdw = radii.VDW[min_mol.atom_charges()]
        for I, pI in enumerate(spherical_pairs):
            A_idx = pI['atom']
            sum_r2_I = np.sum(r2_atomic[A_idx][pI['ab_pairs'][:,0], pI['ab_pairs'][:,1]])
            for J, pJ in enumerate(spherical_pairs):
                if A_idx == pJ['atom']:
                    sum_r2_J = np.sum(r2_atomic[A_idx][pJ['ab_pairs'][:,0], pJ['ab_pairs'][:,1]])
                    A_mat[I, J] += (reg_r2/(R_vdw[A_idx]**4)) * sum_r2_I * sum_r2_J

    if w_ovlp > 0.0 and aux_basis is not None:
        A_mat += w_ovlp * (ovlp3c_I_proj @ ovlp3c_I.T)
        b_tensor += w_ovlp * (ovlp3c_parent.reshape(-1, N_aux) @ ovlp3c_I_proj.T).reshape(Np, Np, M_pairs)

    # Populate Constraints
    d_tensor = np.zeros((Np, Np, n_constraints))
    d_tensor[:, :, 0] = S_parent
    col_idx = 1
    
    if constrain_dipole:
        C_mat[:, col_idx:col_idx+3] = V_pairs
        for x in range(3):
            d_tensor[:, :, col_idx+x] = mu_parent[x]
        col_idx += 3
        
    if constrain_quad:
        for q, (i, j, f, w) in enumerate(quad_map):
            C_mat[:, col_idx+q] = V_quad_pairs[:, q]
            d_tensor[:, :, col_idx+q] = rr_parent[f]

    if reg > 0.0:
        A_mat += reg * np.eye(M_pairs)
        
    # Solve System
    X_I_nm = solve_kkt_tensor_reshaped(A_mat, b_tensor, C_mat, d_tensor)
    Q_A = np.zeros((mol.natm, Np, Np))

    for I, pI in enumerate(spherical_pairs):
        Q_A[pI['atom']] -= X_I_nm[I, :, :] * C_mat[I, 0]

    if return_fit:
        return Q_A, A_mat_df, b_tensor_df, X_I_nm
    return Q_A  
    
    
def getDMFitChargeOperatorSphericalNew(mol, min_basis='minao', reg=1e-12, reg_r2=0.0, reg_S=0.0, 
                                    w_df=1.0, w_mu=0.0, w_ovlp=0.0, w_quad=0.0, aux_basis='weigend-jfit', 
                                    return_fit=False, constrain_dipole=False, constrain_quad=False):
    """
    Fits the orbital pair density to a minimal basis atom-centered spherically
    symmetric density, and extracts the charge operator Q_{A, nm}.
    """
    
    # =========================================================================
    # Build Minimal Basis & Pre-Compute Global Integrals
    # =========================================================================
    
    atom_mols = []
    min_ao_ranges = []
    ao_offset = 0
    #t0 = timer()
    for i in range(mol.natm):
        atm_mol = gto.Mole()
        atm_mol.atom = [[mol.atom_symbol(i), mol.atom_coord(i)]] 
        atm_mol.unit = 'Bohr'
        atm_mol.basis = min_basis
        atm_mol.spin = 0
        atm_mol.charge = mol.atom_charge(i)
        atm_mol.build()
        atom_mols.append(atm_mol)
        
        min_ao_ranges.append((ao_offset, ao_offset + atm_mol.nao))
        ao_offset += atm_mol.nao

    # Out-of-place addition to prevent shallow copy mutation bugs
    min_mol = atom_mols[0]
    for atm in atom_mols[1:]:
        min_mol = min_mol + atm

    Np = mol.nao
    nshl_parent, nshl_min = mol.nbas, min_mol.nbas
    #print("initial mol set-up:", timer()-t0)
    S_parent = mol.intor('int1e_ovlp')
    mu_parent = -mol.intor('int1e_r') 
    rr_parent = mol.intor('int1e_rr')
    
    # =========================================================================
    # Construct Spherically Averaged Basis Pairs & Local Vectors
    # =========================================================================
    spherical_pairs = []
    S_atomic, mu_atomic_shifted, r_atomic, rr_atomic, r2_atomic = [], [], [], [], []

    for A, mol_A in enumerate(atom_mols):
        R_A = mol.atom_coord(A) 
        
        mol_A.set_common_orig(R_A)
        mu_atomic_shifted.append(-mol_A.intor('int1e_r'))
        r_atomic.append(mol_A.intor('int1e_r'))
        rr_atomic.append(mol_A.intor('int1e_rr'))
        r2_atomic.append(mol_A.intor('int1e_r2'))
        mol_A.set_common_orig((0.0, 0.0, 0.0))
        S_atomic.append(mol_A.intor('int1e_ovlp'))

        ao_loc = mol_A.ao_loc_nr()
        n_shells = mol_A.nbas
        
        for i in range(n_shells):
            l_i = mol_A.bas_angular(i)
            for j in range(i + 1):
                if l_i == mol_A.bas_angular(j):
                    a_idx, b_idx = np.arange(ao_loc[i], ao_loc[i+1]), np.arange(ao_loc[j], ao_loc[j+1])
                    spherical_pairs.append({
                        'atom': A,
                        'ab_pairs': np.column_stack((a_idx, b_idx)),
                        'min_ao_a': min_ao_ranges[A][0] + a_idx,
                        'min_ao_b': min_ao_ranges[A][0] + b_idx
                    })

    M_pairs = len(spherical_pairs)

    # Pre-compute Global Contributions
    V_pairs = np.zeros((M_pairs, 3))
    V_quad_pairs = np.zeros((M_pairs, 6))
    
    for I, pI in enumerate(spherical_pairs):
        A_idx = pI['atom']
        R_A = mol.atom_coord(A_idx)
        a, b = pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]
        
        sum_mu = np.sum(mu_atomic_shifted[A_idx][:, a, b], axis=1)
        sum_S = np.sum(S_atomic[A_idx][a, b])
        V_pairs[I] = sum_mu - sum_S * R_A
            
        if w_quad > 0.0 or constrain_quad:
            for q, (i, j, f, w) in enumerate(quad_map):
                V_quad_pairs[I, q] = (np.sum(rr_atomic[A_idx][f, a, b]) + 
                                      R_A[j] * np.sum(r_atomic[A_idx][i, a, b]) + 
                                      R_A[i] * np.sum(r_atomic[A_idx][j, a, b]) + 
                                      R_A[i] * R_A[j] * sum_S)
    
    
    # =========================================================================
    # Build Auxiliary Basis Metric (L2 Overlap)
    # =========================================================================

    if w_ovlp > 0.0 and aux_basis is not None:
        auxmol = df.make_auxmol(mol, aux_basis)
        N_aux, nshl_aux = auxmol.nao, auxmol.nbas
        
        S_aux_inv = np.linalg.pinv(auxmol.intor('int1e_ovlp'), rcond=1e-8)
        mol_parent_aux = mol + auxmol
        ovlp3c_parent = mol_parent_aux.intor('int3c1e', shls_slice=(0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_aux))
        
        ovlp3c_min_aux = (min_mol + auxmol).intor('int3c1e', shls_slice=(0, nshl_min, 0, nshl_min, nshl_min, nshl_min + nshl_aux))
        ovlp3c_I = np.array([np.sum(ovlp3c_min_aux[pI['min_ao_a'], pI['min_ao_b'], :], axis=0) for pI in spherical_pairs])
        ovlp3c_I_proj = ovlp3c_I @ S_aux_inv
    
    # =========================================================================
    # Build Vectorized KKT Matrices (Optimized Atom-Block ERI Processing)
    # =========================================================================
    n_constraints = 1 + (3 if constrain_dipole else 0) + (6 if constrain_quad else 0)
    C_mat = np.zeros((M_pairs, n_constraints))
    b_tensor_df = np.zeros((Np, Np, M_pairs))
    A_mat_df = np.zeros((M_pairs, M_pairs))

    # Group pairs by atom to localize ERI calculations
    pairs_by_atom = [[] for _ in range(mol.natm)]
    for I, pI in enumerate(spherical_pairs):
        pairs_by_atom[pI['atom']].append((I, pI))

    for A, atm_mol_A in enumerate(atom_mols):
        if not pairs_by_atom[A]:
            continue
            
        # 1. Compute (parent, parent | A, A) localized ERIs
        mol_parent_A = mol + atm_mol_A
        nshl_A = atm_mol_A.nbas
        slice_parent_A = (0, nshl_parent, 0, nshl_parent, 
                          nshl_parent, nshl_parent + nshl_A, 
                          nshl_parent, nshl_parent + nshl_A)
        eri_parent_A = mol_parent_A.intor('int2e', shls_slice=slice_parent_A)
        
        for I, pI in pairs_by_atom[A]:
            a_loc, b_loc = pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]
            C_mat[I, 0] = np.sum(S_atomic[A][a_loc, b_loc])
            b_tensor_df[:, :, I] = np.sum(eri_parent_A[:, :, a_loc, b_loc], axis=-1)

        # 2. Compute (A, A | B, B) localized ERIs
        for B in range(A, mol.natm):
            atm_mol_B = atom_mols[B]
            if not pairs_by_atom[B]:
                continue
                
            mol_A_B = atm_mol_A + atm_mol_B
            nshl_B = atm_mol_B.nbas
            slice_AB = (0, nshl_A, 0, nshl_A, nshl_A, nshl_A + nshl_B, nshl_A, nshl_A + nshl_B)
            eri_A_B = mol_A_B.intor('int2e', shls_slice=slice_AB)
            
            for I, pI in pairs_by_atom[A]:
                a_loc, b_loc = pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]
                for J, pJ in pairs_by_atom[B]:
                    c_loc, d_loc = pJ['ab_pairs'][:, 0], pJ['ab_pairs'][:, 1]
                    
                    val = np.sum(eri_A_B[a_loc, b_loc][:, c_loc, d_loc])
                    A_mat_df[I, J] = val
                    if A != B: # Mirror the off-diagonal atom blocks
                        A_mat_df[J, I] = val

    # Apply Weights and Penalities
    b_tensor = w_df * b_tensor_df.copy()
    A_mat = w_df * A_mat_df.copy()

    if reg_S > 0.0:
        b_tensor += reg_S * np.einsum('i,jk->jki', C_mat[:, 0], S_parent)
        A_mat += reg_S * np.outer(C_mat[:, 0], C_mat[:, 0])
        
    if w_mu > 0.0:
        for x in range(3):
            b_tensor += w_mu * np.einsum('i,jk->jki', V_pairs[:, x], mu_parent[x])
        A_mat += w_mu * (V_pairs @ V_pairs.T)
        
    if w_quad > 0.0:
        W_vec = np.array([w for (_, _, _, w) in quad_map])
        for q, (i, j, f, w) in enumerate(quad_map):
            b_tensor += w_quad * w * np.einsum('i,jk->jki', V_quad_pairs[:, q], rr_parent[f])
        A_mat += w_quad * (V_quad_pairs * W_vec) @ V_quad_pairs.T
        
    if reg_r2 > 0.0:
        R_vdw = radii.VDW[min_mol.atom_charges()]
        for I, pI in enumerate(spherical_pairs):
            A_idx = pI['atom']
            sum_r2_I = np.sum(r2_atomic[A_idx][pI['ab_pairs'][:,0], pI['ab_pairs'][:,1]])
            for J, pJ in enumerate(spherical_pairs):
                if A_idx == pJ['atom']:
                    sum_r2_J = np.sum(r2_atomic[A_idx][pJ['ab_pairs'][:,0], pJ['ab_pairs'][:,1]])
                    A_mat[I, J] += (reg_r2/(R_vdw[A_idx]**4)) * sum_r2_I * sum_r2_J

    if w_ovlp > 0.0 and aux_basis is not None:
        A_mat += w_ovlp * (ovlp3c_I_proj @ ovlp3c_I.T)
        b_tensor += w_ovlp * (ovlp3c_parent.reshape(-1, N_aux) @ ovlp3c_I_proj.T).reshape(Np, Np, M_pairs)

    # Populate Constraints
    d_tensor = np.zeros((Np, Np, n_constraints))
    d_tensor[:, :, 0] = S_parent
    col_idx = 1
    
    if constrain_dipole:
        C_mat[:, col_idx:col_idx+3] = V_pairs
        for x in range(3):
            d_tensor[:, :, col_idx+x] = mu_parent[x]
        col_idx += 3
        
    if constrain_quad:
        for q, (i, j, f, w) in enumerate(quad_map):
            C_mat[:, col_idx+q] = V_quad_pairs[:, q]
            d_tensor[:, :, col_idx+q] = rr_parent[f]

    if reg > 0.0:
        A_mat += reg * np.eye(M_pairs)
        
    # Solve System
    X_I_nm = solve_kkt_tensor_reshaped(A_mat, b_tensor, C_mat, d_tensor)
    Q_A = np.zeros((mol.natm, Np, Np))

    for I, pI in enumerate(spherical_pairs):
        Q_A[pI['atom']] -= X_I_nm[I, :, :] * C_mat[I, 0]

    if return_fit:
        return Q_A, A_mat_df, b_tensor_df, X_I_nm
    return Q_A    



# Hard-coded Clebsch-Gordan coefficients for L=1
# Format: CG_L1[(l1, l2)][M] = [(m1, m2, coeff), ...]
# where M, m1, m2 are PySCF magnetic quantum number indices



def getDMFitChargeOperatorSphericalNew2(mol, min_basis='minao', reg=1e-12, reg_r2=0.0, reg_S=0.0, 
                                    w_df=1.0, w_mu=0.0, w_ovlp=0.0, w_quad=0.0, aux_basis='weigend-jfit', 
                                    return_fit=False, constrain_dipole=False, constrain_quad=False):
    
    #t0 = timer()
    
    # =========================================================================
    # FAST SETUP: Single Global Build (No atm_mols)
    # =========================================================================
    min_mol = gto.Mole()
    min_mol.atom = mol.atom 
    min_mol.unit = mol.unit
    min_mol.basis = min_basis
    min_mol.charge = mol.charge
    min_mol.spin = mol.spin
    min_mol.build(dump_input=False, parse_arg=False)

    Np = mol.nao
    nshl_parent = mol.nbas
    nshl_min = min_mol.nbas
    min_aoslices = min_mol.aoslice_by_atom()
    ao_loc = min_mol.ao_loc_nr()
    
    # Pre-build the combined molecule ONCE for cross-ERIs
    mol_parent_min = mol + min_mol

    #print("initial mol set-up:", timer() - t0)

    # Global integrals for parent
    S_parent = mol.intor('int1e_ovlp')
    mu_parent = -mol.intor('int1e_r') 
    rr_parent = mol.intor('int1e_rr')

    # Global 1-electron integrals for min_mol (No manual origin shifting needed)
    S_min_global = min_mol.intor('int1e_ovlp')
    r_min_global = min_mol.intor('int1e_r')
    mu_min_global = -r_min_global
    rr_min_global = min_mol.intor('int1e_rr')
    r2_min_global = min_mol.intor('int1e_r2')

    # =========================================================================
    # Construct Spherically Averaged Basis Pairs
    # =========================================================================
    spherical_pairs = []

    for A in range(mol.natm):
        shl0, shl1, ao0, ao1 = min_aoslices[A]
        for i_g in range(shl0, shl1):
            l_i = min_mol.bas_angular(i_g)
            for j_g in range(shl0, i_g + 1):
                if l_i == min_mol.bas_angular(j_g):
                    a_idx_global = np.arange(ao_loc[i_g], ao_loc[i_g+1])
                    b_idx_global = np.arange(ao_loc[j_g], ao_loc[j_g+1])
                    
                    # Local AO indices perfectly map to the isolated shell slices
                    spherical_pairs.append({
                        'atom': A,
                        'ab_pairs': np.column_stack((a_idx_global - ao0, b_idx_global - ao0)),
                        'min_ao_a': a_idx_global,
                        'min_ao_b': b_idx_global
                    })

    M_pairs = len(spherical_pairs)

    # Pre-compute Global Contributions
    V_pairs = np.zeros((M_pairs, 3))
    V_quad_pairs = np.zeros((M_pairs, 6))
    S_pair_vals = np.zeros(M_pairs)
    r2_shifted_pair_vals = np.zeros(M_pairs)
    
    for I, pI in enumerate(spherical_pairs):
        a_g, b_g = pI['min_ao_a'], pI['min_ao_b']
        R_A = mol.atom_coord(pI['atom'])
        
        sum_S = np.sum(S_min_global[a_g, b_g])
        S_pair_vals[I] = sum_S
        
        # Unshifted global arrays analytically cancel out the manual shift
        V_pairs[I] = np.sum(mu_min_global[:, a_g, b_g], axis=1)
            
        if w_quad > 0.0 or constrain_quad:
            for q, (i, j, f, w) in enumerate(quad_map):
                V_quad_pairs[I, q] = np.sum(rr_min_global[f, a_g, b_g])
                
        if reg_r2 > 0.0:
            # Analytically shift the atom-centered variance for reg_r2
            sum_r = np.sum(r_min_global[:, a_g, b_g], axis=1)
            sum_r2 = np.sum(r2_min_global[a_g, b_g])
            r2_shifted_pair_vals[I] = sum_r2 - 2.0 * np.dot(R_A, sum_r) + np.dot(R_A, R_A) * sum_S
            
    # =========================================================================
    # Build Auxiliary Basis Metric (L2 Overlap)
    # =========================================================================
    if w_ovlp > 0.0 and aux_basis is not None:
        auxmol = df.make_auxmol(mol, aux_basis)
        N_aux, nshl_aux = auxmol.nao, auxmol.nbas
        
        S_aux_inv = np.linalg.pinv(auxmol.intor('int1e_ovlp'), rcond=1e-8)
        mol_parent_aux = mol + auxmol
        ovlp3c_parent = mol_parent_aux.intor('int3c1e', shls_slice=(0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_aux))
        
        ovlp3c_min_aux = (min_mol + auxmol).intor('int3c1e', shls_slice=(0, nshl_min, 0, nshl_min, nshl_min, nshl_min + nshl_aux))
        ovlp3c_I = np.array([np.sum(ovlp3c_min_aux[pI['min_ao_a'], pI['min_ao_b'], :], axis=0) for pI in spherical_pairs])
        ovlp3c_I_proj = ovlp3c_I @ S_aux_inv

    # =========================================================================
    # Build Vectorized KKT Matrices (Optimized Atom-Block ERI Processing)
    # =========================================================================
    n_constraints = 1 + (3 if constrain_dipole else 0) + (6 if constrain_quad else 0)
    C_mat = np.zeros((M_pairs, n_constraints))
    b_tensor_df = np.zeros((Np, Np, M_pairs))
    A_mat_df = np.zeros((M_pairs, M_pairs))

    pairs_by_atom = [[] for _ in range(mol.natm)]
    for I, pI in enumerate(spherical_pairs):
        pairs_by_atom[pI['atom']].append((I, pI))

    for A in range(mol.natm):
        if not pairs_by_atom[A]:
            continue
            
        shl0_A, shl1_A, ao0_A, ao1_A = min_aoslices[A]
        
        # 1. Compute (parent, parent | A, A) ERIs directly from combined mol
        slice_parent_A = (0, nshl_parent, 0, nshl_parent, 
                          nshl_parent + shl0_A, nshl_parent + shl1_A, 
                          nshl_parent + shl0_A, nshl_parent + shl1_A)
        eri_parent_A = mol_parent_min.intor('int2e', shls_slice=slice_parent_A)
        
        for I, pI in pairs_by_atom[A]:
            a_loc, b_loc = pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]
            C_mat[I, 0] = S_pair_vals[I]
            b_tensor_df[:, :, I] = np.sum(eri_parent_A[:, :, a_loc, b_loc], axis=-1)

        # 2. Compute (A, A | B, B) ERIs directly from min_mol
        for B in range(A, mol.natm):
            if not pairs_by_atom[B]:
                continue
                
            shl0_B, shl1_B, ao0_B, ao1_B = min_aoslices[B]
            slice_AB = (shl0_A, shl1_A, shl0_A, shl1_A, shl0_B, shl1_B, shl0_B, shl1_B)
            eri_A_B = min_mol.intor('int2e', shls_slice=slice_AB)
            
            for I, pI in pairs_by_atom[A]:
                a_loc, b_loc = pI['ab_pairs'][:, 0], pI['ab_pairs'][:, 1]
                for J, pJ in pairs_by_atom[B]:
                    c_loc, d_loc = pJ['ab_pairs'][:, 0], pJ['ab_pairs'][:, 1]
                    
                    val = np.sum(eri_A_B[a_loc, b_loc][:, c_loc, d_loc])
                    A_mat_df[I, J] = val
                    if A != B: 
                        A_mat_df[J, I] = val

    # Apply Weights and Penalities
    b_tensor = w_df * b_tensor_df.copy()
    A_mat = w_df * A_mat_df.copy()

    if reg_S > 0.0:
        b_tensor += reg_S * np.einsum('i,jk->jki', C_mat[:, 0], S_parent)
        A_mat += reg_S * np.outer(C_mat[:, 0], C_mat[:, 0])
        
    if w_mu > 0.0:
        for x in range(3):
            b_tensor += w_mu * np.einsum('i,jk->jki', V_pairs[:, x], mu_parent[x])
        A_mat += w_mu * (V_pairs @ V_pairs.T)
        
    if w_quad > 0.0:
        W_vec = np.array([w for (_, _, _, w) in quad_map])
        for q, (i, j, f, w) in enumerate(quad_map):
            b_tensor += w_quad * w * np.einsum('i,jk->jki', V_quad_pairs[:, q], rr_parent[f])
        A_mat += w_quad * (V_quad_pairs * W_vec) @ V_quad_pairs.T
        
    if reg_r2 > 0.0:
        R_vdw = radii.VDW[min_mol.atom_charges()]
        for I, pI in enumerate(spherical_pairs):
            A_idx = pI['atom']
            sum_r2_I = r2_shifted_pair_vals[I]
            for J, pJ in enumerate(spherical_pairs):
                if A_idx == pJ['atom']:
                    sum_r2_J = r2_shifted_pair_vals[J]
                    A_mat[I, J] += (reg_r2/(R_vdw[A_idx]**4)) * sum_r2_I * sum_r2_J

    if w_ovlp > 0.0 and aux_basis is not None:
        A_mat += w_ovlp * (ovlp3c_I_proj @ ovlp3c_I.T)
        b_tensor += w_ovlp * (ovlp3c_parent.reshape(-1, N_aux) @ ovlp3c_I_proj.T).reshape(Np, Np, M_pairs)

    # Populate Constraints
    d_tensor = np.zeros((Np, Np, n_constraints))
    d_tensor[:, :, 0] = S_parent
    col_idx = 1
    
    if constrain_dipole:
        C_mat[:, col_idx:col_idx+3] = V_pairs
        for x in range(3):
            d_tensor[:, :, col_idx+x] = mu_parent[x]
        col_idx += 3
        
    if constrain_quad:
        for q, (i, j, f, w) in enumerate(quad_map):
            C_mat[:, col_idx+q] = V_quad_pairs[:, q]
            d_tensor[:, :, col_idx+q] = rr_parent[f]

    if reg > 0.0:
        A_mat += reg * np.eye(M_pairs)
        
    # Solve System
    X_I_nm = solve_kkt_tensor_reshaped(A_mat, b_tensor, C_mat, d_tensor)
    Q_A = np.zeros((mol.natm, Np, Np))

    for I, pI in enumerate(spherical_pairs):
        Q_A[pI['atom']] -= X_I_nm[I, :, :] * C_mat[I, 0]

    if return_fit:
        return Q_A, A_mat_df, b_tensor_df, X_I_nm
    return Q_A


CG_L1 = {
    (0, 1): {
        0: [(0, 0, 1.0)],
        1: [(0, 1, 1.0)],
        2: [(0, 2, 1.0)]
    },
    (1, 0): {
        0: [(0, 0, 1.0)],
        1: [(1, 0, 1.0)],
        2: [(2, 0, 1.0)]
    },
    (1, 2): {
        0: [(0, 2,  0.31622776601683794), (0, 4,  0.5477225575051661), 
            (1, 1, -0.5477225575051661), (2, 0, -0.5477225575051661)],
        1: [(0, 1, -0.5477225575051661), (1, 2, -0.6324555320336759), 
            (2, 3, -0.5477225575051661)],
        2: [(0, 0, -0.5477225575051661), (1, 3, -0.5477225575051661), 
            (2, 2,  0.31622776601683794), (2, 4, -0.5477225575051661)]
    },
    (2, 1): {
        0: [(2, 0,  0.31622776601683794), (4, 0,  0.5477225575051661), 
            (1, 1, -0.5477225575051661), (0, 2, -0.5477225575051661)],
        1: [(1, 0, -0.5477225575051661), (2, 1, -0.6324555320336759), 
            (3, 2, -0.5477225575051661)],
        2: [(0, 0, -0.5477225575051661), (3, 1, -0.5477225575051661), 
            (2, 2,  0.31622776601683794), (4, 2, -0.5477225575051661)]
    }
}

valence_l_values = [
    [0, 1],    # 0: H (Z=1) - include p for polarisation
    [0, 1],    # 1: He (Z=2) - include p for polarisation
    [0, 1],    # 2: Li (Z=3) - include p for polarisation
    [0, 1],    # 3: Be (Z=4) - include p for polarisation
    [0, 1],    # 4: B (Z=5)
    [0, 1],    # 5: C (Z=6)
    [0, 1],    # 6: N (Z=7)
    [0, 1],    # 7: O (Z=8)
    [0, 1],    # 8: F (Z=9)
    [0, 1],    # 9: Ne (Z=10)
    [0, 1],    # 10: Na (Z=11)
    [0, 1],    # 11: Mg (Z=12)
    [0, 1],    # 12: Al (Z=13)
    [0, 1],    # 13: Si (Z=14)
    [0, 1],    # 14: P (Z=15)
    [0, 1],    # 15: S (Z=16)
    [0, 1],    # 16: Cl (Z=17)
    [0, 1],    # 17: Ar (Z=18)
    [0, 1],    # 18: K (Z=19)
    [0, 1],    # 19: Ca (Z=20)
    [0, 1, 2], # 20: Sc (Z=21) - include 3d
    [0, 1, 2], # 21: Ti (Z=22) - include 3d
    [0, 1, 2], # 22: V (Z=23) - include 3d
    [0, 1, 2], # 23: Cr (Z=24) - include 3d
    [0, 1, 2], # 24: Mn (Z=25) - include 3d
    [0, 1, 2], # 25: Fe (Z=26) - include 3d
    [0, 1, 2], # 26: Co (Z=27) - include 3d
    [0, 1, 2], # 27: Ni (Z=28) - include 3d
    [0, 1, 2], # 28: Cu (Z=29) - include 3d
    [0, 1, 2], # 29: Zn (Z=30) - include 3d
    [0, 1],    # 30: Ga (Z=31) - treat 3d as core
    [0, 1],    # 31: Ge (Z=32) - treat 3d as core
    [0, 1],    # 32: As (Z=33) - treat 3d as core
    [0, 1],    # 33: Se (Z=34) - treat 3d as core
    [0, 1],    # 34: Br (Z=35) - treat 3d as core
    [0, 1]     # 35: Kr (Z=36) - treat 3d as core
]

def getDMFitChargeOperatorDipoleSphericalOld(mol, min_basis='minao', reg=1e-12, reg_r2=0.0, reg_S=0.0, 
                                          w_mu=0.0, w_ovlp=0.0, w_quad=0.0, w_df=1.0, reg_dip=0.0,
                                          aux_basis=None, return_fit=False, 
                                          constrain_dipole=False, constrain_quad=False):
    """
    Fits the orbital pair density to a minimal basis atom-centered spherically
    symmetric density AND exact L=1 representations of the valence shell.
    """
    atom_mols = []
    for i in range(mol.natm):
        atm_mol = gto.Mole()
        atm_mol.atom = [[mol.atom_symbol(i), mol.atom_coord(i)]] 
        atm_mol.unit = 'Bohr'
        atm_mol.basis = min_basis
        atm_mol.spin = 0
        atm_mol.charge = mol.atom_charge(i)
        atm_mol.build()
        atom_mols.append(atm_mol)
    
    min_mol = atom_mols[0]
    for i in range(1, len(atom_mols)):
        min_mol = min_mol + atom_mols[i]
    
    R_vdw = radii.VDW[min_mol.atom_charges()]
    Np = mol.nao
    nshl_parent = mol.nbas
    
    # 1e integrals
    S_parent = mol.intor('int1e_ovlp')
    mu_parent = -mol.intor('int1e_r')
    rr_parent = mol.intor('int1e_rr') 

    # =========================================================================
    # Construct Spherically Averaged & L=1 Basis Pairs
    # =========================================================================
    fitted_pairs = []
    S_atomic = []  
    mu_atomic_shifted = [] 
    r_atomic = []
    rr_atomic = []
    r2_atomic = []  

    for A in range(mol.natm):
        mol_A = atom_mols[A]
        R_A = mol.atom_coord(A) 
        
        mol_A.set_common_orig(R_A)
        S_atomic.append(mol_A.intor('int1e_ovlp'))
        mu_atomic_shifted.append(-mol_A.intor('int1e_r'))
        r_atomic.append(mol_A.intor('int1e_r'))
        rr_atomic.append(mol_A.intor('int1e_rr'))
        r2_atomic.append(mol_A.intor('int1e_r2'))
        mol_A.set_common_orig((0.0, 0.0, 0.0))

        # Identify valence shells
        ao_labels = mol_A.ao_labels(fmt=False) 
        nl_to_aos = {}
        for idx, label in enumerate(ao_labels):
            nl = label[2]  
            if nl not in nl_to_aos:
                nl_to_aos[nl] = []
            nl_to_aos[nl].append(idx)
            
        Z = mol.atom_charge(A)
        # Assumes valence_l_values is defined globally or passed in
        if Z - 1 < len(valence_l_values):
            allowed_l_vals = valence_l_values[Z - 1]
        else:
            allowed_l_vals = [0, 1, 2, 3] 
            
        l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        allowed_l_chars = [l_map.get(l, '') for l in allowed_l_vals]

        val_nls = set()
        for l_char in allowed_l_chars:
            nls_with_l = [nl for nl in nl_to_aos.keys() if nl.endswith(l_char)]
            if nls_with_l:
                nls_with_l.sort(key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0)
                val_nls.add(nls_with_l[-1])

        n_shells = mol_A.nbas
        ao_loc = mol_A.ao_loc_nr()
        
        # Track valence shell IDs
        val_shells_idx = []
        for i in range(n_shells):
            l_i = mol_A.bas_angular(i)
            start_i = ao_loc[i]
            nl = ao_labels[start_i][2]
            
            if nl in val_nls:
                if l_i > 2:
                    raise ValueError(f"Valence shell '{nl}' on atom {A} contains l > 2, which is unsupported for L=1 hard-coding.")
                val_shells_idx.append(i)

        # 1. Spherically Averaged (L=0) Pairs (ForAll Shells)
        for i in range(n_shells):
            l_i = mol_A.bas_angular(i)
            for j in range(i + 1):
                l_j = mol_A.bas_angular(j)
                if l_i == l_j:
                    start_i, end_i = ao_loc[i], ao_loc[i+1]
                    start_j, end_j = ao_loc[j], ao_loc[j+1]
                    # Format is now (a, b, coeff)
                    ab_pairs = [(start_i + m, start_j + m, 1.0) for m in range(end_i - start_i)]
                    fitted_pairs.append({'atom': A, 'ab_pairs': ab_pairs, 'L': 0}) # Tagged as L=0

        # 2. Add L=1 representation of Valence-Valence shell mixing
        for i_idx in range(len(val_shells_idx)):
            i = val_shells_idx[i_idx]
            l_i = mol_A.bas_angular(i)
            start_i = ao_loc[i]
            
            for j_idx in range(i_idx): # j_idx < i_idx intrinsically guarantees l_i != l_j for valence
                j = val_shells_idx[j_idx]
                l_j = mol_A.bas_angular(j)
                start_j = ao_loc[j]
                
                if (l_i, l_j) in CG_L1:
                    cg_dict = CG_L1[(l_i, l_j)]
                    for M in [0, 1, 2]: # Generate 3 density basis functions for M=-1, 0, 1
                        ab_pairs = []
                        for m_i, m_j, coeff in cg_dict[M]:
                            ab_pairs.append((start_i + m_i, start_j + m_j, coeff))
                        fitted_pairs.append({'atom': A, 'ab_pairs': ab_pairs, 'L': 1}) # Tagged as L=1

    M_pairs = len(fitted_pairs)

    # =========================================================================
    # Overlap Metric Preparation
    # =========================================================================
    if w_ovlp > 0.0 and aux_basis is not None:
        #print(f"Preparing Overlap (L2) metric using aux_basis: {aux_basis}...")
        
        auxmol = df.make_auxmol(mol, aux_basis)
        N_aux = auxmol.nao
        nshl_aux = auxmol.nbas
        
        S_aux = auxmol.intor('int1e_ovlp')
        S_aux_inv = np.linalg.pinv(S_aux, rcond=1e-8)
        
        mol_parent_aux = mol + auxmol
        shls_slice = (0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_aux)
        ovlp3c_parent = mol_parent_aux.intor('int3c1e', shls_slice=shls_slice)
        
        ovlp3c_A_list = []
        for A in range(mol.natm):
            mol_A = atom_mols[A]
            mol_A_aux = mol_A + auxmol
            shls_slice = (0, mol_A.nbas, 0, mol_A.nbas, mol_A.nbas, mol_A.nbas + nshl_aux)
            ovlp3c_A_list.append(mol_A_aux.intor('int3c1e', shls_slice=shls_slice))
            
        ovlp3c_I = np.zeros((M_pairs, N_aux))
        for I, pair_I in enumerate(fitted_pairs):
            A_idx = pair_I['atom']
            for a, b, c_ab in pair_I['ab_pairs']:
                ovlp3c_I[I, :] += c_ab * ovlp3c_A_list[A_idx][a, b, :]
            
        ovlp3c_I_proj = ovlp3c_I @ S_aux_inv

    # =========================================================================
    # Pre-compute Global Contributions (Dipole & Quadrupole)
    # =========================================================================
    V_pairs = np.zeros((M_pairs, 3))
    V_quad_pairs = np.zeros((M_pairs, 6))

    for I, pair_I in enumerate(fitted_pairs):
        A_idx = pair_I['atom']
        R_A = mol.atom_coord(A_idx)
        ab_list = pair_I['ab_pairs']
        
        for x in range(3):
            sum_mu = sum(c_ab * mu_atomic_shifted[A_idx][x, a, b] for a, b, c_ab in ab_list)
            sum_S = sum(c_ab * S_atomic[A_idx][a, b] for a, b, c_ab in ab_list)
            V_pairs[I, x] = sum_mu - sum_S * R_A[x]
            
        for q, (i, j, f, w) in enumerate(quad_map):
            sum_rr = sum(c_ab * rr_atomic[A_idx][f, a, b] for a, b, c_ab in ab_list)
            sum_ri = sum(c_ab * r_atomic[A_idx][i, a, b] for a, b, c_ab in ab_list)
            sum_rj = sum(c_ab * r_atomic[A_idx][j, a, b] for a, b, c_ab in ab_list)
            sum_S = sum(c_ab * S_atomic[A_idx][a, b] for a, b, c_ab in ab_list)
            V_quad_pairs[I, q] = sum_rr + R_A[j]*sum_ri + R_A[i]*sum_rj + R_A[i]*R_A[j]*sum_S

    n_constraints = 1
    if constrain_dipole: n_constraints += 3
    if constrain_quad: n_constraints += 6

    A_mat = np.zeros((M_pairs, M_pairs))
    b_tensor = np.zeros((Np, Np, M_pairs))
    C_mat = np.zeros((M_pairs, n_constraints))
    A_mat_df = np.zeros((M_pairs, M_pairs))
    b_tensor_df = np.zeros((Np, Np, M_pairs))

    #print(f"Building KKT Matrices for {M_pairs} basis components...")

    for I, pair_I in enumerate(fitted_pairs):
        A_idx = pair_I['atom']
        ab_list = pair_I['ab_pairs']
        
        for (a, b, c_ab) in ab_list:
            C_mat[I, 0] += c_ab * S_atomic[A_idx][a, b]
            
        if reg_S > 0.0:
            b_tensor[:, :, I] += reg_S * C_mat[I, 0] * S_parent
            
        if w_mu > 0.0:
            for x in range(3):
                b_tensor[:, :, I] += w_mu * V_pairs[I, x] * mu_parent[x]
                
        if w_quad > 0.0:
            for q, (i, j, f, w) in enumerate(quad_map):
                b_tensor[:, :, I] += w_quad * w * V_quad_pairs[I, q] * rr_parent[f]
            
        for J, pair_J in enumerate(fitted_pairs):
            B_idx = pair_J['atom']
            cd_list = pair_J['ab_pairs']
            
            if reg_S > 0.0:
                sum_S_B = sum(c_cd * S_atomic[B_idx][c, d] for c, d, c_cd in cd_list)
                A_mat[I, J] += reg_S * C_mat[I, 0] * sum_S_B
                
            if w_mu > 0.0:
                A_mat[I, J] += w_mu * np.dot(V_pairs[I], V_pairs[J])
                
            if w_quad > 0.0:
                quad_penalty = sum(w * V_quad_pairs[I, q] * V_quad_pairs[J, q] for q, (i, j, f, w) in enumerate(quad_map))
                A_mat[I, J] += w_quad * quad_penalty
            
            if reg_r2 > 0.0 and A_idx == B_idx:
                sum_r2_A = sum(c_ab * r2_atomic[A_idx][a, b] for a, b, c_ab in ab_list)
                sum_r2_B = sum(c_cd * r2_atomic[B_idx][c, d] for c, d, c_cd in cd_list)
                A_mat[I, J] += (reg_r2/(R_vdw[A_idx]**4)) * sum_r2_A * sum_r2_B


    # =========================================================================
    # Optimized Targeted ERI Integral Evaluation via Vectorized Contractions
    # =========================================================================
    #print("Evaluating targeted ERI integrals (Optimized Vectorized)...")
    
    ao2shl_mols = []
    ao_loc_mols = []
    for A in range(mol.natm):
        ao_loc = atom_mols[A].ao_loc_nr()
        ao2shl = {ao: shl for shl in range(atom_mols[A].nbas) 
                          for ao in range(ao_loc[shl], ao_loc[shl+1])}
        ao2shl_mols.append(ao2shl)
        ao_loc_mols.append(ao_loc)

    # 1. Build Shell-Pair Mappings for Fast Einsum Operations
    shell_pairs = {}
    for I, pair_I in enumerate(fitted_pairs):
        A = pair_I['atom']
        for a, b, c_ab in pair_I['ab_pairs']:
            shl_a = ao2shl_mols[A][a]
            shl_b = ao2shl_mols[A][b]
            a_loc = a - ao_loc_mols[A][shl_a]
            b_loc = b - ao_loc_mols[A][shl_b]
            
            sp_key = (A, shl_a, shl_b)
            if sp_key not in shell_pairs:
                na = ao_loc_mols[A][shl_a+1] - ao_loc_mols[A][shl_a]
                nb = ao_loc_mols[A][shl_b+1] - ao_loc_mols[A][shl_b]
                shell_pairs[sp_key] = {'na': na, 'nb': nb, 'tasks': []}
            shell_pairs[sp_key]['tasks'].append((I, a_loc, b_loc, c_ab))

    for sp_key, sp_data in shell_pairs.items():
        na, nb = sp_data['na'], sp_data['nb']
        I_set = sorted(list(set([t[0] for t in sp_data['tasks']])))
        I_map = {I: idx for idx, I in enumerate(I_set)}
        
        C_I = np.zeros((len(I_set), na, nb))
        for (I, a_loc, b_loc, c_ab) in sp_data['tasks']:
            C_I[I_map[I], a_loc, b_loc] += c_ab
            
        sp_data['I_set'] = I_set
        sp_data['C_I'] = C_I

    # 2. Pre-combine molecules entirely to avoid overhead re-compilation
    mol_comb = mol + min_mol
    min_mol_shl_offset = [0]
    for A in range(mol.natm - 1):
        min_mol_shl_offset.append(min_mol_shl_offset[-1] + atom_mols[A].nbas)

    # 3. Process Integrals via Tensor Blocks
    sp_keys = list(shell_pairs.keys())
    for i_idx, sp_P in enumerate(sp_keys):
        A, shl_a, shl_b = sp_P
        data_P = shell_pairs[sp_P]
        I_set, C_I = data_P['I_set'], data_P['C_I']
        
        shl_a_min = min_mol_shl_offset[A] + shl_a
        shl_b_min = min_mol_shl_offset[A] + shl_b
        
        # Parent-Aux integrals for b_tensor 
        shls_slice_b = (0, nshl_parent, 0, nshl_parent, 
                        nshl_parent + shl_a_min, nshl_parent + shl_a_min + 1, 
                        nshl_parent + shl_b_min, nshl_parent + shl_b_min + 1)
        eri_b = mol_comb.intor('int2e', shls_slice=shls_slice_b)
        
        # Vectorized b_tensor update
        b_upd = np.einsum('pqab,iab->pqi', eri_b, C_I, optimize=True)
        for idx, I in enumerate(I_set):
            val = w_df * b_upd[:, :, idx]
            b_tensor[:, :, I] += val
            b_tensor_df[:, :, I] += val
            
        # Aux-Aux integrals for A_mat (exploiting lower triangular symmetry)
        for j_idx in range(i_idx + 1):
            sp_Q = sp_keys[j_idx]
            B, shl_c, shl_d = sp_Q
            data_Q = shell_pairs[sp_Q]
            J_set, C_J = data_Q['I_set'], data_Q['C_I']
            
            shl_c_min = min_mol_shl_offset[B] + shl_c
            shl_d_min = min_mol_shl_offset[B] + shl_d
            
            shls_slice_A = (shl_a_min, shl_a_min+1, shl_b_min, shl_b_min+1, 
                            shl_c_min, shl_c_min+1, shl_d_min, shl_d_min+1)
            eri_A = min_mol.intor('int2e', shls_slice=shls_slice_A)
            
            # Vectorized A_mat update
            A_upd = np.einsum('iab,jcd,abcd->ij', C_I, C_J, eri_A, optimize=True)
            
            for idx_i, I in enumerate(I_set):
                for idx_j, J in enumerate(J_set):
                    val = w_df * A_upd[idx_i, idx_j]
                    A_mat[I, J] += val
                    A_mat_df[I, J] += val
                    if i_idx != j_idx:
                        A_mat[J, I] += val
                        A_mat_df[J, I] += val


    if w_ovlp > 0.0 and aux_basis is not None:
        #print("Adding Auxiliary Overlap Metric Penalty...")
        A_mat += w_ovlp * (ovlp3c_I_proj @ ovlp3c_I.T)
        ovlp3c_parent_flat = ovlp3c_parent.reshape(-1, N_aux)
        b_ovlp_flat = ovlp3c_parent_flat @ ovlp3c_I_proj.T
        b_tensor += w_ovlp * b_ovlp_flat.reshape(Np, Np, M_pairs)

    # Populate Constraints
    d_tensor = np.zeros((Np, Np, n_constraints))
    d_tensor[:, :, 0] = S_parent
    col_idx = 1
    
    if constrain_dipole:
        C_mat[:, col_idx:col_idx+3] = V_pairs
        for x in range(3):
            d_tensor[:, :, col_idx+x] = mu_parent[x]
        col_idx += 3
        
    if constrain_quad:
        for q, (i, j, f, w) in enumerate(quad_map):
            C_mat[:, col_idx+q] = V_quad_pairs[:, q]
            d_tensor[:, :, col_idx+q] = rr_parent[f]
        col_idx += 6
    
    # -------------------------------------------------------------------------
    # Apply Standard and Dipole-Specific (L=1) Regularization
    # -------------------------------------------------------------------------
    if reg > 0.0:
        A_mat += reg * np.eye(M_pairs)
        
    if reg_dip > 0.0:
        for I, pair_I in enumerate(fitted_pairs):
            if pair_I.get('L') == 1:
                A_mat[I, I] += reg_dip
    
    #print("Solving KKT system...")
    X_I_nm = solve_kkt_tensor_reshaped(A_mat, b_tensor, C_mat, d_tensor)

    Q_A = np.zeros((mol.natm, Np, Np))
    mu_A = np.zeros((mol.natm, 3, Np, Np))

    for I, pair_I in enumerate(fitted_pairs):
        A_idx = pair_I['atom']
        ab_list = pair_I['ab_pairs']
        
        Q_A[A_idx] -= X_I_nm[I, :, :] * C_mat[I, 0]

        for x in range(3):
            sum_mu = sum(c_ab * mu_atomic_shifted[A_idx][x, a, b] for a, b, c_ab in ab_list)
            mu_A[A_idx, x] += X_I_nm[I, :, :] * sum_mu

    #print("Done!")
    if return_fit:
        return Q_A, mu_A, A_mat_df, b_tensor_df, X_I_nm
    else:
        return Q_A, mu_A  
    




def getDMFitChargeOperatorDipoleSphericalNew(mol, min_basis='minao', reg=1e-12, reg_r2=0.0, reg_S=0.0, 
                                          w_mu=0.0, w_ovlp=0.0, w_quad=0.0, w_df=1.0, reg_dip=0.0,
                                          aux_basis=None, return_fit=False, 
                                          constrain_dipole=False, constrain_quad=False):
    """
    Fits the orbital pair density to a minimal basis atom-centered spherically
    symmetric density AND exact L=1 representations of the valence shell.
    (Optimized with vectorized NumPy KKT operations)
    """
    atom_mols = []
    for i in range(mol.natm):
        atm_mol = gto.Mole()
        atm_mol.atom = [[mol.atom_symbol(i), mol.atom_coord(i)]] 
        atm_mol.unit = 'Bohr'
        atm_mol.basis = min_basis
        atm_mol.spin = 0
        atm_mol.charge = mol.atom_charge(i)
        atm_mol.build()
        atom_mols.append(atm_mol)
    
    min_mol = atom_mols[0]
    for i in range(1, len(atom_mols)):
        min_mol = min_mol + atom_mols[i]
    
    R_vdw = radii.VDW[min_mol.atom_charges()]
    Np = mol.nao
    nshl_parent = mol.nbas
    
    # 1e integrals
    S_parent = mol.intor('int1e_ovlp')
    mu_parent = -mol.intor('int1e_r')
    rr_parent = mol.intor('int1e_rr') 

    # =========================================================================
    # Construct Spherically Averaged & L=1 Basis Pairs
    # =========================================================================
    fitted_pairs = []
    S_atomic = []  
    mu_atomic_shifted = [] 
    r_atomic = []
    rr_atomic = []
    r2_atomic = []  

    for A in range(mol.natm):
        mol_A = atom_mols[A]
        R_A = mol.atom_coord(A) 
        
        mol_A.set_common_orig(R_A)
        S_atomic.append(mol_A.intor('int1e_ovlp'))
        mu_atomic_shifted.append(-mol_A.intor('int1e_r'))
        r_atomic.append(mol_A.intor('int1e_r'))
        rr_atomic.append(mol_A.intor('int1e_rr'))
        r2_atomic.append(mol_A.intor('int1e_r2'))
        mol_A.set_common_orig((0.0, 0.0, 0.0))

        # Identify valence shells
        ao_labels = mol_A.ao_labels(fmt=False) 
        nl_to_aos = {}
        for idx, label in enumerate(ao_labels):
            nl = label[2]  
            if nl not in nl_to_aos:
                nl_to_aos[nl] = []
            nl_to_aos[nl].append(idx)
            
        Z = mol.atom_charge(A)
        if Z - 1 < len(valence_l_values):
            allowed_l_vals = valence_l_values[Z - 1]
        else:
            allowed_l_vals = [0, 1, 2, 3] 
            
        l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        allowed_l_chars = [l_map.get(l, '') for l in allowed_l_vals]

        val_nls = set()
        for l_char in allowed_l_chars:
            nls_with_l = [nl for nl in nl_to_aos.keys() if nl.endswith(l_char)]
            if nls_with_l:
                nls_with_l.sort(key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0)
                val_nls.add(nls_with_l[-1])

        n_shells = mol_A.nbas
        ao_loc = mol_A.ao_loc_nr()
        
        # Track valence shell IDs
        val_shells_idx = []
        for i in range(n_shells):
            l_i = mol_A.bas_angular(i)
            start_i = ao_loc[i]
            nl = ao_labels[start_i][2]
            
            if nl in val_nls:
                if l_i > 2:
                    raise ValueError(f"Valence shell '{nl}' on atom {A} contains l > 2, unsupported for L=1.")
                val_shells_idx.append(i)

        # 1. Spherically Averaged (L=0) Pairs (ForAll Shells)
        for i in range(n_shells):
            l_i = mol_A.bas_angular(i)
            for j in range(i + 1):
                l_j = mol_A.bas_angular(j)
                if l_i == l_j:
                    start_i, end_i = ao_loc[i], ao_loc[i+1]
                    start_j, end_j = ao_loc[j], ao_loc[j+1]
                    ab_pairs = [(start_i + m, start_j + m, 1.0) for m in range(end_i - start_i)]
                    fitted_pairs.append({'atom': A, 'ab_pairs': ab_pairs, 'L': 0})

        # 2. Add L=1 representation of Valence-Valence shell mixing
        for i_idx in range(len(val_shells_idx)):
            i = val_shells_idx[i_idx]
            l_i = mol_A.bas_angular(i)
            start_i = ao_loc[i]
            
            for j_idx in range(i_idx): 
                j = val_shells_idx[j_idx]
                l_j = mol_A.bas_angular(j)
                start_j = ao_loc[j]
                
                if (l_i, l_j) in CG_L1:
                    cg_dict = CG_L1[(l_i, l_j)]
                    for M in [0, 1, 2]: 
                        ab_pairs = []
                        for m_i, m_j, coeff in cg_dict[M]:
                            ab_pairs.append((start_i + m_i, start_j + m_j, coeff))
                        fitted_pairs.append({'atom': A, 'ab_pairs': ab_pairs, 'L': 1})

    M_pairs = len(fitted_pairs)

    # =========================================================================
    # Overlap Metric Preparation
    # =========================================================================
    if w_ovlp > 0.0 and aux_basis is not None:
        auxmol = df.make_auxmol(mol, aux_basis)
        N_aux = auxmol.nao
        nshl_aux = auxmol.nbas
        
        S_aux = auxmol.intor('int1e_ovlp')
        S_aux_inv = np.linalg.pinv(S_aux, rcond=1e-8)
        
        mol_parent_aux = mol + auxmol
        shls_slice = (0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_aux)
        ovlp3c_parent = mol_parent_aux.intor('int3c1e', shls_slice=shls_slice)
        
        ovlp3c_A_list = []
        for A in range(mol.natm):
            mol_A = atom_mols[A]
            mol_A_aux = mol_A + auxmol
            shls_slice = (0, mol_A.nbas, 0, mol_A.nbas, mol_A.nbas, mol_A.nbas + nshl_aux)
            ovlp3c_A_list.append(mol_A_aux.intor('int3c1e', shls_slice=shls_slice))
            
        ovlp3c_I = np.zeros((M_pairs, N_aux))
        for I, pair_I in enumerate(fitted_pairs):
            A_idx = pair_I['atom']
            for a, b, c_ab in pair_I['ab_pairs']:
                ovlp3c_I[I, :] += c_ab * ovlp3c_A_list[A_idx][a, b, :]
            
        ovlp3c_I_proj = ovlp3c_I @ S_aux_inv

    # =========================================================================
    # Pre-compute Global Contributions (Dipole & Quadrupole)
    # =========================================================================
    V_pairs = np.zeros((M_pairs, 3))
    V_quad_pairs = np.zeros((M_pairs, 6))
    
    # Pre-allocate arrays for vectorization
    C_vec = np.zeros(M_pairs)
    sum_r2_vec = np.zeros(M_pairs)
    mu_vec_sums = np.zeros((M_pairs, 3))
    atom_idx_vec = np.zeros(M_pairs, dtype=int)

    for I, pair_I in enumerate(fitted_pairs):
        A_idx = pair_I['atom']
        R_A = mol.atom_coord(A_idx)
        ab_list = pair_I['ab_pairs']
        atom_idx_vec[I] = A_idx
        
        # Scalar overlaps & r2
        sum_S = sum(c_ab * S_atomic[A_idx][a, b] for a, b, c_ab in ab_list)
        C_vec[I] = sum_S
        
        if reg_r2 > 0.0:
            sum_r2_vec[I] = sum(c_ab * r2_atomic[A_idx][a, b] for a, b, c_ab in ab_list)
        
        # Dipole contributions
        for x in range(3):
            sum_mu = sum(c_ab * mu_atomic_shifted[A_idx][x, a, b] for a, b, c_ab in ab_list)
            mu_vec_sums[I, x] = sum_mu
            V_pairs[I, x] = sum_mu - sum_S * R_A[x]
            
        # Quadrupole contributions
        if w_quad > 0.0 or constrain_quad:
            for q, (i, j, f, w) in enumerate(quad_map):
                sum_rr = sum(c_ab * rr_atomic[A_idx][f, a, b] for a, b, c_ab in ab_list)
                sum_ri = sum(c_ab * r_atomic[A_idx][i, a, b] for a, b, c_ab in ab_list)
                sum_rj = sum(c_ab * r_atomic[A_idx][j, a, b] for a, b, c_ab in ab_list)
                V_quad_pairs[I, q] = sum_rr + R_A[j]*sum_ri + R_A[i]*sum_rj + R_A[i]*R_A[j]*sum_S

    n_constraints = 1
    if constrain_dipole: n_constraints += 3
    if constrain_quad: n_constraints += 6

    A_mat = np.zeros((M_pairs, M_pairs))
    b_tensor = np.zeros((Np, Np, M_pairs))
    C_mat = np.zeros((M_pairs, n_constraints))
    A_mat_df = np.zeros((M_pairs, M_pairs))
    b_tensor_df = np.zeros((Np, Np, M_pairs))

    # =========================================================================
    # FAST Vectorized KKT Construction
    # =========================================================================
    C_mat[:, 0] = C_vec

    if reg_S > 0.0:
        b_tensor += reg_S * np.einsum('ij,I->ijI', S_parent, C_vec, optimize=True)
        A_mat += reg_S * np.outer(C_vec, C_vec)
        
    if w_mu > 0.0:
        b_tensor += w_mu * np.einsum('Ix,xij->ijI', V_pairs, mu_parent, optimize=True)
        A_mat += w_mu * (V_pairs @ V_pairs.T)
        
    if w_quad > 0.0:
        w_arr = np.array([q[3] for q in quad_map])
        f_arr = [q[2] for q in quad_map]
        rr_parent_sub = rr_parent[f_arr]  # Shape: (n_q, Np, Np)
        weighted_V_quad = w_quad * w_arr * V_quad_pairs # Shape: (M_pairs, n_q)
        
        b_tensor += np.einsum('Iq,qij->ijI', weighted_V_quad, rr_parent_sub, optimize=True)
        A_mat += weighted_V_quad @ V_quad_pairs.T
        
    if reg_r2 > 0.0:
        atom_mask = (atom_idx_vec[:, None] == atom_idx_vec[None, :])
        r2_scaled = sum_r2_vec / (R_vdw[atom_idx_vec]**2) 
        A_mat += reg_r2 * atom_mask * np.outer(r2_scaled, r2_scaled)


    # =========================================================================
    # Optimized Targeted ERI Integral Evaluation via Vectorized Contractions
    # =========================================================================
    ao2shl_mols = []
    ao_loc_mols = []
    for A in range(mol.natm):
        ao_loc = atom_mols[A].ao_loc_nr()
        ao2shl = {ao: shl for shl in range(atom_mols[A].nbas) 
                          for ao in range(ao_loc[shl], ao_loc[shl+1])}
        ao2shl_mols.append(ao2shl)
        ao_loc_mols.append(ao_loc)

    # 1. Build Shell-Pair Mappings
    shell_pairs = {}
    for I, pair_I in enumerate(fitted_pairs):
        A = pair_I['atom']
        for a, b, c_ab in pair_I['ab_pairs']:
            shl_a = ao2shl_mols[A][a]
            shl_b = ao2shl_mols[A][b]
            a_loc = a - ao_loc_mols[A][shl_a]
            b_loc = b - ao_loc_mols[A][shl_b]
            
            sp_key = (A, shl_a, shl_b)
            if sp_key not in shell_pairs:
                na = ao_loc_mols[A][shl_a+1] - ao_loc_mols[A][shl_a]
                nb = ao_loc_mols[A][shl_b+1] - ao_loc_mols[A][shl_b]
                shell_pairs[sp_key] = {'na': na, 'nb': nb, 'tasks': []}
            shell_pairs[sp_key]['tasks'].append((I, a_loc, b_loc, c_ab))

    for sp_key, sp_data in shell_pairs.items():
        na, nb = sp_data['na'], sp_data['nb']
        I_set = sorted(list(set([t[0] for t in sp_data['tasks']])))
        I_map = {I: idx for idx, I in enumerate(I_set)}
        
        C_I = np.zeros((len(I_set), na, nb))
        for (I, a_loc, b_loc, c_ab) in sp_data['tasks']:
            C_I[I_map[I], a_loc, b_loc] += c_ab
            
        sp_data['I_set'] = I_set
        sp_data['C_I'] = C_I

    # 2. Pre-combine molecules
    mol_comb = mol + min_mol
    min_mol_shl_offset = [0]
    for A in range(mol.natm - 1):
        min_mol_shl_offset.append(min_mol_shl_offset[-1] + atom_mols[A].nbas)

    # 3. Process Integrals via Tensor Blocks
    sp_keys = list(shell_pairs.keys())
    for i_idx, sp_P in enumerate(sp_keys):
        A, shl_a, shl_b = sp_P
        data_P = shell_pairs[sp_P]
        I_set, C_I = data_P['I_set'], data_P['C_I']
        
        shl_a_min = min_mol_shl_offset[A] + shl_a
        shl_b_min = min_mol_shl_offset[A] + shl_b
        
        # Parent-Aux integrals
        shls_slice_b = (0, nshl_parent, 0, nshl_parent, 
                        nshl_parent + shl_a_min, nshl_parent + shl_a_min + 1, 
                        nshl_parent + shl_b_min, nshl_parent + shl_b_min + 1)
        eri_b = mol_comb.intor('int2e', shls_slice=shls_slice_b)
        
        # FAST Update: Block assignment into b_tensor
        b_upd = np.einsum('pqab,iab->pqi', eri_b, C_I, optimize=True)
        val_b = w_df * b_upd
        b_tensor[:, :, I_set] += val_b
        b_tensor_df[:, :, I_set] += val_b
            
        # Aux-Aux integrals
        for j_idx in range(i_idx + 1):
            sp_Q = sp_keys[j_idx]
            B, shl_c, shl_d = sp_Q
            data_Q = shell_pairs[sp_Q]
            J_set, C_J = data_Q['I_set'], data_Q['C_I']
            
            shl_c_min = min_mol_shl_offset[B] + shl_c
            shl_d_min = min_mol_shl_offset[B] + shl_d
            
            shls_slice_A = (shl_a_min, shl_a_min+1, shl_b_min, shl_b_min+1, 
                            shl_c_min, shl_c_min+1, shl_d_min, shl_d_min+1)
            eri_A = min_mol.intor('int2e', shls_slice=shls_slice_A)
            
            # FAST Update: Block assignment into A_mat
            A_upd = np.einsum('iab,jcd,abcd->ij', C_I, C_J, eri_A, optimize=True)
            val_A = w_df * A_upd
            
            A_mat[np.ix_(I_set, J_set)] += val_A
            A_mat_df[np.ix_(I_set, J_set)] += val_A
            
            if i_idx != j_idx:
                A_mat[np.ix_(J_set, I_set)] += val_A.T
                A_mat_df[np.ix_(J_set, I_set)] += val_A.T


    if w_ovlp > 0.0 and aux_basis is not None:
        A_mat += w_ovlp * (ovlp3c_I_proj @ ovlp3c_I.T)
        ovlp3c_parent_flat = ovlp3c_parent.reshape(-1, N_aux)
        b_ovlp_flat = ovlp3c_parent_flat @ ovlp3c_I_proj.T
        b_tensor += w_ovlp * b_ovlp_flat.reshape(Np, Np, M_pairs)

    # Populate Constraints
    d_tensor = np.zeros((Np, Np, n_constraints))
    d_tensor[:, :, 0] = S_parent
    col_idx = 1
    
    if constrain_dipole:
        C_mat[:, col_idx:col_idx+3] = V_pairs
        for x in range(3):
            d_tensor[:, :, col_idx+x] = mu_parent[x]
        col_idx += 3
        
    if constrain_quad:
        for q, (i, j, f, w) in enumerate(quad_map):
            C_mat[:, col_idx+q] = V_quad_pairs[:, q]
            d_tensor[:, :, col_idx+q] = rr_parent[f]
        col_idx += 6
    
    # -------------------------------------------------------------------------
    # Apply Standard and Dipole-Specific (L=1) Regularization
    # -------------------------------------------------------------------------
    if reg > 0.0:
        A_mat += reg * np.eye(M_pairs)
        
    if reg_dip > 0.0:
        for I, pair_I in enumerate(fitted_pairs):
            if pair_I.get('L') == 1:
                A_mat[I, I] += reg_dip
    
    X_I_nm = solve_kkt_tensor_reshaped(A_mat, b_tensor, C_mat, d_tensor)

    # =========================================================================
    # FAST Vectorized Output Tensor Generation
    # =========================================================================
    Q_A = np.zeros((mol.natm, Np, Np))
    mu_A = np.zeros((mol.natm, 3, Np, Np))

    for A in range(mol.natm):
        I_mask = (atom_idx_vec == A)
        if not np.any(I_mask):
            continue
            
        X_A = X_I_nm[I_mask]           # Shape: (M_A, Np, Np)
        C_A = C_mat[I_mask, 0]         # Shape: (M_A)
        
        Q_A[A] = -np.einsum('Iij,I->ij', X_A, C_A, optimize=True)
        mu_A[A] = np.einsum('Iij,Ix->xij', X_A, mu_vec_sums[I_mask], optimize=True)

    if return_fit:
        return Q_A, mu_A, A_mat_df, b_tensor_df, X_I_nm
    else:
        return Q_A, mu_A
    



def getDMFitChargeOperatorDipoleSphericalNew2(mol, min_basis='minao', reg=1e-12, reg_r2=0.0, reg_S=0.0, 
                                          w_mu=0.0, w_ovlp=0.0, w_quad=0.0, w_df=1.0, reg_dip=0.0,
                                          aux_basis=None, return_fit=False, 
                                          constrain_dipole=False, constrain_quad=False):
    t0 = timer()

    # =========================================================================
    # FAST SETUP: Single Global Build
    # =========================================================================
    min_mol = gto.Mole()
    min_mol.atom = mol.atom 
    min_mol.unit = mol.unit
    min_mol.basis = min_basis
    min_mol.charge = mol.charge
    min_mol.spin = mol.spin
    min_mol.build(dump_input=False, parse_arg=False)

    Np = mol.nao
    nshl_parent = mol.nbas
    nshl_min = min_mol.nbas
    min_aoslices = min_mol.aoslice_by_atom()
    ao_loc = min_mol.ao_loc_nr()
    ao_labels = min_mol.ao_labels(fmt=False)
    
    mol_parent_min = mol + min_mol
    R_vdw = radii.VDW[min_mol.atom_charges()]

    # Global 1-electron integrals (No manual origin shifting needed)
    S_parent = mol.intor('int1e_ovlp')
    mu_parent = -mol.intor('int1e_r')
    rr_parent = mol.intor('int1e_rr') 

    S_min_global = min_mol.intor('int1e_ovlp')
    r_min_global = min_mol.intor('int1e_r')
    mu_min_global = -r_min_global
    rr_min_global = min_mol.intor('int1e_rr')
    r2_min_global = min_mol.intor('int1e_r2')

    # =========================================================================
    # Construct Spherically Averaged & L=1 Basis Pairs
    # =========================================================================
    fitted_pairs = []

    for A in range(mol.natm):
        shl0_A, shl1_A, ao0_A, ao1_A = min_aoslices[A]
        R_A = mol.atom_coord(A) 
        
        # Identify valence shells for atom A
        nl_to_aos = {}
        for global_idx in range(ao0_A, ao1_A):
            nl = ao_labels[global_idx][2]
            if nl not in nl_to_aos:
                nl_to_aos[nl] = []
            nl_to_aos[nl].append(global_idx)
            
        Z = mol.atom_charge(A)
        allowed_l_vals = valence_l_values[Z - 1] if Z - 1 < len(valence_l_values) else [0, 1, 2, 3] 
        l_map = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        allowed_l_chars = [l_map.get(l, '') for l in allowed_l_vals]

        val_nls = set()
        for l_char in allowed_l_chars:
            nls_with_l = [nl for nl in nl_to_aos.keys() if nl.endswith(l_char)]
            if nls_with_l:
                nls_with_l.sort(key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 0)
                val_nls.add(nls_with_l[-1])

        val_shells_global_idx = []
        for i_g in range(shl0_A, shl1_A):
            start_ao_g = ao_loc[i_g]
            nl = ao_labels[start_ao_g][2]
            if nl in val_nls:
                l_i = min_mol.bas_angular(i_g)
                if l_i > 2:
                    raise ValueError(f"Valence shell '{nl}' on atom {A} contains l > 2, unsupported for L=1.")
                val_shells_global_idx.append(i_g)

        # 1. Spherically Averaged (L=0) Pairs
        for i_g in range(shl0_A, shl1_A):
            l_i = min_mol.bas_angular(i_g)
            for j_g in range(shl0_A, i_g + 1):
                l_j = min_mol.bas_angular(j_g)
                if l_i == l_j:
                    start_i, end_i = ao_loc[i_g], ao_loc[i_g+1]
                    start_j, end_j = ao_loc[j_g], ao_loc[j_g+1]
                    
                    a_g = np.arange(start_i, end_i)
                    b_g = np.arange(start_j, end_j)
                    coeffs = np.ones(len(a_g))
                    
                    fitted_pairs.append({
                        'atom': A, 'L': 0,
                        'a_loc': a_g - ao0_A, 'b_loc': b_g - ao0_A,
                        'a_g': a_g, 'b_g': b_g, 'coeffs': coeffs
                    })

        # 2. Add L=1 representation of Valence-Valence shell mixing
        for i_idx in range(len(val_shells_global_idx)):
            i_g = val_shells_global_idx[i_idx]
            l_i = min_mol.bas_angular(i_g)
            start_i = ao_loc[i_g]
            
            for j_idx in range(i_idx): 
                j_g = val_shells_global_idx[j_idx]
                l_j = min_mol.bas_angular(j_g)
                start_j = ao_loc[j_g]
                
                if (l_i, l_j) in CG_L1:
                    cg_dict = CG_L1[(l_i, l_j)]
                    for M in [0, 1, 2]: 
                        a_g_list, b_g_list, c_list = [], [], []
                        for m_i, m_j, coeff in cg_dict[M]:
                            a_g_list.append(start_i + m_i)
                            b_g_list.append(start_j + m_j)
                            c_list.append(coeff)
                            
                        fitted_pairs.append({
                            'atom': A, 'L': 1,
                            'a_loc': np.array(a_g_list) - ao0_A,
                            'b_loc': np.array(b_g_list) - ao0_A,
                            'a_g': np.array(a_g_list), 'b_g': np.array(b_g_list),
                            'coeffs': np.array(c_list)
                        })

    M_pairs = len(fitted_pairs)

    # =========================================================================
    # Overlap Metric Preparation
    # =========================================================================
    if w_ovlp > 0.0 and aux_basis is not None:
        auxmol = df.make_auxmol(mol, aux_basis)
        N_aux, nshl_aux = auxmol.nao, auxmol.nbas
        
        S_aux_inv = np.linalg.pinv(auxmol.intor('int1e_ovlp'), rcond=1e-8)
        ovlp3c_parent = (mol + auxmol).intor('int3c1e', shls_slice=(0, nshl_parent, 0, nshl_parent, nshl_parent, nshl_parent + nshl_aux))
        ovlp3c_min_aux = (min_mol + auxmol).intor('int3c1e', shls_slice=(0, nshl_min, 0, nshl_min, nshl_min, nshl_min + nshl_aux))
        
        ovlp3c_I = np.zeros((M_pairs, N_aux))
        for I, pI in enumerate(fitted_pairs):
            ovlp3c_I[I] = np.sum(pI['coeffs'][:, None] * ovlp3c_min_aux[pI['a_g'], pI['b_g'], :], axis=0)
            
        ovlp3c_I_proj = ovlp3c_I @ S_aux_inv

    # =========================================================================
    # Pre-compute Global Contributions (Analytically Shifted)
    # =========================================================================
    V_pairs = np.zeros((M_pairs, 3))
    V_quad_pairs = np.zeros((M_pairs, 6))
    
    C_vec = np.zeros(M_pairs)
    sum_r2_vec = np.zeros(M_pairs)
    mu_vec_sums = np.zeros((M_pairs, 3))
    atom_idx_vec = np.zeros(M_pairs, dtype=int)

    for I, pI in enumerate(fitted_pairs):
        A_idx = pI['atom']
        R_A = mol.atom_coord(A_idx)
        a_g, b_g, c = pI['a_g'], pI['b_g'], pI['coeffs']
        atom_idx_vec[I] = A_idx
        
        sum_S = np.sum(c * S_min_global[a_g, b_g])
        C_vec[I] = sum_S
        
        # Analytically Shifted 1e Math 
        sum_mu_global = np.sum(c * mu_min_global[:, a_g, b_g], axis=1)
        mu_vec_sums[I] = sum_mu_global + sum_S * R_A  # Shifted mu
        V_pairs[I] = sum_mu_global                    # Exact cancellation 
            
        if w_quad > 0.0 or constrain_quad:
            for q, (i, j, f, w) in enumerate(quad_map):
                V_quad_pairs[I, q] = np.sum(c * rr_min_global[f, a_g, b_g]) # Exact cancellation
                
        if reg_r2 > 0.0:
            sum_r2_global = np.sum(c * r2_min_global[a_g, b_g])
            sum_r_global = np.sum(c * r_min_global[:, a_g, b_g], axis=1)
            sum_r2_vec[I] = sum_r2_global - 2.0 * np.dot(R_A, sum_r_global) + np.dot(R_A, R_A) * sum_S

    n_constraints = 1 + (3 if constrain_dipole else 0) + (6 if constrain_quad else 0)
    C_mat = np.zeros((M_pairs, n_constraints))
    b_tensor_df = np.zeros((Np, Np, M_pairs))
    A_mat_df = np.zeros((M_pairs, M_pairs))

    # =========================================================================
    # FAST Vectorized Atom-Block ERI Processing
    # =========================================================================
    pairs_by_atom = [[] for _ in range(mol.natm)]
    for I, pI in enumerate(fitted_pairs):
        pairs_by_atom[pI['atom']].append((I, pI))

    for A in range(mol.natm):
        if not pairs_by_atom[A]: continue
            
        shl0_A, shl1_A, ao0_A, ao1_A = min_aoslices[A]
        slice_parent_A = (0, nshl_parent, 0, nshl_parent, 
                          nshl_parent + shl0_A, nshl_parent + shl1_A, 
                          nshl_parent + shl0_A, nshl_parent + shl1_A)
        eri_parent_A = mol_parent_min.intor('int2e', shls_slice=slice_parent_A)
        
        for I, pI in pairs_by_atom[A]:
            a_loc, b_loc, c_I = pI['a_loc'], pI['b_loc'], pI['coeffs']
            b_tensor_df[:, :, I] = np.sum(c_I * eri_parent_A[:, :, a_loc, b_loc], axis=-1)

        for B in range(A, mol.natm):
            if not pairs_by_atom[B]: continue
                
            shl0_B, shl1_B, ao0_B, ao1_B = min_aoslices[B]
            slice_AB = (shl0_A, shl1_A, shl0_A, shl1_A, shl0_B, shl1_B, shl0_B, shl1_B)
            eri_A_B = min_mol.intor('int2e', shls_slice=slice_AB)
            
            for I, pI in pairs_by_atom[A]:
                a_loc, b_loc, c_I = pI['a_loc'], pI['b_loc'], pI['coeffs']
                for J, pJ in pairs_by_atom[B]:
                    c_loc, d_loc, c_J = pJ['a_loc'], pJ['b_loc'], pJ['coeffs']
                    
                    val = np.sum(np.outer(c_I, c_J) * eri_A_B[a_loc, b_loc][:, c_loc, d_loc])
                    A_mat_df[I, J] = val
                    if A != B: 
                        A_mat_df[J, I] = val

    # =========================================================================
    # Apply Weights, Metrics, and Regularization
    # =========================================================================
    b_tensor = w_df * b_tensor_df.copy()
    A_mat = w_df * A_mat_df.copy()
    C_mat[:, 0] = C_vec

    if reg_S > 0.0:
        b_tensor += reg_S * np.einsum('ij,I->ijI', S_parent, C_vec, optimize=True)
        A_mat += reg_S * np.outer(C_vec, C_vec)
        
    if w_mu > 0.0:
        b_tensor += w_mu * np.einsum('Ix,xij->ijI', V_pairs, mu_parent, optimize=True)
        A_mat += w_mu * (V_pairs @ V_pairs.T)
        
    if w_quad > 0.0:
        w_arr = np.array([q[3] for q in quad_map])
        f_arr = [q[2] for q in quad_map]
        weighted_V_quad = w_quad * w_arr * V_quad_pairs 
        b_tensor += np.einsum('Iq,qij->ijI', weighted_V_quad, rr_parent[f_arr], optimize=True)
        A_mat += weighted_V_quad @ V_quad_pairs.T
        
    if reg_r2 > 0.0:
        atom_mask = (atom_idx_vec[:, None] == atom_idx_vec[None, :])
        r2_scaled = sum_r2_vec / (R_vdw[atom_idx_vec]**2) 
        A_mat += reg_r2 * atom_mask * np.outer(r2_scaled, r2_scaled)

    if w_ovlp > 0.0 and aux_basis is not None:
        A_mat += w_ovlp * (ovlp3c_I_proj @ ovlp3c_I.T)
        b_ovlp_flat = ovlp3c_parent.reshape(-1, N_aux) @ ovlp3c_I_proj.T
        b_tensor += w_ovlp * b_ovlp_flat.reshape(Np, Np, M_pairs)

    # Populate Constraints
    d_tensor = np.zeros((Np, Np, n_constraints))
    d_tensor[:, :, 0] = S_parent
    col_idx = 1
    
    if constrain_dipole:
        C_mat[:, col_idx:col_idx+3] = V_pairs
        for x in range(3): d_tensor[:, :, col_idx+x] = mu_parent[x]
        col_idx += 3
        
    if constrain_quad:
        for q, (i, j, f, w) in enumerate(quad_map):
            C_mat[:, col_idx+q] = V_quad_pairs[:, q]
            d_tensor[:, :, col_idx+q] = rr_parent[f]

    if reg > 0.0:
        A_mat += reg * np.eye(M_pairs)
        
    if reg_dip > 0.0:
        for I, pI in enumerate(fitted_pairs):
            if pI['L'] == 1:
                A_mat[I, I] += reg_dip
                
    X_I_nm = solve_kkt_tensor_reshaped(A_mat, b_tensor, C_mat, d_tensor)

    # =========================================================================
    # FAST Vectorized Output Tensor Generation
    # =========================================================================
    Q_A = np.zeros((mol.natm, Np, Np))
    mu_A = np.zeros((mol.natm, 3, Np, Np))

    for A in range(mol.natm):
        I_mask = (atom_idx_vec == A)
        if not np.any(I_mask): continue
            
        X_A = X_I_nm[I_mask]           
        C_A = C_mat[I_mask, 0]         
        
        Q_A[A] = -np.einsum('Iij,I->ij', X_A, C_A, optimize=True)
        mu_A[A] = np.einsum('Iij,Ix->xij', X_A, mu_vec_sums[I_mask], optimize=True)

    if return_fit:
        return Q_A, mu_A, A_mat_df, b_tensor_df, X_I_nm
    return Q_A, mu_A    
    
getDMFitChargeOperatorSpherical = getDMFitChargeOperatorSphericalNew2
getDMFitChargeOperatorDipoleSpherical = getDMFitChargeOperatorDipoleSphericalNew2

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
    elif rad_method == "vdw2":
        # vdw radii of the atoms this version is closer to the CHELP/Merz-Kollman style of grid
        r_vdw = radii.VDW[Z]
        r_grid = np.hstack([np.array([1]),1.0+rad_scal * np.arange(1,n_rad)])
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
    if block_size is None:
        block_size = N_grid
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
                              S=None,r_int=None,reg_type="none",reg_param=0.0,store_esp=True):
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
    
    
    #reg = 1.0e-7
    
    # matrices for fitting
    w_D = w.reshape((N_grid,1)) * D
    A_fit = (D.T).dot( w_D )
    A_fit0_diag = np.diag(A_fit)+0.
    if reg_type in ["mulliken","magnitude"]:
        A_fit[np.diag_indices_from(A_fit)] *= (1.0+reg_param)
    A_fit_inv = inv(A_fit)
    if store_esp:
        # get the ESP potential operators at grid points
        esp = getESPMat(mol,grid_coords,block_size) 
        b_fit = np.einsum('ka,knm->anm',w_D,esp,optimize=True)
    else:
        # if esp is not stored esp is evaluated in blocks of size "block_size"
        N_block = int(np.ceil(N_grid/block_size))
        b_fit = np.zeros([N_Q,N_AO,N_AO])
        for i in range(0,N_block):
            start = i*block_size
            end = min(start+block_size,N_grid)
            esp = -mol.intor('int1e_grids',grids=grid_coords[start:end,:])
            b_fit += np.einsum('ka,knm->anm',w_D[start:end],esp)
            # optimisation
            #K = end - start
            #b_partial_flat = w_D[start:end].T @ esp.reshape(K, -1)
            #b_fit += b_partial_flat.reshape(N_Q, N_AO, N_AO)
        esp = None
        
        
    
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

def getSimpleDFBasis(mol,gamma=1.0):
    '''
    This returns the set of gaussian parameters for the density-fitted charges
    '''
    # empty auxiliary mol with no basis functions
    mol_aux = mol.copy()
    mol_aux.basis = None
    mol_aux.build()
    
    # get vdw radii 
    R_vdw = radii.VDW[mol_aux.atom_charges()]
    #R_vdw = radii.COVALENT[mol_aux.atom_charges()]
    # decay parameters for auxiliary gaussians
    beta = 1/((gamma*R_vdw)**2)
    # normalisation coefficients so that int C_A g_A d^3r = 1
    C = 1.0/( ( (2*beta/np.pi)**(0.75) ) * ( (np.pi/beta)**(1.5) ))
    # add the basis functions to the mol_aux object
    new_atom_list = []
    custom_basis = {}
    # 2. Loop over every atom k in the original molecule
    for k in range(mol.natm):
        element = mol.elements[k]
        coord = mol.atom_coords()[k]
        # Create a unique label for the atom (e.g., 'O:0', 'H:1', 'H:2')
        # PySCF reads the true element from the letters before the colon
        unique_label = f"{element}:{k}" 
        new_atom_list.append((unique_label, coord))

        # Define the single s-type primitive Gaussian
        # Format: [ [l, (alpha, contraction_coeff)] ]
        # l=0 for s-type. Contraction coeff is 1.0 since it's uncontracted.
        custom_basis[unique_label] = [[0, [beta[k], 1.0]]]

    # Build the new molecule with the uniquely mapped basis
    mol_aux = gto.M(
        atom=new_atom_list,
        basis=custom_basis,
        charge=mol.charge,
        spin=mol.spin,
        unit='Bohr' # or 'Angstrom', matching your original mol
    )
    mol_aux.build()
             
    return beta, C, mol_aux

def getFlexibleDFBasis(mol,gamma=[1.0],l_max=0):
    '''
    This returns the set of gaussian parameters for the density-fitted charges
    '''
    # empty auxiliary mol with no basis functions
    mol_aux = mol.copy()
    mol_aux.basis = None
    mol_aux.build()
    
    # get vdw radii 
    R_vdw = radii.VDW[mol_aux.atom_charges()]
    #R_vdw = radii.COVALENT[mol_aux.atom_charges()]
    # decay parameters for auxiliary gaussians
    beta_0 = 1/((R_vdw)**2)
    beta = np.outer(beta_0 , 1.0/(np.array(gamma)**2))
    # normalisation coefficients so that int C_A g_A d^3r = 1
    C = 1.0/( ( (2*beta/np.pi)**(0.75) ) * ( (np.pi/beta)**(1.5) ))
    # add the basis functions to the mol_aux object
    new_atom_list = []
    custom_basis = {}
    # 2. Loop over every atom k in the original molecule
    for k in range(mol.natm):
        element = mol.elements[k]
        coord = mol.atom_coords()[k]
        # Create a unique label for the atom (e.g., 'O:0', 'H:1', 'H:2')
        # PySCF reads the true element from the letters before the colon
        unique_label = f"{element}:{k}" 
        new_atom_list.append((unique_label, coord))

        # Define the single s-type primitive Gaussian
        # Format: [ [l, (alpha, contraction_coeff)] ]
        # l=0 for s-type. Contraction coeff is 1.0 since it's uncontracted.
        custom_basis[unique_label] = [[0, [beta_l, 1.0]] for beta_l in beta[k,:]]
        if l_max>0:
            for l in range(1,l_max+1):
                custom_basis[unique_label] += [[l, [beta_l, 1.0]] for beta_l in beta[k,:]]

    # Build the new molecule with the uniquely mapped basis
    mol_aux = gto.M(
        atom=new_atom_list,
        basis=custom_basis,
        charge=mol.charge,
        spin=mol.spin,
        unit='Bohr' # or 'Angstrom', matching your original mol
    )
    mol_aux.build()
             
    return beta, C, mol_aux

def solve_kkt_tensor(A, b, C, d):
    """
    Solves the constrained minimization problem over the last dimension of a tensor.
    
    A: (N, N) matrix
    b: (..., N) tensor
    C: (N, M) constraint matrix
    d: (..., M) tensor of constraint values
    """
    N = A.shape[0]
    M = C.shape[1]
    
    # 1. Build the KKT Matrix: Shape (N+M, N+M)
    top_row = np.hstack((A, C))
    bottom_row = np.hstack((C.T, np.zeros((M, M))))
    kkt_mat = np.vstack((top_row, bottom_row))
    
    # 2. Build the RHS Tensor: Shape (..., N+M)
    # We append the constraint conditions to the last dimension of b
    rhs = np.concatenate((b, d), axis=-1)
    
    # 3. Solve the system
    # Method A: Direct inverse mapping to your sum notation 
    # x_{...i} = \sum_j (KKT^-1)_ij rhs_{...j}
    kkt_inv = np.linalg.inv(kkt_mat)
    solution = np.einsum('ij,...j->...i', kkt_inv, rhs)
    
    # Method B: More numerically stable solver (equivalent result)
    # rhs_flat = rhs.reshape(-1, N+M).T
    # sol_flat = np.linalg.solve(kkt_mat, rhs_flat).T
    # solution = sol_flat.reshape(rhs.shape)
    
    # 4. Extract x, dropping the M Lagrange multipliers from the last axis
    x = solution[..., :N]
    
    return x

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
        self.store_esp = True
        
        # espf regularisation
        self.espf_reg_type = None
        self.espf_reg_param = 0.0
        
        # density fitting charge parameters
        if self.multipole_order == 0:
            self.df_min_basis = setup_default_min_basis()
            self.df_reg = 1.0e-12
            self.df_reg_r2 = 0.0
            self.df_reg_S = 0.0 
            self.df_w_df = 1.0 
            self.df_w_mu = 1.0
            self.df_w_ovlp = 1.0
            self.df_w_quad = 0.0
            self.df_aux_basis = 'weigend-jfit'
            self.df_constrain_dip = False
            self.df_constrain_quad = False
            self.df_reg_dip = 0.0
        else:
            self.df_min_basis = setup_default_min_basis(add_pol=True)
            self.df_reg = 1.0e-12
            self.df_reg_r2 = 0.0
            self.df_reg_S = 0.0 
            self.df_w_df = 1.0 
            self.df_w_mu = 0.0
            self.df_w_ovlp = 1.0
            self.df_w_quad = 1.0e-1
            self.df_aux_basis = 'weigend-jfit'
            self.df_constrain_dip = True
            self.df_constrain_quad = False
            self.df_reg_dip = 0.0
            
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
        #t0 = timer()
        if self.multipole_method == "espf":
            self.getESPFMultipoleOperators(mol=mol)
        elif self.multipole_method == "mulliken":
            self.getMullikenMultipoleOperators(mol=mol)
        elif self.multipole_method == "df":
            self.getDensityFittedMultipoleOperators(mol=mol)
        else:
            raise Exception("Error:",self.multipole_method,"is not a recognised multipole method.")
        #print("Charge operator time = ", timer()-t0)
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
                                               block_size=self.grid_block_size,reg_type=self.espf_reg_type,reg_param=self.espf_reg_param,store_esp=self.store_esp)
        self.espf_fit_vars = fit_vars
        self.Q = Q 
        return 
    
    def getDensityFittedMultipoleOperators(self,mol=None):
        '''
        Generates the Minimal basis density fitted charges
        '''
        if mol is None:
            mol = self.mol

        
        if self.multipole_order == 0 :
            
            self.Q = getDMFitChargeOperatorSpherical(mol, min_basis=self.df_min_basis, reg=self.df_reg, reg_r2=self.df_reg_r2,
                                                     reg_S=self.df_reg_S, w_df=self.df_w_df, w_mu=self.df_w_mu, 
                                                     w_ovlp=self.df_w_ovlp, w_quad=self.df_w_quad, aux_basis=self.df_aux_basis, 
                                                    return_fit=False, constrain_dipole=self.df_constrain_dip, 
                                                     constrain_quad=self.df_constrain_quad)
            #if self.multipole_order > 0 :
            #    N = self.Q.shape[0]
            #    nao = self.Q.shape[1]
            #    self.Q = np.concatenate((self.Q,np.zeros([3*N,nao,nao])),axis=0)
            
        elif self.multipole_order == 1 :
            
            Q,mu = getDMFitChargeOperatorDipoleSpherical(mol, min_basis=self.df_min_basis, reg=self.df_reg, reg_r2=self.df_reg_r2,
                                                     reg_S=self.df_reg_S, w_df=self.df_w_df, w_mu=self.df_w_mu, 
                                                     w_ovlp=self.df_w_ovlp, w_quad=self.df_w_quad, aux_basis=self.df_aux_basis, 
                                                    return_fit=False, constrain_dipole=self.df_constrain_dip, 
                                                     constrain_quad=self.df_constrain_quad, reg_dip=self.df_reg_dip)
            self.Q = np.concatenate((Q,mu.swapaxes(0, 1).reshape(-1, mu.shape[2], mu.shape[3])),axis=0)
            
        else:
            print("Error! Only charges and dipoles implemented.")
        
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
    
    def getESPGrid(self,mol=None):
        '''
        Just returns grid coordinates, associated atoms, and the weights. Useful for analysis
        '''
        if mol is None:
            mol = self.mol
        grid_coords,grid_atms = getESPFGrid(mol,self.n_ang,self.n_rad,rad_scal=self.rad_scal,rad_method=self.rad_grid_method,ang_method=self.ang_grid_method)
        weights = getGridWeights(mol,grid_coords,weight_mode=self.weight_mode,hard_cut_vdw_scal=self.hard_cut_scal,sigma=self.smooth_sigma,smooth_func=self.weight_smooth_func)
        return grid_coords, grid_atms, weights