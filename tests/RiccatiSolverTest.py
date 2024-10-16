#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:38:16 2024

@author: bojin
"""

from context import *
from src.LinAlg.Utils import assemble_sparse

print('------------ Testing Riccati Solver ------------')
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()


#%%
def pymess_riccati_test():
    print('------------ PyMESS Direct Solver------------')
    m = sio.mmread("data/mess/NSE_RE_500_lvl1_M.mtx").tocsr()
    a = sio.mmread("data/mess/NSE_RE_500_lvl1_A.mtx").tocsr()
    g = sio.mmread("data/mess/NSE_RE_500_lvl1_G.mtx").tocsr()
    b = sio.mmread("data/mess/NSE_RE_500_lvl1_B.mtx")
    c = sio.mmread("data/mess/NSE_RE_500_lvl1_C.mtx")
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed0.mtx")

    model = {'A': a, 'B': b, 'C': c, 'M': m, 'G': g}

    mg = sp.csr_matrix(g.shape)
    model['E_full'] = assemble_sparse([[m, mg], [mg.T, None]])
    model['A_full'] = assemble_sparse([[a, g], [g.T, None]])

    solver = GRiccatiDAE2Solver(model)
    solver.param[solver.param['solver_type']]['nm']['maxit'] = 40
    solver.param[solver.param['solver_type']]['nm']['k0'] = k0
    solver.param[solver.param['solver_type']]['type'] = mess.MESS_OP_TRANSPOSE

    solver.solve_riccati()

    # solve equation
    z = solver.facZ

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    status = solver.status
    h2norm = solver.sys_energy(matq, pid=1)
    h2norm_pa = solver.sys_energy(matq, chunk_size=500)

    # get residual
    res2 = status.res2_norm
    res2_0 = status.res2_0
    it = status.it
    print('Results are printed as follows : ')
    print("H2 Norm = : %e (parallel: %e )\n" % (h2norm, h2norm_pa))
    print("Size of Low Rank Solution Factor Z: %d x %d \n" % (z.shape))
    print("it = %d \t rel_res2 = %e\t res2 = %e \n" % (it, res2 / res2_0, res2))


#%%
def mmess_nmriccati_test(case=0):
    print(f'------------ MMESS Newton Solver (case {case})------------')
    m = sio.mmread("data/mess/NSE_RE_500_lvl1_M.mtx").tocsr()
    a = sio.mmread("data/mess/NSE_RE_500_lvl1_A.mtx").tocsr()
    g = sio.mmread("data/mess/NSE_RE_500_lvl1_G.mtx").tocsr()
    b = sio.mmread("data/mess/NSE_RE_500_lvl1_B.mtx")
    c = sio.mmread("data/mess/NSE_RE_500_lvl1_C.mtx")
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed0.mtx")

    model = {'A': a, 'B': b, 'C': c, 'M': m, 'G': g}

    mg = sp.csr_matrix(g.shape)
    model['E_full'] = assemble_sparse([[m, mg], [mg.T, None]])
    model['A_full'] = assemble_sparse([[a, g], [g.T, None]])

    solver = GRiccatiDAE2Solver(model, backend='matlab')
    solver.param[solver.param['solver_type']]['eqn']['type'] = 'T'
    solver.param[solver.param['solver_type']]['adi']['maxiter'] = 300
    solver.param[solver.param['solver_type']]['shifts']['num_desired'] = 5
    solver.param[solver.param['solver_type']]['shifts']['method'] = 'projection'
    solver.param[solver.param['solver_type']]['nm']['K0'] = k0
    solver.param[solver.param['solver_type']]['nm']['info'] = False
    if case == 1:
        # Z * D * Z
        solver.param[solver.param['solver_type']]['LDL_T'] = True
        solver.eqn['Q'] = np.identity(c.shape[0])
        solver.eqn['R'] = np.identity(b.shape[1])

    solver.solve_riccati()

    # solve equation
    z = solver.facZ['Z']

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    status = solver.status
    h2norm = solver.sys_energy(matq, pid=1)
    h2norm_pa = solver.sys_energy(matq, chunk_size=500)

    # get residual
    res = status['res'][-1]
    rc = status['rc'][-1]
    it = status['niter']
    print('Results are printed as follows : ')
    print("H2 Norm = : %e (parallel: %e )\n" % (h2norm, h2norm_pa))
    print("Size of Low Rank Solution Factor Z: %d x %d \n" % (z.shape))
    print("it = %d \t res = %e\t rc = %e \n" % (it, res, rc))


#%%
def mmess_radiriccati_test(case):
    print(f'------------ MMESS RADI Solver (case {case})------------')
    m = sio.mmread("data/mess/NSE_RE_500_lvl1_M.mtx").tocsr()
    a = sio.mmread("data/mess/NSE_RE_500_lvl1_A.mtx").tocsr()
    g = sio.mmread("data/mess/NSE_RE_500_lvl1_G.mtx").tocsr()
    b = sio.mmread("data/mess/NSE_RE_500_lvl1_B.mtx")
    c = sio.mmread("data/mess/NSE_RE_500_lvl1_C.mtx")
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed0.mtx")

    model = {'A': a, 'B': b, 'C': c, 'M': m, 'G': g}

    mg = sp.csr_matrix(g.shape)
    model['E_full'] = assemble_sparse([[m, mg], [mg.T, None]])
    model['A_full'] = assemble_sparse([[a, g], [g.T, None]])

    solver = GRiccatiDAE2Solver(model, method='radi', backend='matlab')
    solver.param[solver.param['solver_type']]['eqn']['type'] = 'T'
    if case == 0:
        # only K
        solver.param[solver.param['solver_type']]['radi']['K0'] = k0
    if case == 1:
        # Z * inv(Y) * Z'
        solver.param[solver.param['solver_type']]['radi']['get_ZZt'] = False
    if case == 2:
        # Z * Z'
        solver.param[solver.param['solver_type']]['radi']['get_ZZt'] = True
    if case == 3:
        # Z * D * Z
        solver.param[solver.param['solver_type']]['radi']['get_ZZt'] = False
        solver.param[solver.param['solver_type']]['LDL_T'] = True
        solver.eqn['Q'] = np.identity(c.shape[0])
        solver.eqn['R'] = np.identity(b.shape[1])

    solver.solve_riccati()

    # solve equation
    z = solver.facZ

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    status = solver.status
    h2norm = solver.sys_energy(matq, pid=1)
    h2norm_pa = solver.sys_energy(matq, chunk_size=500)

    # get residual
    res = status['res'].flatten()[-1]
    it = status['niter']
    print('Results are printed as follows : ')
    print(f"Case {case}: \n")
    print(f"H2 Norm = : {h2norm} (parallel: {h2norm_pa})\n")
    if z['Z'] is not None:
        print("Size of Low Rank Solution Factor Z: %d x %d \n" % (z['Z'].shape))
    print("it = %d \t res = %e \n" % (it, res))


#%%
#pymess_riccati_test()
#mmess_nmriccati_test()
mmess_nmriccati_test(case=1)
# mmess_radiriccati_test(case=0)
# mmess_radiriccati_test(case=1)
# mmess_radiriccati_test(case=2)
# mmess_radiriccati_test(case=3)
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
print('Elapsed Time = %e' % (elapsed_time))
print(f"Average CPU usage: {round(np.average(cpu_usage_diff), 2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')
