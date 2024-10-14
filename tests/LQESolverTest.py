#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:44:28 2024

@author: bojin
"""

from context import *

print('------------ Testing LQE Solver ------------')
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
m = sio.mmread("data/mess/NSE_RE_500_lvl1_M.mtx").tocsr()
a = sio.mmread("data/mess/NSE_RE_500_lvl1_A.mtx").tocsr()
g = sio.mmread("data/mess/NSE_RE_500_lvl1_G.mtx").tocsr()
b = sio.mmread("data/mess/NSE_RE_500_lvl1_B.mtx")
c = sio.mmread("data/mess/NSE_RE_500_lvl1_C.mtx")
model = {'A': a, 'B': b, 'C': c, 'M': m, 'G': g}


def pymess_riccati_test(model):
    print('------------ PyMESS Direct Solver------------')
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed1.mtx")
    solver = LQESolver(model)
    solver.sensor_noise(alpha=1)
    solver.disturbance(B=model['B'])
    solver.param['riccati_solver']['nm']['k0'] = k0
    solver.param['riccati_solver']['nm']['output'] = 0

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()

    solver.solve()
    norm_pa = solver.sys_energy(MatQ=matq, chunk_size=80)
    norm_se = solver.sys_energy(MatQ=matq, pid=1)
    print('Results are printed as follows : ')
    print("H2 Norm = : %e (parallel: %e) \n" % (norm_se, norm_pa))
    print("Size of Low Rank Solution Factor Z: %d x %d \n" % (solver.facZ.shape))
    print("it = %d \t rel_res2 = %e\t res2 = %e \n" % (
        solver.status.it, solver.status.res2_norm / solver.status.res2_0, solver.status.res2_norm))


def pymess_riccati_test2(model):
    print('------------ PyMESS Accumulation Solver------------')
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed1.mtx")
    solver = LQESolver(model)
    solver.sensor_noise(alpha=1)
    solver.disturbance(B=model['B'])
    solver.param['riccati_solver']['nm']['k0'] = k0
    solver.param['riccati_solver']['nm']['output'] = 0

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()

    sol = solver.iter_solve(num_iter=2, MatQ=matq)
    norm_pa = sol['sqnorm_sys']
    print('Results are printed as follows : ')
    print("H2 Norm = : %e \n" % (norm_pa))
    print("Size of 1st Low Rank Solution Factor Z: %d x %d \n" % (sol['size'][0]))
    print("Size of 2nd Low Rank Solution Factor Z: %d x %d \n" % (sol['size'][1]))
    print("1st: it = %d \t rel_res2 = %e\t res2 = %e \n" % (
        sol['status'][0].it, sol['status'][0].res2_norm / sol['status'][0].res2_0, sol['status'][0].res2_norm))

    print("2nd: it = %d \t rel_res2 = %e\t res2 = %e \n" % (
        sol['status'][1].it, sol['status'][1].res2_norm / sol['status'][1].res2_0, sol['status'][1].res2_norm))


def mmess_nmriccati_test(model):
    print('------------ MMESS Newton Solver------------')
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed1.mtx")
    solver = LQESolver(model, backend='matlab')
    solver.sensor_noise(alpha=1)
    solver.disturbance(B=model['B'])
    solver.param['riccati_solver']['nm']['K0'] = k0.T
    solver.param['riccati_solver']['nm']['info'] = False
    solver.param['riccati_solver']['nm']['inexact'] = False
    solver.param['riccati_solver']['adi']['maxiter'] = 2000
    solver.param['riccati_solver']['shifts']['method'] = 'projection'

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    solver.solve()

    norm_pa = solver.sys_energy(MatQ=matq, chunk_size=80)
    norm_se = solver.sys_energy(MatQ=matq, pid=1)
    print('Results are printed as follows : ')
    print("Elapsed Time = : %e \n" % (solver.status['etime']))
    print("H2 Norm = : %e (parallel: %e) \n" % (norm_se, norm_pa))
    print("Size of Low Rank Solution Factor Z: %d x %d \n" % (solver.facZ['Z'].shape))
    print("it = %d \t rel_res2 = %e\t res2 = %e \n" % (
        solver.status['niter'], solver.status['res'][-1], solver.status['rc'][-1]))


def mmess_nmriccati_test2(model):
    print('------------ MMESS Newton Accumulation Solver------------')
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed1.mtx")
    solver = LQESolver(model, backend='matlab')
    solver.sensor_noise(alpha=1)
    solver.disturbance(B=model['B'])
    solver.param['riccati_solver']['nm']['K0'] = k0.T
    solver.param['riccati_solver']['nm']['info'] = False
    solver.param['riccati_solver']['nm']['inexact'] = False
    solver.param['riccati_solver']['adi']['maxiter'] = 2000
    solver.param['riccati_solver']['shifts']['method'] = 'projection'

    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    sol = solver.iter_solve(num_iter=2, MatQ=matq)
    norm_pa = sol['sqnorm_sys']

    print('Results are printed as follows : ')
    print("1st Elapsed Time = : %e \n" % (sol['status'][0]['etime']))
    print("2nd Elapsed Time = : %e \n" % (sol['status'][1]['etime']))

    print("H2 Norm = : %e \n" % (norm_pa))
    print("Size of 1st Low Rank Solution Factor Z: %d x %d \n" % (sol['size'][0]))
    print("Size of 2nd Low Rank Solution Factor Z: %d x %d \n" % (sol['size'][1]))

    print("1st: it = %d \t rel_res2 = %e\t res2 = %e \n" % (
        sol['status'][0]['niter'], sol['status'][0]['res'][-1], sol['status'][0]['rc'][-1]))

    print("2nd: it = %d \t rel_res2 = %e\t res2 = %e \n" % (
        sol['status'][1]['niter'], sol['status'][1]['res'][-1], sol['status'][1]['rc'][-1]))


def mmess_radiriccati_test(model, case):
    print(f'------------ MMESS RADI Solver (case {case})------------')
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed1.mtx")
    solver = LQESolver(model, backend='matlab', method='radi')
    if case == 0:
        # only K
        solver.param['riccati_solver']['radi']['K0'] = k0.T
    if case == 1:
        # Z * inv(Y) * Z'
        solver.param['riccati_solver']['radi']['get_ZZt'] = False
    if case == 2:
        # Z * Z'
        solver.param['riccati_solver']['radi']['get_ZZt'] = True
    if case == 3:
        # Z * D * Z
        solver.param['riccati_solver']['radi']['get_ZZt'] = False
        solver.param['riccati_solver']['LDL_T'] = True

    solver.sensor_noise(alpha=1)
    solver.disturbance(B=model['B'])
    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    solver.solve()

    norm_pa = solver.sys_energy(MatQ=matq, chunk_size=80)
    norm_se = solver.sys_energy(MatQ=matq, pid=1)
    print('Results are printed as follows : ')
    print("Elapsed Time = : %e \n" % (solver.status['etime']))
    print("H2 Norm = : %e (parallel: %e) \n" % (norm_se, norm_pa))
    if solver.facZ['Z'] is not None:
        print("Size of Low Rank Solution Factor Z: %d x %d \n" % (solver.facZ['Z'].shape))
    print("it = %d \t res = %e\t \n" % (
        solver.status['niter'], solver.status['res'].flatten()[-1]))


def mmess_radiriccati_test2(model, case, mode):
    print(f'------------ MMESS RADI Accumulation Solver (case {case}, mode {mode})------------')
    k0 = sio.mmread("data/mess/NSE_RE_500_lvl1_Feed1.mtx")
    solver = LQESolver(model, backend='matlab', method='radi')
    if case == 0:
        # only K
        solver.param['riccati_solver']['radi']['K0'] = k0.T
    if case == 1:
        # Z * inv(Y) * Z'
        solver.param['riccati_solver']['radi']['get_ZZt'] = False
    if case == 2:
        # Z * Z'
        solver.param['riccati_solver']['radi']['get_ZZt'] = True
    if case == 3 and mode == 0:  #LDL_T formulation with eqn.haveUV == 'True' is not yet implemented.
        # Z * D * Z
        solver.param['riccati_solver']['radi']['get_ZZt'] = False
        solver.param['riccati_solver']['LDL_T'] = True

    solver.sensor_noise(alpha=1)
    solver.disturbance(B=model['B'])
    matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
    sol = solver.iter_solve(num_iter=2, MatQ=matq, mode=mode)
    norm_pa = sol['sqnorm_sys']

    print('Results are printed as follows : ')
    print("1st Elapsed Time = : %e \n" % (sol['status'][0]['etime']))
    print("2nd Elapsed Time = : %e \n" % (sol['status'][1]['etime']))

    print("H2 Norm = : %e \n" % (norm_pa))
    print(f"Size of 1st Low Rank Solution Factor Z: {sol['size'][0]} \n")
    print(f"Size of 2nd Low Rank Solution Factor Z: {sol['size'][1]} \n")

    print("1st: it = %d \t res = %e \n" % (
        sol['status'][0]['niter'], sol['status'][0]['res'].flatten()[-1]))
    print("2nd: it = %d \t res = %e \n" % (
        sol['status'][1]['niter'], sol['status'][1]['res'].flatten()[-1]))


#%%
pymess_riccati_test(model)
pymess_riccati_test2(model)
mmess_nmriccati_test(model)
mmess_nmriccati_test2(model)
mmess_radiriccati_test(model, 0)
mmess_radiriccati_test(model, 1)
mmess_radiriccati_test(model, 2)
mmess_radiriccati_test(model, 3)
mmess_radiriccati_test2(model, 0, 0)
mmess_radiriccati_test2(model, 1, 0)
mmess_radiriccati_test2(model, 2, 0)
mmess_radiriccati_test2(model, 3, 0)
mmess_radiriccati_test2(model, 0, 1)
mmess_radiriccati_test2(model, 1, 1)
mmess_radiriccati_test2(model, 2, 1)
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
print('Elapsed Time = %e' % (elapsed_time))
print(f"Average CPU usage: {round(np.average(cpu_usage_diff), 2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')
