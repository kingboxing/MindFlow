#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:38:16 2024

@author: bojin
"""


from context import *

print('------------ Testing Riccati Solver ------------')
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
m = sio.mmread("data/mess/NSE_RE_100_lvl1_M.mtx").tocsr()
a = sio.mmread("data/mess/NSE_RE_100_lvl1_A.mtx").tocsr()
g = sio.mmread("data/mess/NSE_RE_100_lvl1_G.mtx").tocsr()
b = sio.mmread("data/mess/NSE_RE_100_lvl1_B.mtx")
c = sio.mmread("data/mess/NSE_RE_100_lvl1_C.mtx")

ssmodel = {'A': a, 'B': b, 'C': c, 'M': m, 'G': g}
solver = GRiccatiDAE2Solver(ssmodel)
solver.update_riccati_params({'nm':{'maxit':40}})
solver.update_riccati_params({'type':mess.MESS_OP_TRANSPOSE})
solver.solve_riccati()

# solve equation
z = solver.facZ

matq = cholesky(m.tocsc(), ordering_method="natural").L().transpose()
status = solver.status
h2norm = solver.squared_h2norm(matq)
h2norm_pa = solver.squared_h2norm(matq, chunk_size = 500)

# get residual
res2 = status.res2_norm
res2_0 = status.res2_0
it = status.it
print('Results are printed as follows : ')
print("H2 Norm = : %e (parallel: %e )\n"%(h2norm, h2norm_pa))
print("Size of Low Rank Solution Factor Z: %d x %d \n"%(z.shape))
print("it = %d \t rel_res2 = %e\t res2 = %e \n" % (it, res2 / res2_0, res2))
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
print('Elapsed Time = %e' % (elapsed_time))
print(f"Average CPU usage: {round(np.average(cpu_usage_diff),2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')