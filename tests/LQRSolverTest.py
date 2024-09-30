#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:41:28 2024

@author: bojin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:44:28 2024

@author: bojin
"""

from context import *

print('------------ Testing LQR Solver ------------')
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
m = sio.mmread("data/pymess/NSE_RE_100_lvl1_M.mtx").tocsr()
a = sio.mmread("data/pymess/NSE_RE_100_lvl1_A.mtx").tocsr()
g = sio.mmread("data/pymess/NSE_RE_100_lvl1_G.mtx").tocsr()
b = sio.mmread("data/pymess/NSE_RE_100_lvl1_B.mtx")
c = sio.mmread("data/pymess/NSE_RE_100_lvl1_C.mtx")

ssmodel = {'A': a, 'B': b, 'C': c, 'M': m, 'G': g}

solver = LQRSolver(ssmodel)
solver.control_penalty(beta=1)
solver.measure(Cz=ssmodel['C'])
matq =cholesky(m.tocsc(), ordering_method="natural").L().transpose()

sol = solver.solve(MatQ=matq)
print('Results are printed as follows : ')
print("H2 Norm = : %e \n"%(sol[2]))
print("Size of Low Rank Solution Factor Z: %d x %d \n"%(solver.facZ.shape))
print("it = %d \t rel_res2 = %e\t res2 = %e \n" % (sol[1][0].it, sol[1][0].res2_norm / sol[1][0].res2_0, sol[1][0].res2_norm))

#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
print('Elapsed Time = %e' % (elapsed_time))
print(f"Average CPU usage: {round(np.average(cpu_usage_diff),2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')