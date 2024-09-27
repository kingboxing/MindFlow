#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) Peter Benner, Martin Koehler, Jens Saak and others
#               2009-2018
#

"""to document"""
from scipy.io import mmread
import tracemalloc
import psutil
import time
import numpy as np
from pymess import EquationGRiccatiDAE2, lrnm, Options, MESS_LRCFADI_PARA_ADAPTIVE_V, MESS_MEMORY_HIGH

"""Solve Riccati Equation DAE2 System."""
print('------------ Testing Riccati Equation DAE2 System Solver ------------')
tracemalloc.start()
process = psutil.Process()
cpu_usage_before = psutil.cpu_percent(interval=None, percpu=True)
start_time = time.time()
#%%
# read data
m = mmread("NSE_RE_100_lvl1_M.mtx").tocsr()
a = mmread("NSE_RE_100_lvl1_A.mtx").tocsr()
g = mmread("NSE_RE_100_lvl1_G.mtx").tocsr()
b = mmread("NSE_RE_100_lvl1_B.mtx")
c = mmread("NSE_RE_100_lvl1_C.mtx")
delta = -0.02

#create opt instance
opt = Options()
opt.adi.output = 0
opt.nm.output = 0
opt.nm.res2_tol = 1e-2
opt.adi.paratype = MESS_LRCFADI_PARA_ADAPTIVE_V
opt.adi.memory_usage = MESS_MEMORY_HIGH

# create equation
eqn = EquationGRiccatiDAE2(opt, m, a, g, b, c, delta)

# solve equation
z, status = lrnm(eqn, opt)

# get residual
res2 = status.res2_norm
res2_0 = status.res2_0
it = status.it
print("Size of Low Rank Solution Factor Z: %d x %d \n"%(z.shape))
print("it = %d \t rel_res2 = %e\t res2 = %e \n" % (it, res2 / res2_0, res2))
print(status.lrnm_stat())
#%%
elapsed_time = time.time() - start_time
cpu_usage_after = psutil.cpu_percent(interval=None, percpu=True)
cpu_usage_diff = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print('Elapsed Time = %e' % (elapsed_time))
print(f"Current memory usage: {current / (1024 * 1024):.2f} MB")
print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")
print(f"Average CPU usage: {round(np.average(cpu_usage_diff),2)}")
cores_used = sum(1 for usage in cpu_usage_diff if usage > 0)
print(f"Number of CPU cores actively used: {cores_used}")
print('------------ Testing completed ------------')