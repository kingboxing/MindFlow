import matlab.engine
import scipy.sparse as sp
from scipy.io import mmread
from src.Interface.Py2Mat import add_matlab_path, python2matlab, matlab2python
from src.Params.Params import DefaultParameters
from src.LinAlg.Utils import assemble_sparse

# Start MATLAB engine
eng = matlab.engine.start_matlab()
add_matlab_path(eng)
#%% eqn model
m = mmread("data/mess/NSE_RE_500_lvl1_M.mtx").tocsr()
a = mmread("data/mess/NSE_RE_500_lvl1_A.mtx").tocsr()
g = mmread("data/mess/NSE_RE_500_lvl1_G.mtx").tocsr()
b = mmread("data/mess/NSE_RE_500_lvl1_B.mtx")
c = mmread("data/mess/NSE_RE_500_lvl1_C.mtx")
k0 = mmread("data/mess/NSE_RE_500_lvl1_Feed0.mtx")
model = {'A_': assemble_sparse([[a, g], [g.T, None]])}
mg = sp.csr_matrix(g.shape)
model['E_'] = assemble_sparse([[m,mg],[mg.T, None]])
model['B'] = b
model['C'] = c
model_mat = python2matlab(model)
#%% params
param = DefaultParameters().parameters['radiriccati_mmess']['riccati_solver']
param['eqn']['type']='T'
#param['radi']['K0'] = python2matlab(k0)
param['radi']['get_ZZt'] = True
#%%
output = eng.GRiccatiDAE2RADISolver(model_mat, param)
out = matlab2python(output)
#eng.quit()