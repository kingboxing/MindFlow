import matlab.engine
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
from src.Interface.Py2Mat import add_matlab_path, python2matlab, matlab2python, dict2spmat, convert_to_matlab_dtype, convert_to_matlab_struct
from src.Params.Params import DefaultParameters
from src.LinAlg.Utils import assemble_sparse

def NMRiccatiSiolver(eng):
    # %% eqn model
    m = mmread("data/mess/NSE_RE_500_lvl1_M.mtx").tocsr()
    a = mmread("data/mess/NSE_RE_500_lvl1_A.mtx").tocsr()
    g = mmread("data/mess/NSE_RE_500_lvl1_G.mtx").tocsr()
    b = mmread("data/mess/NSE_RE_500_lvl1_B.mtx")
    c = mmread("data/mess/NSE_RE_500_lvl1_C.mtx")
    k0 = mmread("data/mess/NSE_RE_500_lvl1_Feed0.mtx")
    model = {'A_': assemble_sparse([[a, g], [g.T, None]])}
    mg = sp.csr_matrix(g.shape)
    model['E_'] = assemble_sparse([[m, mg], [mg.T, None]])
    model['B'] = b
    model['C'] = c
    model_mat = python2matlab(model)
    # %% params
    param = DefaultParameters().parameters['riccati_lrnm_mmess']['riccati_solver']
    param['eqn']['type'] = 'T'
    param['adi']['maxiter'] = 300
    param['shifts']['num_desired'] = 5
    param['shifts']['method'] = 'projection'
    param['nm']['K0'] = python2matlab(k0)
    # %%
    output = eng.GRiccatiDAE2NMSolver(model_mat, param)
    out = matlab2python(output)
    return out
def main():
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    add_matlab_path(eng, ['OptimControl'])
    # define sparse matrix
    A = sp.csr_matrix(np.array([[1, 0, 1], [0, 2, 0], [0, 5, 3]]))
    # Define a Python dictionary
    python_dict = {
        'param1': 42,  # int
        'param2': [1, 2, 3],  # list
        'param3': 'test',  # str
        'param4': True,  # bool
        'param5': 1.2242,  # float
        'param6': 1.23 + 1.234 * 1j,  # complex
        'param7': (34, 12.2),  # tuple
        'param8': {
            'subparam1': 'test',
            'subparam2': [4, 5, 6],
            'subparam3': {
                'deep_param1': 99,
                'deep_param2': 'deep_value'
            }
        },  # dict
        'param9': [True, False],
        'param10': np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
        'param11': A,
    }

    # Convert Python dictionary to MATLAB-compatible data types
    matlab_dict = convert_to_matlab_dtype(python_dict)
    # Create MATLAB structure in the MATLAB workspace dynamically
    convert_to_matlab_struct(eng, matlab_dict, struct_name='struct_var')
    # Access the created MATLAB struct in the MATLAB workspace
    struct_in_matlab = eng.workspace['struct_var']
    print("MATLAB struct created in workspace:", struct_in_matlab)  # Optional verification

    # Convert the Python dictionary to MATLAB struct and pass it to MATLAB
    mdict = python2matlab(python_dict)
    # Test passing the dictionary to MATLAB
    mat = eng.pass_dict(mdict)
    # Convert the entire MATLAB dictionary back to Python
    pdict = matlab2python(mat)
    # Convert the sparse matrix from MATLAB back to Python
    Am = pdict['param11']
    print("Original sparse matrix from Python:\n", A.toarray())
    print("Converted sparse matrix from MATLAB to Python:\n", Am.toarray())
    print("Converted dictionary from MATLAB to Python:\n", pdict)
    # Print the current working directory in MATLAB (optional)
    print("Current MATLAB directory:", eng.pwd())

    out = NMRiccatiSiolver(eng)
    print(f'mess_lrnm took {out.etime} seconds.')
    # Stop MATLAB engine
    eng.quit()

# Run the main function
if __name__ == "__main__":
    main()