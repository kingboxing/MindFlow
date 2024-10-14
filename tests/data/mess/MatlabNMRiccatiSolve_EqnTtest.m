clc
clear
%%
re = 500;
level = 1;
istest = false;
%% Set operations
opts = struct();
[oper, opts] = operatormanager(opts, 'dae_2');

[eqn, K0, ~] = mess_get_NSE(re, level);
opts.nm.K0 = K0;
%opts.radi.K0 = K0;
%%
% First we run the Newton-ADI Method
opts.norm = 2;

% ADI tolerances and maximum iteration number
opts.adi.maxiter       = 300;
opts.adi.res_tol       = 1e-12;
opts.adi.rel_diff_tol  = 1e-16;
opts.adi.info          = 0;
opts.adi.LDL_T         = false;
opts.adi.accumulateK   = true;
opts.adi.accumulateDeltaK = false;
opts.adi.compute_sol_fac = false;
eqn.type = 'T';
%%
n = size(eqn.A_, 1);
opts.shifts.num_desired = 5; % *nout;
opts.shifts.num_Ritz    = 50;
opts.shifts.num_hRitz   = 25;
opts.shifts.method      = 'projection';
%opts.shifts.b0          = ones(n, 1);
%%
% Newton tolerances and maximum iteration number
opts.nm.maxiter          = 20;
opts.nm.res_tol          = 1e-10;
opts.nm.rel_diff_tol     = 1e-16;
opts.nm.info             = 1;
opts.nm.projection.freq  = 0;
opts.nm.projection.ortho = true;
% in case you want to e.g. specify the factored Newton solver for
% the projected equations uncomment the following
% opts.nm.projection.meth = 'care_nwt_fac';

opts.nm.res              = struct('maxiter', 10, ...
                                  'tol', 1e-6, ...
                                  'orth', 0);
opts.nm.linesearch       = true;
opts.nm.inexact          = 'superlinear';
opts.nm.tau              = 0.1;
opts.nm.accumulateRes    = true;

%% use low-rank Newton-Kleinman-ADI
t_mess_lrnm = tic;
[outnm, eqnnm, optsnm, opernm] = mess_lrnm(eqn, opts, oper);
t_elapsed1 = toc(t_mess_lrnm);
mess_fprintf(opts, 'mess_lrnm took %6.2f seconds \n\n', t_elapsed1);
%%
if not(istest)
    figure(1);
    semilogy(outnm.res, 'LineWidth', 3);
    title('0 = C^T C + A^T X E + E^T X A - E^T X BB^T X E');
    xlabel('number of newton iterations');
    ylabel('normalized residual norm');
    pause(1);
end
[mZ, nZ] = size(outnm.Z);
mess_fprintf(opts, 'size outnm.Z: %d x %d\n\n', mZ, nZ);
