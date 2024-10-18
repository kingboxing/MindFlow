clc
clear
%%
re = 500;
level = 5;
istest = false;

%% Set operations
opts = struct();
[oper, opts] = operatormanager(opts, 'dae_2');

[eqn, K0, ~] = mess_get_NSE(re, level);
%% Lets try the RADI method and compare
eqn.type                = 'T';
eqn.Q                   = eye(size(eqn.C,1));
eqn.R                   = eye(size(eqn.B,2));
opts.norm               = 2;
opts.LDL_T              = false;
% RADI-MESS settings
%opts.shifts.history     = 5 * size(eqn.C, 1);
opts.shifts.num_desired = 5;

% choose either of the three shift methods, here
opts.shifts.method = 'gen-ham-opti';
%     opts.shifts.method = 'heur';
%     opts.shifts.method = 'projection';

opts.shifts.naive_update_mode = false; % .. Suggest false
% (smart update is faster;
%  convergence is the same).
opts.shifts.info              = 0;
%%
opts.radi.compute_sol_fac     = true; % Turned on for numerical stability reasons.
opts.radi.get_ZZt             = false;
opts.radi.maxiter             = 300;
opts.radi.res_tol             = 1e-10;
opts.radi.rel_diff_tol        = 0;
opts.radi.info                = 0;
%opts.radi.K0                  = K0;
%%

t_mess_lrradi = tic;
outradi = mess_lrradi(eqn, opts, oper);
t_elapsed2 = toc(t_mess_lrradi);
mess_fprintf(opts, 'mess_lrradi took %6.2f seconds \n', t_elapsed2);

if not(istest)
    figure();
    semilogy(outradi.res, 'LineWidth', 3);
    title('0 = C^TC + A^T X E + E^T X A - E^T X BB^T X E');
    xlabel('number of iterations');
    ylabel('normalized residual norm');
end