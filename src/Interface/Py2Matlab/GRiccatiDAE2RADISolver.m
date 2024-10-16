function output = GRiccatiDAE2RADISolver(model, param)
%% function [out, eqn, opts, oper] = mess_lrradi(eqn,opts, oper)
%
% Solve continuous-time Riccati equations with sparse coefficients with
% the RADI method [1]. With X = Z*inv(Y)*Z',
%
%   eqn.type = 'N'
%     A*X*E' + E*X*A' - E*X*C'*C*X*E' + B*B' = 0
%   or
%     A*X*E' + E*X*A' - E*X*C'*Q\C*X*E' + B*R*B' = 0
%
%   eqn.type = 'T'
%     A'*X*E + E'*X*A - E'*X*B*B'*X*E + C'*C = 0
%   or
%     A'*X*E + E'*X*A - E'*X*B*R\B'*X*E + C'*Q*C = 0
%
% Matrix A can have the form A = Ã + U*V' if U (eqn.U) and V (eqn.V) are
% provided U and V are dense (n x m3) matrices and should satisfy m3 << n
%
% Input
%   model       struct contains data for creating equation matrices (from python dict)
%
%   opts        struct contains parameters for the algorithm (from python dict)
%
% Output
%   output      struct containing output information
%
% Output fields in struct output:
%
%   out.Z           low rank solution factor, the solution is
%                   opts.radi.get_ZZt = false: X = Z*inv(Y)*Z'
%                   opts.radi.get_ZZt = true: X = Z*Z'
%                   (opts.radi.compute_sol_fac = true and not only initial K0)
%
%   out.Y           small square solution factor, the solution is
%                   opts.radi.get_ZZt = false: X = Z*inv(Y)*Z'
%                   (opts.radi.compute_sol_fac = true and not only initial K0)
%
%   out.D           solution factor for LDL^T formulation, the solution is
%                   opts.LDL_T = true: X = Z*D*Z'
%                   (opts.LDL_T = true)
%
%   out.K           stabilizing Feedback matrix
%                   The feedback matrix K can be accumulated during the iteration:
%                   eqn.type = 'N' -> K = (E*X*C')' or K = (E*X*C)'/Q
%                   eqn.type = 'T' -> K = (E'*X*B)' or K = (E'*X*B)'/R
%
%   out.timesh      time of the overall shift computation
%
%   out.p           used shifts
%
%   out.niter       number of RADI iterations
%
%   out.res         array of relative RADI residual norms
%                   (opts.radi.res_tol nonzero)
%
%   out.rc          array of relative RADI change norms
%                   (opts.radi.rel_diff_tol nonzero)
%
%   out.res_fact    final Riccati residual factor W of the iteration
%
%   out.res0        norm of the normalization residual term
%
%   output.etime    elapsed time of solving

%% initilise
opts = struct();
[oper, opts] = operatormanager(opts, 'dae_2');
opts = update_param(opts, param);
eqn = ssmodel(model);
%% update eqn parameters
eqn.type = opts.eqn.type;
eqn.haveE = opts.eqn.haveE;
eqn.haveUV = opts.eqn.haveUV;
if isfield(opts.eqn, 'sizeUV1')
    eqn.sizeUV1 = opts.eqn.sizeUV1;
end
opts=rmfield(opts, "eqn");

%% solve using newton method
ts = tic;
out = mess_lrradi(eqn, opts, oper);
te = toc(ts);
%% output
output = out;
output.etime = te;
output.res = transpose(out.res);

if isfield(out, 'D')
    output.D = spmat2struct(sparse(out.D));
end
if isfield(out, 'Y')
    output.Y = spmat2struct(sparse(out.Y));
end
end