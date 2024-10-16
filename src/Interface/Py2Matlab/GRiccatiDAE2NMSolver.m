function output = GRiccatiDAE2NMSolver(model, param)
%% Solve continuous-time Riccati equations with sparse coefficients with Newton's method (NM)
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
%
% Matrix A can have the form A = Ã + U*V' if U (eqn.U) and V (eqn.V) are
% provided U and V are dense (n x m3) matrices and should satisfy m3 << n
%
%
% The solution is approximated as X = Z*Z', or if opts.LDL_T is true as
% X = L*D*L'
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
%   output.Z                low-rank solution factor
%
%   output.adi              struct with the output of the all ADI iterations
%
%   output.niter            number of NM iterations
%
%   output.K                feedback matrix
%                           (T): K = B' ZZ' E
%                           (N): K = C ZZ' E
%                           or
%                           (T): K = R \ B' ZDZ' E
%                           (N): K = Q \ C ZDZ' E
%
%   output.D                solution factor for LDL^T formulation
%                           (opts.LDL_T = true)
%
%   output.res              array of relative NM residual norms
%
%   output.rc               array of relative NM change norms
%
%   output.res0             norm of the normalization residual term
%
%   output.etime            elapsed time of solving

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
out = mess_lrnm(eqn, opts, oper);
te = toc(ts);
%% output
output = rmfield(out, 'adi');
output.etime= te;
if isfield(out, 'D')
    output.D = spmat2struct(sparse(out.D));
end

end