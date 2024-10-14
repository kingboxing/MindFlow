function eqn = ssmodel(struct_data)
    % SSMODEL Creates sparse matrices for substructs marked with 'type' = 'sparse matrix'.
    %
    % This function iterates through the fields of an input struct containing substructs.
    % If a substruct contains the field 'type' with the value 'sparse matrix', it will be
    % passed to the struct2spmat function to create a sparse matrix, provided it also contains
    % the necessary fields: 'data', 'rows', 'cols', and 'shape'.
    % Any substructs without the 'type' field or where 'type' is not 'sparse matrix' will
    % remain unchanged in the new output struct.
    %
    % Input:
    %   struct_data - A struct where each field is a substruct. If a substruct is marked with
    %                 'type' = 'sparse matrix', it will be converted to a sparse matrix.
    %
    % Output:
    %   eqn - A struct where fields that were substructs marked as 'sparse matrix'
    %         will contain sparse matrices, while other substructs will remain unchanged.
    %   
    %   Output fields in struct eqn:
    %      
    %   The first order system
    %
    %           E * z'(t) = A * z(t) + B * u(t)
    %                y(t) = C * u(t)
    %
    %   is encoded in the eqn structure
    %
    %   The fieldnames for A and E have to end with _  to indicate that the data
    %   are inputdata for the algorithm. Further A_ and E_ have to be
    %   substructured as given below.
    %
    %   eqn.A_ = [ A11 A12;     sparse (nf x nf) matrix A
    %              A21  0 ]
    %   eqn.E_ = [ E1  0;       sparse (nf x nf) matrix E
    %              0  0 ]
    %
    %   The sizes of A11 and E1 have to coincide and the value needs to
    %   be specified in eqn.manifold_dim. Also B has n = eqn.manifold_dim rows 
    %   and C n = eqn.manifold_dim columns.
    %
    %   Furthermore, A12 needs to have full column-rank and A21 full row-rank.
    %
    %   eqn.B       dense/sparse (n x m1) matrix B
    %
    %   eqn.C       dense/sparse (m2 x n) matrix C
    %
    %   eqn.R       dense symmetric and invertible (m1 x m1) matrix
    %               (required for LDL^T formulation)
    %
    %   eqn.Q       dense symmetric (m2 x m2) matrix
    %               (required for LDL^T formulation)
    %
    %   eqn.U       dense (n x m3) matrix U
    %               (required if eqn.V is present)
    %
    %   eqn.V       dense (n x m3) matrix V
    %               (required if eqn.U is present)
    %
    %   Depending on the operator chosen by the operatormanager, additional
    %   fields may be needed. For the "default", e.g., eqn.A_ and eqn.E_ hold
    %   the A and E matrices. For the second order ode types these are given
    %   implicitly by the M, D, K matrices stored in eqn.M_, eqn.E_ and eqn.K_,
    %   respectively.

    %
    % See also: struct2spmat, sparse


    %% Initialize the output struct for sparse matrices
    eqn = struct();
    
    %% Get the field names of the input struct
    fields = fieldnames(struct_data);
    % Iterate over each field in the input struct
    for i = 1:length(fields)
        field = fields{i};
        substruct = struct_data.(field);
        % Check if the substruct has the field 'type' and its value is 'sparse matrix'
        if isstruct(substruct) && isfield(substruct, 'type') && strcmp(substruct.type, 'sparse matrix')
            % Validate that the substruct contains all the necessary fields
            if isfield(substruct, 'data') && isfield(substruct, 'rows') && ...
               isfield(substruct, 'cols') && isfield(substruct, 'shape')
                % If validation passes, create the sparse matrix using struct2spmat
                eqn.(field) = struct2spmat(substruct);
            else
                % If any required field is missing, raise an error
                error(['Substruct "', field, '" marked as "sparse matrix" is missing one or more required fields: data, rows, cols, shape']);
            end
        else
            % If 'type' is not 'sparse matrix', copy the substruct as-is
            eqn.(field) = substruct;
        end
    end
    %%
    eqn.manifold_dim = size(eqn.B, 1);
    if eqn.manifold_dim ~= size(eqn.C, 2)
        error('Incompatible size of matrices B (rows) and C (cols).');
    end

end