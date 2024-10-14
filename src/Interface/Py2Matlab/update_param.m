function updated_opts = update_param(opts, param)
    %UPDATE_PARAM Recursively updates parameters in the struct opts.
    %
    % This function takes an existing struct (opts) and updates it with the fields and
    % values from a new struct (param). The update happens recursively for nested structs.
    % Additionally, any field in the param struct that has the value 'default' will
    % cause that field to be deleted from the updated struct.
    %
    % Input:
    %   opts - The original struct with default parameters that needs to be updated.
    %   param - The struct containing the updated parameters from python.
    %
    % Output:
    %   updated_opts - The updated struct, after applying the changes from
    %                    param.
    %
    % Behavior:
    % - If a field in param has the value 'default', the corresponding field
    %   in existing_struct is removed (default value is handled inside the toolbox).
    % - If both opts and param have the same field, and the field
    %   contains nested structs, the update is applied recursively.
    % - If there is a type mismatch between corresponding fields, a warning is
    %   issued, and the field in opts is overwritten by the value from
    %   param.
    %
    % See also: fieldnames, isstruct, rmfield, class, warning
    
    % Get the field names of the new struct
    fields = fieldnames(param);

    % Iterate over each field in the new struct
    for i = 1:length(fields)
        field = fields{i};

        % If the new field value is 'default', remove the field from the existing struct
        if isequal(param.(field), 'default')
            if isfield(opts, field)
                opts = rmfield(opts, field);  % Remove the field
                disp(['Field "', field, '" removed due to "default" value.']);
            end
        else
            % Check if the field exists in the existing struct
            if isfield(opts, field)
                % Validate type compatibility
                if isstruct(opts.(field)) && isstruct(param.(field))
                    % If both fields are structs, update recursively
                    opts.(field) = update_param(opts.(field), param.(field));
                elseif isstruct(opts.(field)) ~= isstruct(param.(field))
                    % If one is struct and the other is not, raise a warning
                    warning(['Field "', field, '" type mismatch: one is a struct and the other is not. Overwriting the existing field.']);
                    opts.(field) = param.(field);  % Overwrite the field
                elseif ~isa(opts.(field), class(param.(field)))
                    % If they are of different types, raise a warning
                    warning(['Field "', field, '" type mismatch: "', class(opts.(field)), '" vs "', class(param.(field)), '". Overwriting the existing field.']);
                    opts.(field) = param.(field);  % Overwrite the field
                else
                    % If types match, simply update the field
                    opts.(field) = param.(field);
                end
            else
                % If the field doesn't exist, add it
                if isstruct(param.(field))
                    opts.(field) = struct();
                    opts.(field) = update_param(opts.(field), param.(field));
                else
                    opts.(field) = param.(field);
            end
        end
    end
    
    % Return the updated struct
    updated_opts = opts;

end