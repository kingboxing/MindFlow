function S = struct2spmat(struct_data)

% Extract fields from the input struct
data = struct_data.data;
rows = struct_data.rows;
cols = struct_data.cols;
shape = struct_data.shape;

% Create the sparse matrix using the extracted data
S = sparse(rows, cols, data, shape(1), shape(2));

end