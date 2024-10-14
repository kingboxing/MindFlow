function struct_data = spmat2struct(S)

% Extract non-zero elements, row indices, and column indices from the sparse matrix
[rows, cols, data] = find(S);
% Get the size of the matrix
shape = size(S);

% Create a struct with the extracted data
struct_data.type = 'sparse matrix';
struct_data.format = 'csr';
struct_data.data = data;
struct_data.rows = rows;
struct_data.cols = cols;
struct_data.shape = shape;

end