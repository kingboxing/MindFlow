from dolfin import *
import numpy as np

 # Define mesh
mesh = UnitSquareMesh(10, 10)
value=1.0
# Define mixed element (e.g., scalar and vector element)
P1 = FiniteElement('P', triangle, 1)  # Scalar element
P2 = VectorElement('P', triangle, 1)  # Vector element
mixed_element = MixedElement([P1, P2])



# Specify coordinates and subspace index
point_coordinates = (0.2, 0.8)
subspace_index = 0  # Apply to the first subspace (scalar component)

# Apply the point source
# Create the mixed function space
V = FunctionSpace(mesh, mixed_element)


# Create the function in the mixed space
u = Function(V)

dofs_coor = V.tabulate_dof_coordinates()#.reshape((-1, 2))
dofs_sub = V.sub(0).dofmap().dofs() # index of subsapce in dofs_coor 
vertex_coords = dofs_coor[dofs_sub, :]

# Convert the point coordinates to a Point object
point = Point(*point_coordinates)
closest_vertex_index = dofs_sub[np.argmin([point.distance(Point(*vertex)) for vertex in vertex_coords])]

coord_clo=dofs_coor[closest_vertex_index]
# Set the value at the closest vertex in the specified subspace
u.vector()[closest_vertex_index] = value





# Print the results for each subspace
print("Scalar component (subspace 0):", u.sub(0).vector().get_local())
print("Vector component (subspace 1):", u.sub(1).vector().get_local())

# Optionally, visualize the scalar and vector components
import matplotlib.pyplot as plt
plot(u.sub(0), title="Scalar component")
plt.show()