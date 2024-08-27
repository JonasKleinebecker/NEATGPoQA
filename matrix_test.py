import numpy as np

unitary_matrix = np.eye(4)
unitary_matrix[[0,1,2,3]] = unitary_matrix[[1,0,3,2]]
print(unitary_matrix)