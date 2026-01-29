grid_size = 3

DEBUG = False

matrices = []

def int_to_binary(n):
    binary = ''
    while n > 0:
        binary = str(n % 2) + binary
        n >>= 1
    return binary

def binary_to_int(binary_str):
    result = 0
    for bit in binary_str:
        result = (result << 1) | int(bit)
    return result

def rotate(matrix, grid_size=grid_size, times=1):
    temp = matrix.copy()
    new_matrix = matrix.copy()
    for _ in range(times):
        for i in range(grid_size):
            for j in range(grid_size):
                new_matrix[(grid_size - j - 1) * grid_size + i] = temp[i * grid_size + j]
        temp = new_matrix.copy()

    return new_matrix

def are_edges_used(matrix, grid_size=grid_size):
    ups = False
    downs = False
    lefts = False
    rights = False
    for i in [0]:
        for j in range(grid_size):
            if matrix[i * grid_size + j] == 1:
                ups = True
            if matrix[(grid_size - i - 1) * grid_size + j] == 1:
                downs = True
            if matrix[j * grid_size + i] == 1:
                lefts = True
            if matrix[j * grid_size + (grid_size - i - 1)] == 1:
                rights = True
    return ups and downs and lefts and rights    

def is_there_something_in_the_middle(matrix, grid_size=grid_size):
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            if matrix[i * grid_size + j] == 1:
                return True
    return False

def print_matrix(matrix, seq=0):
    if seq:
        for element in matrix:
            print(element, end=' ')
    else:
        for i in range(grid_size):
            for j in range(grid_size):
                print(matrix[i * grid_size + j], end=' ')
            print()
        

for i in range(1, 2 ** (grid_size ** 2)):
    binary = int_to_binary(i)
    binary = '0' * (grid_size ** 2 - len(binary)) + binary
    matrices.append([int(binary[j]) for j in range(grid_size ** 2)])

# for matrix in matrices:
#     print_matrix(matrix, 1)
#     print()
    
# for matrix in matrices:
#     print_matrix(matrix)
#     print()
    
if DEBUG:
    mat1 = matrices[65] 
    print_matrix(mat1, 1)
    print()
    print()
    print_matrix(mat1)
    print()
    print_matrix(rotate(mat1, times=1))
    print(f"This is at the index {matrices.index(rotate(mat1, times=1))} in the list of matrices")
    print()
    print_matrix(rotate(mat1, times=2))
    print(f"This is at the index {matrices.index(rotate(mat1, times=2))} in the list of matrices")
    print()
    print_matrix(rotate(mat1, times=3))
    print(f"This is at the index {matrices.index(rotate(mat1, times=3))} in the list of matrices")

indices_to_be_removed = []
for i in range(len(matrices)):
    if i in indices_to_be_removed:
        continue
    if DEBUG: print(f"Checking for matrix {i}, {matrices[i]}")
    if not are_edges_used(matrices[i]): #or not is_there_something_in_the_middle(matrices[i]):
    # if not are_edges_used(matrices[i]) or not is_there_something_in_the_middle(matrices[i]):
        if DEBUG: print(f"\tMatrix {i} has no edges used")
        indices_to_be_removed.append(i)
    for j in range(1, 4):
        dup = matrices.index(rotate(matrices[i], times=j))
        if dup != i and dup not in indices_to_be_removed:
            if DEBUG: print(f"\tMatrix {dup} is the same as matrix {i} rotated {j} times")
            indices_to_be_removed.append(dup)
            # break

print(indices_to_be_removed)

print(len(matrices))

# remove duplicates
for index in sorted(indices_to_be_removed, reverse=True):
    if DEBUG: print(f"Removing matrix {index}, total matrices left currently: {len(matrices)}")
    matrices.pop(index)
    if DEBUG: print(f"Total matrices left after removal: {len(matrices)}")
    
print(f"Total proper matrices we have: {len(matrices)}")

matrices_as_int = [binary_to_int(''.join(map(str, matrix))) for matrix in matrices]

for matrix in [42, 32]:
    print(matrix)
    print_matrix(matrices[matrix], 1)
    print()
    print_matrix(matrices[matrix])
    print()
    print(binary_to_int(''.join(map(str, matrices[matrix]))))
    
import json

with open('matrices.json', 'w') as f:
    json.dump(matrices_as_int, f, indent=2)