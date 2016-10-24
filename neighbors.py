"""neighbors contains the definition of neighbors."""

def neighbors(matrix, i, j):
    """
    neighbors returns the positions of the neighbors of cell (i,j)
    in the given matrix.

    These are always in the domain of the matrix and guaranteed
    to be valid positions.
    """
    directions = [
        # Verticals and horizontals.
        (0,  -1), (0,   1), (1,  0), (-1, 0),
        # Diagonals.
        (1,  -1), (-1, -1), (-1, 1), (1,  1)
    ]
    neighbors = []
    for direction in directions:
        # Check if horizontal directions are valid.
        if i + direction[0] >= len(matrix) or i + direction[0] < 0:
            continue
        # Check if vertical directions are valid.
        if j + direction[1] >= len(matrix) or j + direction[1] < 0:
            continue
        # If both horizontal and vertical directions are valid, add
        # this neighbor position.
        neighbors.append((i + direction[0], j + direction[1]))
    return neighbors

if __name__ == '__main__':
    matrix1 = [[0.25, 0.20, 0.25, 0.30], [0.20, 0.30, 0.25, 0.30], [0.25, 0.20, 0.40, 0.10], [0.30, 0.30, 0.10, 0.30]]
    matrix2 = [[0.1, 0.2, 0.3, 0.4], [0.9, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0.7, 0.6]]
    
    # In matrix 1, find the neighbors of cell (3, 2).
    print('MATRIX 1')
    for row in matrix1:
        print(row)
    print()
    print('neighbors positions')
    print(neighbors(matrix1, 3, 2), '\n')
    print('neighbors')
    for neighbor in neighbors(matrix1, 3, 2):
        print(matrix1[neighbor[0]][neighbor[1]])
    print()
    
    # In matrix 2, find the neighbors of cell (0, 0).
    print('MATRIX 2')
    for row2 in matrix2:
        print(row2)
    print()
    print('neighbors positions')
    print(neighbors(matrix2, 0, 0), '\n')
    print('neighbors')
    for element in neighbors(matrix2, 0, 0):
        print(matrix2[element[0]][element[1]])
