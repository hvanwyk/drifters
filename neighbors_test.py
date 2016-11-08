from drifter_transit import neighbors

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
