from drifter_transit import fix_dangling_node

if __name__ == '__main__':
    matrix1 = [[0.25, 0.20, 0.25, 0.0], [0.20, 0.30, 0.25, 0.0], [0.25, 0.20, 0.40, 0.0], [0.30, 0.30, 0.10, 1.0]]
    matrix2 = [[0.1, 0.2, 0.3, 0.0], [0.9, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.7, 1.0]]
    
    print('MATRIX 1:')
    for row in matrix1:
        print(row)

    print()

    print('MATRIX 2:')
    for row in matrix2:
        print(row)
  
    print()

    print('MATRIX 1 after fix:')
    fix_dangling_node(matrix1, 3, 3)
    for row in matrix1:
        print(row)

    print()
    
    print('MATRIX 2 after fix:')
    fix_dangling_node(matrix2, 3, 3)
    for row in matrix2:
        print(row)
