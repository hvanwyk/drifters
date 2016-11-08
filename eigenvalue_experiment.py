import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
# This program takes two n x n transition matrices.

def eigenvalue_experiment(matrix1, matrix2, lambda_):
    largest_eigenvalues = []
    second_largest_eigenvalues = []
    eigenvectors = []

    for i in range(len(lambda_)):
        M_lambda = np.dot(lambda_[i], matrix2) + np.dot(1-lambda_[i], matrix1)
        w, v = la.eig(M_lambda)
        eigenvectors.append(v)
        largest = np.amax(w)
        largest_eigenvalues.append(abs(largest))
        w = np.delete(w, largest)
        second_largest = np.amax(w)
        second_largest_eigenvalues.append(abs(second_largest))

        # try removing all ones from second largest eigenvalues and plot only those
    return largest_eigenvalues, second_largest_eigenvalues, eigenvectors

if __name__ == '__main__':
    matrix1 = [[0.25, 0.20, 0.25, 0.30], [0.20, 0.30, 0.25, 0.30], [0.25, 0.20, 0.40, 0.10], [0.30, 0.30, 0.10, 0.30]]
    matrix2 = [[0.1, 0.2, 0.3, 0.4], [0.9, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0.7, 0.6]]
    lambda_ = np.linspace(0, 1, 50)
    largest, next_largest, eigenvectors = eigenvalue_experiment(matrix1, matrix2, lambda_)

    print('LARGEST EIGENVALUES:')
    for x in largest:
        print(x)
    
    print()
   
    print('SECOND LARGEST EIGENVALUES:')
    for y in next_largest:
        print(y)
   
    print()

    bins = np.linspace(0, 1, 10)
    digitized = np.digitize(eigenvectors, bins)

    plt.title('Largest & Second Largest Eigenvalues')
    plt.plot(largest, 'b-', label='largest eigenvalues')
    plt.plot(next_largest, 'm-', label='second largest eigenvalues')
    plt.legend(loc='best')
    plt.show()
    for element in digitized:
        plt.plot(element, 'g-')
    plt.title('Eigenvectors in Bins')
    plt.show()

    plt.title('Eigenvectors')
    for eigenvector in eigenvectors:
        plt.plot(eigenvector, 'r-', label='eigenvectors')
    plt.show()
