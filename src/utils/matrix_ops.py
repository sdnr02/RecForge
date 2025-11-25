from typing import List, Dict, Tuple

def transpose_matrix(matrix: List[List[int]]) -> List[List[int]] | None:
    """Method to transpose a matrix"""
    # Ensuring the length and height of the matrix are greater than 0
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return None

    # Getting the height and length of the matrix
    height_of_matrix = len(matrix)
    length_of_matrix = len(matrix[0])

    # Initializing an empty matrix
    new_matrix = []
    for i in range(length_of_matrix):
        row = []
        for j in range(height_of_matrix):
            row.append(0)
        new_matrix.append(row)

    # Iterating through the old matrix
    for i in range(0, len(matrix)):
        # Iterating through length
        for j in range(0, len(matrix[0])):
            # Swapping the position of the matrix
            new_matrix[j][i] = matrix[i][j]

    # Returning the transposed matrix
    return new_matrix

def rotate_matrix_clockwise(matrix: List[List[int]]) -> List[List[int]] | None:
    """Method to rotate a matrix clockwise by 90 degrees"""
    # First step is to transpose the matrix
    transposed_matrix = transpose_matrix(matrix)

    # Checking that the transposed matrix exists
    if not transposed_matrix:
        return None
    
    # Reversing each individual list
    for i in range(0, len(transposed_matrix)):
        # Initializing the two pointers
        left = 0
        right = len(transposed_matrix[0]) - 1

        # Two pointer approach for reversing a list
        while left <= right:
            # Swapping the elements
            temporary_variable = transposed_matrix[i][left]
            transposed_matrix[i][left] = transposed_matrix[i][right]
            transposed_matrix[i][right] = temporary_variable

            # Incrementing and decrementing the pointers
            left = left + 1
            right = right - 1

    # Finally returning the matrix that is now rotated by 90
    return transposed_matrix

def spiral_matrix(matrix: List[List[int]]) -> List[int] | None:
    """Method to extract the elements of a list in spiral order"""
    # Ensuring the length and height of the matrix are greater than 0
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return None

    # Initializing variables for the boundaries of the matrix
    top = 0
    bottom = len(matrix) - 1
    left = 0
    right = len(matrix[0]) - 1

    # Initializing the result list
    result_list = []
    
    # Creating a loop until the base condition is satisfied
    while top <= bottom and left <= right:
        # Traversing right
        for column_variable in range(left, right+1):
            result_list.append(matrix[top][column_variable])
        # Now we reduce the boundary to remove the top row
        top = top + 1

        # Traversing down
        for row_variable in range(top, bottom+1):
            result_list.append(matrix[row_variable][right])
        # Now we reduce the boundary to remove the rightmost column
        right = right - 1

        # Checking that there is a row to traverse
        if top <= bottom:
            # Traversing left
            for column_variable in range(right, left-1, -1):
                result_list.append(matrix[bottom][column_variable])
            # Now we reduce the boundary to remove the bottom row
            bottom = bottom - 1

        # Checking that there is a column to traverse
        if left <= right:
            # Traversing up
            for row_variable in range(bottom, top-1, -1):
                result_list.append(matrix[row_variable][left])
            # Now we reduce the boundary to remove the leftmost column
            left = left + 1

    # Returning the final list with the added variables
    return result_list

def diagonal_traverse(matrix: List[List[int]]) -> List[int] | None:
    """Method to traverse a matrix with its diagonals that span from right to bottom moving from top to left"""
    # Ensuring the length and height of the matrix are greater than 0
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return None
    
    # Initializing the result list
    result_list = []
    
    # Initializing starting position
    row_index = 0
    column_index = 0
    
    # First Pass - Diagonals starting from top row
    while column_index < len(matrix[0]):
        # Initializing pointer variables
        i = row_index
        j = column_index
        
        # Checking to see if pointers are within bounds
        while i < len(matrix) and j >= 0:
            # Appending element to traversal
            result_list.append(matrix[i][j])

            # Incrementing and decrementing pointers
            i = i + 1
            j = j - 1
        
        # Incrementing column pointer
        column_index = column_index + 1
    
    # Initializing starting positions (skip top-left corner)
    row_index = 1
    column_index = len(matrix[0]) - 1
    
    # Second Pass - Diagonals starting from left column
    while row_index < len(matrix):
        # Initializing pointer variables
        i = row_index
        j = column_index
        
        # Checking to see if pointers are within bounds
        while i < len(matrix) and j >= 0:
            # Appending element to traversal
            result_list.append(matrix[i][j])
            
            # Incrementing and decrementing pointers
            i = i + 1
            j = j - 1
        
        # Incrementing row pointer
        row_index = row_index + 1
    
    # Returning the final result
    return result_list

def set_missing_to_zero(matrix: List[List[int]], flag_value: int = -1) -> List[List[int]]:
    """Method to set all of the flagged variable rows"""
    # Checking that the matrix exists
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return None
    
    # Getting the length and height of the matrix
    rows = len(matrix)
    columns = len(matrix[0])

    # Initializing empty lists to mark both rows and columns
    marked_rows = []
    marked_columns = []

    # Iteratign through the rows and the columns of the 2D matrix
    for i in range(0, rows):
        for j in range(0, columns):
            # Checking if the value of the matrix matches the flag value
            if matrix[i][j] == flag_value:
                # Adding the values to the marked rows
                if i not in marked_rows:
                    marked_rows.append(i)
                # Adding the values to the marked columns
                if j not in marked_columns:
                    marked_columns.append(j)
    
    # Turn the marked rows to zero
    for row in marked_rows:
        for j in range(0, columns):
            matrix[row][j] = 0
    
    # Turn the marked columns to zero
    for column in marked_columns:
        for i in range(0, rows):
            matrix[i][column] = 0

    # Returning the corrected matrix
    return matrix

def get_sparse_representation(matrix: List[List[int]]) -> Dict[Tuple[int, int], int]:
    """Convert dense matrix (storing all values including zeros) to sparse format (storing only non-zero values)"""
    # Checking that the matrix exists
    if len(matrix) <= 0 or len(matrix[0]) <= 0:
        return None

    # Creating a hash map for sparse representation of the matrix
    sparse_map = {}

    # Iterating through the rows and columns of the 2D matrix
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            if matrix[i][j] != 0:
                sparse_map[(i,j)] = matrix[i][j]

    # Return the sparse representation of the matrix
    return sparse_map