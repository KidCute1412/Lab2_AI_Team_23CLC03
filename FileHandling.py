def readFile(file_path):
    """
    Reads a input file and returns a list of lists containing integers.
    """
    input_data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = list(map(int, line.split(',')))
            input_data.append(row)
    return input_data


def read_map():
    map =  readFile("input1.txt")
    islands = []
    size = len(map)
    for i in range(size):
        for j in range(size):
            if map[i][j] != 0:
                islands.append((i, j, map[i][j]))
    return size, islands

# def run_all_tests():
#     """
#     Runs all tests for the file handling module.
#     """
#     for i in range(1, 6):
#         file_path = f"input{i}.txt"
#         try:
#             grid_size, islands = read_map(file_path)
#             print(f"Test {i}: Grid size: {grid_size}, Islands: {islands}")
#         except Exception as e:
#             print(f"Test {i} failed with error: {e}")
        
    

# # Example usage
# grid_size, islands = read_map()
# for island in islands:
#     print(island)
