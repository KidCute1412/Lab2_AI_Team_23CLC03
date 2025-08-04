def read_file(file_path):
    """
    Reads a input file and returns a list of lists containing integers.
    """
    input_data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = list(map(int, line.split(',')))
            input_data.append(row)
    return input_data


def read_map(file_path):
    map =  read_file(file_path)
    islands = []
    size = len(map)
    for i in range(size):
        for j in range(size):
            if map[i][j] != 0:
                islands.append((i, j, map[i][j]))
    return size, islands

