"""
Hashiwokakero (Bridges) Puzzle CNF Constraint Generator

This module implements CNF (Conjunctive Normal Form) constraints for the Hashiwokakero puzzle.
The puzzle involves connecting islands with bridges according to specific rules.
"""
import time
from hashi_core import HashiwokakeroSolver
from hashi_pysat import HashiSATSolver
from hashi_bf import HashiBruteSolver
from hashi_back import HashiBacktrackSolver

def read_map():
    map =  [[0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 2],
            [0, 0, 0, 0, 0],
            [4, 0, 3, 0, 0]]
    islands = []
    size = len(map)
    for i in range(size):
        for j in range(size):
            if map[i][j] != 0:
                islands.append((i, j, map[i][j]))
    return size, islands

def create_sample_puzzle():
    grid_size, islands = read_map()
    return grid_size, islands

def main():
    print("Hashiwokakero CNF Constraint Generator & Solver")
    print("=" * 60)

    grid_size, islands = create_sample_puzzle()

    print("\n" + "=" * 60)
    print("Choose a solving method:")
    print("1. PySAT")
    print("2. Brute-Force")
    print("3. Backtrack")
    print("=" * 60)

    choice = input("Enter your choice: ").strip()
    start_time = time.time()
    if choice == "1":
        print("\n" + "=" * 60)
        print("SOLVING WITH PYSAT + CONNECTIVITY CHECK")
        print("=" * 60)

        solver = HashiSATSolver(grid_size, islands)
        solution = solver.solve_with_connectivity_check(max_iterations=20)
        if solution:
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.4f} seconds")
            solver.print_solution(solution)
            solver.visualize_solution(solution)
        else:
            print("No valid solution found.")

    elif choice == "2":
        print("\n" + "=" * 60)
        print("SOLVING WITH BRUTE-FORCE")
        print("=" * 60)

        solver = HashiBruteSolver(grid_size, islands)
        solution = solver.solve_brute_force(max_iterations=1000)
        if solution:
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.4f} seconds")
            solver.print_solution(solution)
            solver.visualize_solution(solution)
            
        else:
            print("No valid solution found.")

    elif choice == "3":
        print("\n" + "=" * 60)
        print("SOLVING WITH BACKTRACK")
        print("=" * 60)

        solver = HashiBacktrackSolver(grid_size, islands)
        solution = solver.solve_backtracking()
        if solution:
            print(f"Number of steps taken: {solver.backtrack_counter}")
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.4f} seconds")
            solver.print_solution(solution)
            solver.visualize_solution(solution)
        else:
            print("No valid solution found.")

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()