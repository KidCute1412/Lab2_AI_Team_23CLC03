"""
Hashiwokakero (Bridges) Puzzle CNF Constraint Generator

This module implements CNF (Conjunctive Normal Form) constraints for the Hashiwokakero puzzle.
The puzzle involves connecting islands with bridges according to specific rules.
"""
import time

from hashi_pysat import HashiSATSolver
from hashi_bf import HashiBruteSolver
from hashi_back import HashiBacktrackSolver
from hashi_astar import HashiAStarSolver
import file_handling as fh

def run_solver(name, solver_class, grid_size, islands, solve_method):
    print(f"\n>>> Solving with {name}")
    solver = solver_class(grid_size, islands)
    start = time.time()
    solution = solve_method(solver)
    end = time.time() - start
    print(f"Time taken: {end:.4f} seconds")
    if solution:
        solver.visualize_solution(solution)
    else:
        print("No valid solution found.")


def main1():
    for i in range(1, 2):
        file_path = f"Inputs/input-{i:02}.txt"
        print(f"\nReading input from {file_path}")
        grid_size, islands = fh.read_map(file_path)
        # print(f"Grid size: {grid_size}, Islands: {len(islands)}")

        run_solver("PySAT", HashiSATSolver, grid_size, islands, lambda s: s.solve_with_connectivity_check(max_iterations=20))
        run_solver("Brute Force", HashiBruteSolver, grid_size, islands, lambda s: s.solve_brute_force(max_iterations=1000))
        run_solver("Backtrack", HashiBacktrackSolver, grid_size, islands, lambda s: s.solve_backtracking())
        run_solver("A*", HashiAStarSolver, grid_size, islands, lambda s: s.solve_astar())


    

def main2():  #chọn từng thuật toán để test
    print("Hashiwokakero CNF Constraint Generator & Solver")
    print("=" * 60)
    # Read input file
    list_of_files = []
    for i in range(1, 11):
        file_path = f"Inputs/input-{i:02}.txt"
        list_of_files.append(file_path)
    print("Available input files:")
    for idx, file in enumerate(list_of_files, start=1):
        print(f"input-{idx:02}.txt")
    file_choice = input("Enter the number of the file you want to use: ").strip()
    if file_choice.isdigit() and 1 <= int(file_choice) <= len(list_of_files):
        file_path = list_of_files[int(file_choice) - 1]
        grid_size, islands = fh.read_map(file_path)
    else:
        print("Invalid choice. Exiting.")
        return
    # print(f"Grid size: {grid_size}, Islands: {len(islands)}")
    # Choose solving method
    print("\n" + "=" * 60)
    print("Choose a solving method:")
    print("1. PySAT")
    print("2. Brute-Force")
    print("3. Backtrack")
    print("4. A*")
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
            # solver.print_solution(solution)
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
            # solver.print_solution(solution)
            solver.visualize_solution(solution)
        else:
            print("No valid solution found.")
    elif choice == "4":
        print("\n" + "=" * 60)
        print("SOLVING WITH A*")
        print("=" * 60)

        solver = HashiAStarSolver(grid_size, islands)
        solution = solver.solve_astar()
        if solution:
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.4f} seconds")
            # solver.print_solution(solution)
            solver.visualize_solution(solution)
        else:
            print("No valid solution found.")
    else:
        print("Invalid choice")

def main():
    print("Welcome to the Hashiwokakero CNF Constraint Generator & Solver")
    print("=" * 60)
    choice = input("Choose mode:\n1. Run all solvers on multiple inputs\n2. Choose a specific input and solver\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        main1()
    elif choice == "2":
        main2()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()