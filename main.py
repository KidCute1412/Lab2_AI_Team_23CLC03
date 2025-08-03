import HashiwokakeroSolver as hashi  
import FileHandling as util
import Astar as Astar

def hashi_solver(py_solver: hashi.HashiwokakeroSolver):
    if hashi.PYSAT_AVAILABLE:
        # First try the regular solver

        solution = py_solver.solve_with_connectivity_check(max_iterations=20)

        if solution:
            # Print the solution
            # solver.print_solution(solution)
            py_solver.visualize_solution(solution)
    else:
        print("PySAT not available. To solve:")
        print("1. Install PySAT: pip install python-sat")
        print("2. Run the program again")

def main():
    """Main function to demonstrate the CNF constraint generation and solving."""
    # print("Hashiwokakero CNF Constraint Generator & Solver")
    # print("=" * 60)
    
    # Create sample puzzle
    # grid_size, islands = create_sample_puzzle()
    grid_size, islands = util.read_map()
    
    # Initialize solver
    solver = hashi.HashiwokakeroSolver(grid_size, islands)
    hashi_solver(solver)
    # Generate CNF constraints
    # clauses = solver.generate_cnf_constraints()

    
    # print(f"\nGenerated {len(clauses)} CNF clauses")
    # print("\n10 first clauses:")
    # for i, clause in enumerate(clauses):
    #     if i >= 10:
    #         break
    #     print(f"  Clause {i+1}: {clause}")
    
    # # Try to solve with PySAT
    # print("\n" + "="*60)
    # print("SOLVING WITH PYSAT + CONNECTIVITY CHECK")
    # print("="*60)
    







if __name__ == "__main__":
    main()