"""
Hashiwokakero (Bridges) Puzzle CNF Constraint Generator

This module implements CNF (Conjunctive Normal Form) constraints for the Hashiwokakero puzzle.
The puzzle involves connecting islands with bridges according to specific rules.
"""

from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations
import math

# PySAT imports
try:
    from pysat.solvers import Glucose3
    from pysat.formula import CNF
    PYSAT_AVAILABLE = True
except ImportError:
    PYSAT_AVAILABLE = False
    print("Warning: PySAT not installed. Install with: pip install python-sat")


class HashiwokakeroSolver:
    def __init__(self, grid_size: int, islands: List[Tuple[int, int, int]]):
        """
        Initialize the Hashiwokakero solver.
        
        Args:
            grid_size: Size of the grid (assumed square)
            islands: List of (row, col, required_bridges) tuples
        """
        self.grid_size = grid_size
        self.islands = islands
        self.island_positions = {(row, col): required for row, col, required in islands}
        self.clauses = []
        self.variables = {} # (island1, island2, count) -> variable_number
        self.var_counter = 1
        # Generate all possible bridge connections
        self.possible_bridges = self._find_possible_bridges()
        
    def _find_possible_bridges(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find all possible bridge connections between islands."""
        bridges = []
        island_coords = [(row, col) for row, col, _ in self.islands]
        # NEED RECODE
        for i, (r1, c1) in enumerate(island_coords):
            for j, (r2, c2) in enumerate(island_coords):
                if i >= j:  # Avoid duplicates and self-connections
                    continue
                    
                # Check if islands are in the same row or column (perpendicular constraint)
                if r1 == r2 or c1 == c2:
                    # Check if path is clear (no other islands in between)
                    if self._is_path_clear((r1, c1), (r2, c2)):
                        bridges.append(((r1, c1), (r2, c2)))
        
        return bridges
    
    def _is_path_clear(self, island1: Tuple[int, int], island2: Tuple[int, int]) -> bool:
        """Check if path between two islands is clear of other islands."""
        r1, c1 = island1
        r2, c2 = island2
        
        if r1 == r2:  # Horizontal path
            start_col, end_col = min(c1, c2), max(c1, c2)
            for col in range(start_col + 1, end_col):
                if (r1, col) in self.island_positions:
                    return False
        else:  # Vertical path
            start_row, end_row = min(r1, r2), max(r1, r2)
            for row in range(start_row + 1, end_row):
                if (row, c1) in self.island_positions:
                    return False
        
        return True
    
    def _get_bridge_variable(self, island1: Tuple[int, int], island2: Tuple[int, int], count: int) -> int:
        """
        Get or create a variable for a bridge with specific count between two islands.
        
        Args:
            island1, island2: Coordinates of the islands
            count: Number of bridges (1 or 2)
        
        Returns:
            Variable number for this bridge configuration
        """
        # Ensure consistent ordering
        if island1 > island2:
            island1, island2 = island2, island1
            
        key = (island1, island2, count)
        if key not in self.variables:
            self.variables[key] = self.var_counter
            self.var_counter += 1
            
        return self.variables[key]
    
    def _get_bridge_exists_variable(self, island1: Tuple[int, int], island2: Tuple[int, int]) -> int:
        """Get variable indicating if any bridge exists between two islands."""
        if island1 > island2:
            island1, island2 = island2, island1
            
        key = (island1, island2, 'exists')
        if key not in self.variables:
            self.variables[key] = self.var_counter
            self.var_counter += 1
            
        return self.variables[key]
    
    def generate_cnf_constraints(self) -> List[List[int]]:
        """Generate all CNF constraints for the Hashiwokakero puzzle."""
        self.clauses = []
        
        # Constraint 1: At most two bridges between any pair of islands
        self._add_at_most_two_bridges_constraints()
        
        # Constraint 2: Bridge existence consistency
        self._add_bridge_existence_constraints()
        
        # Constraint 3: Exact bridge count for each island
        self._add_bridge_count_constraints()
        
        # Constraint 4: No crossing bridges
        self._add_no_crossing_constraints()
        
        # Constraint 5: Connectivity constraint (simplified - full connectivity is complex)
        self._add_basic_connectivity_constraints()
        
        return self.clauses
    
    def _add_at_most_two_bridges_constraints(self):
        """Constraint: At most two bridges connect any pair of islands."""
        for bridge in self.possible_bridges:
            island1, island2 = bridge
            
            # If there are 2 bridges, there cannot be 1 bridge and vice versa
            var_1 = self._get_bridge_variable(island1, island2, 1)
            var_2 = self._get_bridge_variable(island1, island2, 2)
            
            # At most one of {1 bridge, 2 bridges} can be true
            self.clauses.append([-var_1, -var_2])
    
    def _add_bridge_existence_constraints(self):
        """Constraint: Bridge existence is consistent with bridge counts."""
        for bridge in self.possible_bridges:
            island1, island2 = bridge
            
            var_exists = self._get_bridge_exists_variable(island1, island2)
            var_1 = self._get_bridge_variable(island1, island2, 1)
            var_2 = self._get_bridge_variable(island1, island2, 2)
            
            # If 1 or 2 bridges exist, then bridge exists
            self.clauses.append([-var_1, var_exists])
            self.clauses.append([-var_2, var_exists])
            
            # If bridge exists, then either 1 or 2 bridges
            self.clauses.append([-var_exists, var_1, var_2])
    
    def _add_bridge_count_constraints(self):
        """Constraint: Number of bridges connected to each island must match its number."""
        for row, col, required_bridges in self.islands:
            island = (row, col)
            
            # Find all bridges connected to this island
            connected_bridges = []
            for bridge in self.possible_bridges:
                island1, island2 = bridge
                if island == island1 or island == island2:
                    connected_bridges.append(bridge)
            
            # Generate constraints for exact count
            self._add_exact_count_constraint(island, connected_bridges, required_bridges)
    
    def _add_exact_count_constraint(self, island: Tuple[int, int], bridges: List[Tuple], required: int):
        """Add constraint that exactly 'required' bridges connect to island."""
        # Create weighted variables for bridge contributions
        # Each bridge connection can contribute 1 or 2 bridges
        weighted_vars = []
        
        for bridge in bridges:
            island1, island2 = bridge
            var_1 = self._get_bridge_variable(island1, island2, 1)  # Contributes 1
            var_2 = self._get_bridge_variable(island1, island2, 2)  # Contributes 2
            
            # Add with their weights: (variable, weight)
            weighted_vars.extend([(var_1, 1), (var_2, 2)])
        
        # Use weighted cardinality constraints
        self._add_weighted_cardinality_constraint(weighted_vars, required, island)
    
    def _add_weighted_cardinality_constraint(self, weighted_vars: List[Tuple[int, int]], target: int, island: Tuple[int, int]):
        """
        Add weighted cardinality constraint: exactly 'target' should be the sum of weights of true variables.
        
        Args:
            weighted_vars: List of (variable, weight) tuples
            target: Target sum
            island: Island for debugging
        """
        self._add_weighted_cardinality_enumeration(weighted_vars, target)

            
    def _add_weighted_cardinality_enumeration(self, weighted_vars: List[Tuple[int, int]], target: int):
        """Add weighted cardinality constraint by enumerating valid combinations."""
        n = len(weighted_vars)
        variables = [var for var, weight in weighted_vars]
        weights = [weight for var, weight in weighted_vars]
        
        # Generate all possible combinations of variable assignments
        for assignment in range(2**n):
            total_weight = 0
            selected_vars = []
            
            for i in range(n):
                if (assignment >> i) & 1:  # Variable i is true
                    total_weight += weights[i]
                    selected_vars.append(variables[i])
                else:  # Variable i is false
                    selected_vars.append(-variables[i])
            
            # If this assignment doesn't give the target weight, forbid it
            if total_weight != target:
                # Add clause that forbids this assignment
                clause = []
                for i in range(n):
                    if (assignment >> i) & 1:  # Variable i was true in this assignment
                        clause.append(-variables[i])  # So negate it in the clause
                    else:  # Variable i was false in this assignment
                        clause.append(variables[i])   # So assert it in the clause
                
                if clause:  # Only add non-empty clauses
                    self.clauses.append(clause)

    
    def _add_no_crossing_constraints(self):
        """Constraint: Bridges must not cross each other."""
        bridge_pairs = list(combinations(self.possible_bridges, 2))
        
        for bridge1, bridge2 in bridge_pairs:
            if self._bridges_intersect(bridge1, bridge2):
                # These bridges cannot both exist
                var1_exists = self._get_bridge_exists_variable(bridge1[0], bridge1[1])
                var2_exists = self._get_bridge_exists_variable(bridge2[0], bridge2[1])
                
                self.clauses.append([-var1_exists, -var2_exists])
    
    def _bridges_intersect(self, bridge1: Tuple[Tuple[int, int], Tuple[int, int]], 
                          bridge2: Tuple[Tuple[int, int], Tuple[int, int]]) -> bool:
        """Check if two bridges intersect."""
        (r1, c1), (r2, c2) = bridge1
        (r3, c3), (r4, c4) = bridge2
        
        # If bridges share an endpoint, they don't cross
        if (r1, c1) in [(r3, c3), (r4, c4)] or (r2, c2) in [(r3, c3), (r4, c4)]:
            return False
        
        # Check if one is horizontal and other is vertical
        bridge1_horizontal = (r1 == r2)
        bridge2_horizontal = (r3 == r4)
        
        if bridge1_horizontal and not bridge2_horizontal:
            # Bridge1 is horizontal, bridge2 is vertical
            return (min(c1, c2) < c3 < max(c1, c2) and 
                   min(r3, r4) < r1 < max(r3, r4))
        elif not bridge1_horizontal and bridge2_horizontal:
            # Bridge1 is vertical, bridge2 is horizontal
            return (min(r1, r2) < r3 < max(r1, r2) and 
                   min(c3, c4) < c1 < max(c3, c4))
        
        # Both horizontal or both vertical - they're parallel, no intersection
        return False
    
    def _add_basic_connectivity_constraints(self):
        """
        Add basic connectivity constraints.
        Note: Full connectivity is NP-hard to encode in CNF, so this is simplified.
        """
        # Ensure each island has at least one bridge if required_bridges > 0
        for row, col, required_bridges in self.islands:
            if required_bridges > 0:
                island = (row, col)
                connected_bridges = []
                
                for bridge in self.possible_bridges:
                    island1, island2 = bridge
                    if island == island1 or island == island2:
                        var_exists = self._get_bridge_exists_variable(island1, island2)
                        connected_bridges.append(var_exists)
                
                # At least one bridge must exist
                if connected_bridges:
                    self.clauses.append(connected_bridges)
    

    
    def _interpret_solution(self, model: List[int]) -> Dict:
        """
        Interpret the SAT model to extract bridge information.
        
        Args:
            model: List of positive/negative variable assignments
            
        Returns:
            Dictionary with solution details
        """
        solution = {
            'bridges': [],
            'bridge_counts': {},
            'total_bridges': 0
        }
        
        # Convert model to a set of true variables
        true_vars = set(abs(var) for var in model if var > 0)
        
        # Extract bridge information
        for key, var_num in self.variables.items():
            if var_num in true_vars:
                island1, island2, bridge_type = key
                
                if isinstance(bridge_type, int):  # Bridge count variables (1 or 2)
                    bridge_info = {
                        'from': island1,
                        'to': island2,
                        'count': bridge_type
                    }
                    solution['bridges'].append(bridge_info)
                    solution['total_bridges'] += bridge_type
                    
                    # Update bridge counts for each island
                    for island in [island1, island2]:
                        if island not in solution['bridge_counts']:
                            solution['bridge_counts'][island] = 0
                        solution['bridge_counts'][island] += bridge_type
        
        return solution
    
    def _add_clause_to_forbid_solution(self, solution: Dict):
        """
        Add a clause that forbids the current solution.
        This forces the SAT solver to find a different solution.
        """
        # Collect all the variables that are true in this solution
        true_bridge_vars = []
        
        for bridge in solution['bridges']:
            island1, island2 = bridge['from'], bridge['to']
            count = bridge['count']
            var = self._get_bridge_variable(island1, island2, count)
            true_bridge_vars.append(var)
        
        # Create a clause that prevents all these variables from being true simultaneously
        # At least one of them must be false in any new solution
        forbid_clause = [-var for var in true_bridge_vars]
        
        if forbid_clause:  # Only add if there are variables to forbid
            self.clauses.append(forbid_clause)
            print(f"Added clause to forbid disconnected solution: {forbid_clause}")
        
    def solve_with_connectivity_check(self, max_iterations: int = 10) -> Optional[Dict]:
        """
        Solve the CNF with iterative connectivity checking.
        If a solution is found but not connected, forbid it and try again.
        
        Args:
            max_iterations: Maximum number of iterations to try
            
        Returns:
            Dictionary with solution information, or None if no connected solution found
        """
        if not PYSAT_AVAILABLE:
            print("Error: PySAT not available. Install with: pip install python-sat")
            return None
        
        if not self.clauses:
            print("Error: No clauses generated. Call generate_cnf_constraints() first.")
            return None
        
        print("Solving with connectivity checking...")
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}:")
            
            # Create CNF formula with current clauses
            cnf = CNF()
            for clause in self.clauses:
                cnf.append(clause)
            
            # Solve using Glucose3 solver
            solver = Glucose3()
            solver.append_formula(cnf)
            
            # Check satisfiability
            is_satisfiable = solver.solve()
            
            if not is_satisfiable:
                solver.delete()
                # print("✗ No more solutions exist (UNSAT)")
                return None
            
            # Get the model and interpret solution
            model = solver.get_model()
            solver.delete()
            
            solution = self._interpret_solution(model)
            # print(f"Found solution with {solution['total_bridges']} bridges")
            
            # Check connectivity
            if len(self.islands) == 0:
                # print("✓ Empty puzzle - trivially connected")
                return solution
            
            # Use any island as starting point for DFS
            start_island = (self.islands[0][0], self.islands[0][1])
            visited = set()
            
            is_connected = dfs_check_connectivity(
                start_island, 
                visited, 
                self.island_positions, 
                solution['bridges']
            )
            
            if is_connected:
                # print("✓ Solution is connected!")
                return solution
            else:
                # print(f"✗ Solution is disconnected (visited {len(visited)}/{len(self.islands)} islands)")
                # print(f"  Visited islands: {sorted(visited)}")
                # print(f"  Missing islands: {sorted(set(self.island_positions.keys()) - visited)}")
                
                # Forbid this solution and try again
                self._add_clause_to_forbid_solution(solution)
                # print("  Added constraint to forbid this solution")
        
        print(f"\n✗ Could not find connected solution in {max_iterations} iterations")
        return None
    



    # def print_solution(self, solution: Dict):
    #     """Print the solution in a readable format."""
    #     if not solution:
    #         return
            
    #     print("\n" + "="*50)
    #     print("SOLUTION")
    #     print("="*50)
        
    #     # print(f"Total bridges built: {solution['total_bridges']}")
    #     # print(f"Number of bridge connections: {len(solution['bridges'])}")
        
    #     # print("\nBridge connections:")
    #     for i, bridge in enumerate(solution['bridges'], 1):
    #         count_str = "single" if bridge['count'] == 1 else "double"
            # print(f"  {i}. {bridge['from']} <--> {bridge['to']} ({count_str} bridge)")
        
        # print("\nBridge count per island:")
        # for island, count in solution['bridge_counts'].items():
        #     required = self.island_positions.get(island, 0)
        #     status = "✓" if count == required else "✗"
        #     # print(f"  {island}: {count}/{required} bridges {status}")
        
        # # Check if solution satisfies all requirements
        # all_satisfied = all(
        #     solution['bridge_counts'].get(island, 0) == required
        #     for island, required in self.island_positions.items()
        # )
        
        # print(f"\nSolution status: {'✓ VALID' if all_satisfied else '✗ INVALID'}")
    
    def visualize_solution(self, solution: Dict):
        """Create a simple text visualization of the solution."""
        if not solution:
            return
            
        print("\n" + "="*50)
        # print("GRID VISUALIZATION")
        print("="*50)
        
        # Create empty grid
        grid = [['0' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place islands
        for row, col, required in self.islands:
            grid[row][col] = str(required)
        
        # Add bridge representations (simplified)
        for bridge in solution['bridges']:
            (r1, c1), (r2, c2) = bridge['from'], bridge['to']
            count = bridge['count']
            
            if r1 == r2:  # Horizontal bridge
                start_col, end_col = min(c1, c2), max(c1, c2)
                for col in range(start_col + 1, end_col):
                    grid[r1][col] = '=' if count == 2 else '-'
            else:  # Vertical bridge
                start_row, end_row = min(r1, r2), max(r1, r2)
                for row in range(start_row + 1, end_row):
                    grid[row][c1] = '$' if count == 2 else '|'
        
        # Print grid
        # print("Legend: Numbers=Islands, -/|=Single bridge, =/$=Double bridge")
        print()
        for row in grid:
            print(' '.join(row))


def dfs_check_connectivity(island: Tuple[int, int], visited: Set[Tuple[int, int]],
                          island_positions: Dict[Tuple[int, int], int],
                          bridges: List[Dict]) -> bool:
    """Perform DFS to check if all islands are connected via bridges.
    Args:
        island: Starting island position (row, col)
        visited: Set of visited islands
        island_positions: Dictionary of island positions and their required bridges
        bridges: List of bridge dictionaries with 'from', 'to', 'count' keys
    Returns:
        True if all islands are reachable, False otherwise
    """
    # Build adjacency list from bridges
    adjacency = {}
    for pos in island_positions:
        adjacency[pos] = []
    
    for bridge in bridges:
        from_island = bridge['from']
        to_island = bridge['to']
        if from_island in adjacency and to_island in adjacency:
            adjacency[from_island].append(to_island)
            adjacency[to_island].append(from_island)
    
    # DFS traversal
    stack = [island]
    visited.clear()
    
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        
        # Add all connected neighbors
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                stack.append(neighbor)
    
    # Check if all islands were visited
    return len(visited) == len(island_positions)
# def read_map():
#     map =  [[0 , 2 , 0 , 5 , 0 , 0 , 2],
#             [0 , 0 , 0 , 0 , 0 , 0 , 0],
#             [4 , 0 , 2 , 0 , 2 , 0 , 4],
#             [0 , 0 , 0 , 0 , 0 , 0 , 0],
#             [0 , 1 , 0 , 5 , 0 , 2 , 0],
#             [0 , 0 , 0 , 0 , 0 , 0 , 0],
#             [4 , 0 , 0 , 0 , 0 , 0 , 3]]
#     islands = []
#     size = len(map)
#     for i in range(size):
#         for j in range(size):
#             if map[i][j] != 0:
#                 islands.append((i, j, map[i][j]))
#     return size, islands


# def create_sample_puzzle():
#     """Create a sample Hashiwokakero puzzle for testing."""
#     # 5x5 grid with some islands
#     grid_size, islands = read_map()
#     return grid_size, islands


# def run():
#     """Main function to demonstrate the CNF constraint generation and solving."""
#     print("Hashiwokakero CNF Constraint Generator & Solver")
#     print("=" * 60)
    
#     # Create sample puzzle
#     # grid_size, islands = create_sample_puzzle()
    
#     # Initialize solver
#     solver = HashiwokakeroSolver(grid_size, islands)
    
#     # Generate CNF constraints
#     # clauses = solver.generate_cnf_constraints()

    
#     # print(f"\nGenerated {len(clauses)} CNF clauses")
#     # print("\n10 first clauses:")
#     # for i, clause in enumerate(clauses):
#     #     if i >= 10:
#     #         break
#     #     print(f"  Clause {i+1}: {clause}")
    
#     # # Try to solve with PySAT
#     # print("\n" + "="*60)
#     # print("SOLVING WITH PYSAT + CONNECTIVITY CHECK")
#     # print("="*60)
    
#     if PYSAT_AVAILABLE:
#         # First try the regular solver

#         solution = solver.solve_with_connectivity_check(max_iterations=20)

#         if solution:
#             # Print the solution
#             # solver.print_solution(solution)
#             solver.visualize_solution(solution)
#     else:
#         print("PySAT not available. To solve:")
#         print("1. Install PySAT: pip install python-sat")
#         print("2. Run the program again")






# if __name__ == "__main__":
#     main()