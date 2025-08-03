from main import HashiwokakeroSolver as hashi
from typing import List, Tuple, Dict, Set, Optional
import heapq as hq

class Node:
    def __init__(self, current_assignment, clauses, g_val: int):
        self.current_assignment = current_assignment # Biểu diễn gán giá trị của các biến
        self.g_val = g_val #số variable đã được gán 
        self.h_val = self.calculateHeuristic(current_assignment, clauses) #số mệnh đề chưa được thỏa mãn
        self.f_val = g_val + self.h_val

    def __eq__(self, other):
        return self.current_assignment == other.current_assignment
    
    def __lt__(self, other):
       return self.f_val < other.f_val
    def __hash__(self): #
        return hash(frozenset(self.current_assignment.items()))

    def calculateHeuristic(self, assignment, clauses):
        """Calculate heuristic: number of unsatisfied clauses."""
        count = 0
        for clause in clauses:
            # Check if clause is satisfied
            satisfied = False
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    # Positive literal and variable is True, or negative literal and variable is False
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                        satisfied = True
                        break
            
            if not satisfied:
                count += 1
        return count
    

class AStarSolver:
    def __init__(self, grid_size: int, islands: List[Tuple[int, int, int]]):
        self.grid_size = grid_size
        self.islands = islands
        self.cnf_converter = hashi(grid_size, islands)
        self.clauses = self.cnf_converter.generate_cnf_constraints()
        # Get all variables
        self.variables = set(abs(v) for v in self.cnf_converter.variables.values())
        # Start with empty assignment
        self.start_node = Node(current_assignment={}, clauses=self.clauses, g_val=0)
    
    def is_clause_unsatisfiable(self, clause, assignment):
        """Check if a clause is unsatisfiable given current assignment."""
        for lit in clause:
            var = abs(lit)
            if var not in assignment:
                return False  # Still has unassigned literals
            # If any literal is satisfied, clause is not unsatisfiable
            if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                return False
        return True  # All literals are assigned and false
    
    def has_conflict(self, assignment, clauses):
        """Check if current assignment leads to any unsatisfiable clauses."""
        for clause in clauses:
            if self.is_clause_unsatisfiable(clause, assignment):
                return True
        return False
        
    def get_unit_clauses(self, assignment):
        """Find unit clauses (clauses with only one unassigned literal)."""
        unit_assignments = {}
        
        for clause in self.clauses:
            unassigned_lits = []
            satisfied = False
            
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    # Check if this literal satisfies the clause
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                        satisfied = True
                        break
                else:
                    unassigned_lits.append(lit)
            
            if not satisfied and len(unassigned_lits) == 1:
                # Unit clause found
                lit = unassigned_lits[0]
                var = abs(lit)
                value = lit > 0
                
                if var in unit_assignments and unit_assignments[var] != value:
                    # Conflict: same variable assigned different values
                    return None
                unit_assignments[var] = value
        
        return unit_assignments
    
    def select_variable(self, assignment):
        """Select next variable using Most Constraining Variable heuristic."""
        unassigned = self.variables - set(assignment.keys())
        if not unassigned:
            return None
            
        # Count how many clauses each variable appears in
        var_frequency = {}
        for var in unassigned:
            count = 0
            for clause in self.clauses:
                if var in [abs(lit) for lit in clause]:
                    # Check if clause is not yet satisfied
                    satisfied = any(
                        (lit > 0 and assignment.get(abs(lit), False)) or 
                        (lit < 0 and not assignment.get(abs(lit), True))
                        for lit in clause if abs(lit) in assignment
                    )
                    if not satisfied:
                        count += 1
            var_frequency[var] = count
        # Return variable that appears in most unsatisfied clauses
        return max(unassigned, key=lambda v: var_frequency.get(v, 0))

    def solve(self) -> Optional[Dict[int, bool]]:
        """Solve using A* with constraint propagation."""
        print(f"Starting A* solver with {len(self.clauses)} clauses and {len(self.variables)} variables")
        print(f"Initial heuristic: {self.start_node.h_val}")
        
        open_list = []
        hq.heappush(open_list, self.start_node)
        closed_set = set()
        nodes_explored = 0
        max_nodes = 10000  # Prevent infinite search

        while open_list and nodes_explored < max_nodes:
            # A* Step 1: Select node with MINIMUM f-value
            current_node = hq.heappop(open_list)
            assignment = current_node.current_assignment.copy()
            nodes_explored += 1
            
            if nodes_explored % 100 == 0:  # More frequent updates
                print(f"Explored {nodes_explored} nodes, queue size: {len(open_list)}")
                print(f"Current node f-value: {current_node.f_val} (g={current_node.g_val}, h={current_node.h_val})")
                print(f"Current assignment size: {len(assignment)}")

            # Check if already explored this state (avoid cycles)
            state_key = frozenset(assignment.items())
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            # Goal test: check if all clauses are satisfied
            if current_node.h_val == 0:
                print(f"Solution found after exploring {nodes_explored} nodes!")
                return assignment

            # Apply unit propagation to current assignment
            unit_assignments = self.get_unit_clauses(assignment)
            if unit_assignments is None:
                # Conflict detected during unit propagation
                print(f"Conflict during unit propagation at node {nodes_explored}")
                continue
                
            if unit_assignments:
                print(f"Unit propagation found {len(unit_assignments)} forced assignments")
                assignment.update(unit_assignments)
            
            # Check for conflicts after unit propagation
            if self.has_conflict(assignment, self.clauses):
                print(f"Conflict after unit propagation at node {nodes_explored}")
                continue

            # Select next variable to assign
            next_var = self.select_variable(assignment)
            if next_var is None:
                print(f"No unassigned variables left at node {nodes_explored}")
                # Let's check if this is actually a solution
                final_h = Node(assignment, self.clauses, len(assignment)).h_val
                print(f"Final heuristic value: {final_h}")
                if final_h == 0:
                    print("Found solution with all variables assigned!")
                    return assignment
                continue

            print(f"Selected variable {next_var} for assignment")

            # A* Step 2: Generate successor nodes
            successors_added = 0
            for value in [True, False]:
                new_assignment = assignment.copy()
                new_assignment[next_var] = value
                
                # Early conflict detection (pruning)
                if not self.has_conflict(new_assignment, self.clauses):
                    # Create new node with correct g-value and h-value
                    new_g_val = len(new_assignment)  # Number of assigned variables
                    new_node = Node(new_assignment, self.clauses, new_g_val)
                    
                    print(f"  Adding successor: var{next_var}={value}, f={new_node.f_val} (g={new_node.g_val}, h={new_node.h_val})")
                    
                    # A* Step 3: Add to open list (heapq will maintain min-heap by f-value)
                    hq.heappush(open_list, new_node)
                    successors_added += 1
                else:
                    print(f"  Pruned successor: var{next_var}={value} (conflict)")
            
            if successors_added == 0:
                print(f"No valid successors generated at node {nodes_explored}")

        if nodes_explored >= max_nodes:
            print(f"Search terminated after {max_nodes} nodes (limit reached)")
        else:
            print(f"No solution found after exploring {nodes_explored} nodes.")
        return None


def read_map():
    map =  [[0 , 2 , 0 , 5 , 0 , 0 , 2],
            [0 , 0 , 0 , 0 , 0 , 0 , 0],
            [4 , 0 , 2 , 0 , 2 , 0 , 4],
            [0 , 0 , 0 , 0 , 0 , 0 , 0],
            [0 , 1 , 0 , 5 , 0 , 2 , 0],
            [0 , 0 , 0 , 0 , 0 , 0 , 0],
            [4 , 0 , 0 , 0 , 0 , 0 , 3]]
    islands = []
    size = len(map)
    for i in range(size):
        for j in range(size):
            if map[i][j] != 0:
                islands.append((i, j, map[i][j]))
    return size, islands

def convert_assignment_to_solution(cnf_solver: hashi, assignment: Dict[int, bool]) -> Dict:
    bridge_counts = {}
    bridges = []

    for (island1, island2, count), var in cnf_solver.variables.items():
        if isinstance(count, int) and assignment.get(var, False): 
            bridges.append({'from': island1, 'to': island2, 'count': count})
            for island in [island1, island2]:
                bridge_counts[island] = bridge_counts.get(island, 0) + count

    total = sum(b['count'] for b in bridges)
    return {
        'total_bridges': total,
        'bridges': bridges,
        'bridge_counts': bridge_counts
    } 
def create_simple_test():
    """Create a very simple test case that definitely has a solution."""
    # 3x3 grid with just 2 islands
    grid_size = 3
    islands = [
        (0, 0, 1),  # Island at (0,0) needs 1 bridge
        (0, 2, 1),  # Island at (0,2) needs 1 bridge  
    ]
    return grid_size, islands

def test_simple_case():
    """Test with a simple case first."""
    print("="*60)
    print("TESTING SIMPLE CASE")
    print("="*60)
    
    grid_size, islands = create_simple_test()
    print(f"Simple test: {len(islands)} islands")
    
    solver = AStarSolver(grid_size, islands)
    print(f"CNF has {len(solver.clauses)} clauses")
    print(f"Variables: {len(solver.variables)}")
    
    # Print some clauses for debugging
    print("\nFirst 10 clauses:")
    for i, clause in enumerate(solver.clauses[:10]):
        print(f"  {i+1}: {clause}")
    
    assignment = solver.solve()
    
    if assignment:
        print("A* found a solution for simple case!")
        cnf_solver = solver.cnf_converter
        solution = convert_assignment_to_solution(cnf_solver, assignment)
        cnf_solver.print_solution(solution)
        return True
    else:
        print("A* failed on simple case!")
        return False

if __name__ == "__main__":
    # Test simple case first
    simple_success = test_simple_case()
    
    if not simple_success:
        print("Stopping - simple case failed")
        exit(1)
    
    print("\n" + "="*60)
    print("TESTING COMPLEX CASE")
    print("="*60)
    
    grid_size, islands = read_map()
    
    solver = AStarSolver(grid_size, islands)
    assignment = solver.solve()

    if assignment:
        print("A* found a solution.")
        cnf_solver = solver.cnf_converter
        solution = convert_assignment_to_solution(cnf_solver, assignment)

        cnf_solver.print_solution(solution)
        cnf_solver.visualize_solution(solution)
    else:
        print("No solution found.")



        

           
