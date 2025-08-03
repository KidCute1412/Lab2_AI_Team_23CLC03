from main import HashiwokakeroSolver as hashi
from typing import List, Tuple, Dict, Set, Optional
import heapq as hq

class Node:
    def __init__(self, current_assignment, clauses, g_val: int):
        self.current_assignment = current_assignment #current variable assignment
        self.g_val = g_val #number of variables assigned so far
        self.h_val = self.calculateHeuristic(current_assignment, clauses) #number of unsatisfied clauses
        self.f_val = g_val + self.h_val

    def __eq__(self, other):
        return self.current_assignment == other.current_assignment
    
    def __lt__(self, other):
       return self.f_val < other.f_val
    def __hash__(self): #
        return hash(frozenset(self.current_assignment.items()))

    def calculateHeuristic(self, assignment, clauses):
        """Calculate heuristic: number of unsatisfied clauses in the current assignment."""
        count = 0
        for clause in clauses: #iterate through each clause
            #check if clause is satisfied
            satisfied = False #a bool indicating if the clause is satisfied
            for lit in clause: #iterate through each literal in the clause
                var = abs(lit) #get the variable from the literal, abs is used to get the variable number
                if var in assignment: #check if variable is assigned
                    #positive literal and variable is True, or negative literal and variable is False
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]): #as long as one literal is satisfied, the clause is satisfied
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
    
    def isUnsatisfiable(self, clause, assignment):
        """Check if a clause is unsatisfiable given current assignment."""
        for lit in clause:  #iterate through each literal in the clause
            var = abs(lit) #get the variable from the literal, abs is used to get the variable number
            if var not in assignment:
                return False  #still has unassigned literals
            if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                return False #clause is satisfied by this literal
        return True  
    
    def hasConflict(self, assignment, clauses):
        """Check if current assignment leads to any unsatisfiable clauses."""
        for clause in clauses:
            if self.isUnsatisfiable(clause, assignment):
                return True
        return False
        
    def getUnitClauses(self, assignment):
        """Find unit clauses (clauses with only one unassigned literal)."""
        unit_assignments = {} #store unit assignments found during propagation
        for clause in self.clauses:
            unassigned = [lit for lit in clause if abs(lit) not in assignment]  #get unassigned literals
            if any((lit > 0 and assignment.get(abs(lit), False)) or (lit < 0 and not assignment.get(abs(lit), True)) for lit in clause):
                continue  #clause already satisfied, skip
            if len(unassigned) == 1:  #only one unassigned literal
                lit = unassigned[0]  #get the unassigned literal
                var, val = abs(lit), lit > 0  #get variable and its value (True for positive, False for negative)
                if var in unit_assignments and unit_assignments[var] != val:
                    return None  #conflict detected, variable already assigned a different value
                unit_assignments[var] = val  #add unit assignment             
        return unit_assignments
    
    def selectVariable(self, assignment):
        """Select next variable using Most Constraining Variable heuristic."""
        unassigned = self.variables - set(assignment.keys()) #get unassigned variables
        if not unassigned:
            return None    #no unassigned variables left then return None
        #count how many clauses each variable appears in
        freq = {} #dictionary to store frequency of each variable
        for var in unassigned:
            count = 0
            for clause in self.clauses:
                if var in [abs(lit) for lit in clause]: #if variable appears in the clause
                    #check if clause is not yet satisfied
                    satisfied = any(
                        (lit > 0 and assignment.get(abs(lit), False)) or 
                        (lit < 0 and not assignment.get(abs(lit), True))
                        for lit in clause if abs(lit) in assignment
                    )
                    if not satisfied:
                        count += 1
            freq[var] = count
        # Return variable that appears in most unsatisfied clauses
        return max(unassigned, key=lambda v: freq.get(v, 0))

    def solve(self) -> Optional[Dict[int, bool]]:
        """Solve using A* with constraint propagation."""        
        open_list = []
        hq.heappush(open_list, self.start_node)
        closed_set = set()
        nodes_explored = 0
        max_nodes = 10000  #prevent infinite search

        while open_list and nodes_explored < max_nodes: 
            #select node with MINIMUM f-value
            current_node = hq.heappop(open_list)
            assignment = current_node.current_assignment.copy()
            nodes_explored += 1

            #check if already explored this state (avoid cycles)
            state_key = frozenset(assignment.items()) 
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            #goal test: check if all clauses are satisfied
            if current_node.h_val == 0:
                return assignment

            #apply unit propagation to current assignment
            unit_assignments = self.getUnitClauses(assignment)
            if unit_assignments is None: #conflict detected during unit propagation
                continue 
                
            if unit_assignments: #if there are unit clauses, apply them
                assignment.update(unit_assignments)
            
            #check for conflicts after unit propagation
            if self.hasConflict(assignment, self.clauses):
                continue

            #select next variable to assign
            next_var = self.selectVariable(assignment)
            if next_var is None:
                final_h = Node(assignment, self.clauses, len(assignment)).h_val
                if final_h == 0:
                    return assignment
                continue
            #generate successor nodes
            successors_added = 0
            for value in [True, False]:
                new_assignment = assignment.copy()
                new_assignment[next_var] = value
                
                if not self.hasConflict(new_assignment, self.clauses):
                    new_g_val = len(new_assignment)  #number of assigned variables
                    new_node = Node(new_assignment, self.clauses, new_g_val)                                    
                    hq.heappush(open_list, new_node)
                    successors_added += 1
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



        

           
