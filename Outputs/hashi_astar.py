from hashi_core import HashiwokakeroSolver
from hashi_core import dfs_check_connectivity
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
    

class HashiAStarSolver(HashiwokakeroSolver):
    def __init__(self, grid_size: int, islands: List[Tuple[int, int, int]]):
        super().__init__(grid_size, islands)
        self.clauses = self.generate_cnf_constraints()
        # Get all variables
        self.variable_ids = set(abs(v) for v in self.variables.values())
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
        unassigned = self.variable_ids - set(assignment.keys())

        if not unassigned:
            return None    #no unassigned variables left then return None
        #count how many clauses each variable appears in
        freq = {} #dictionary to store frequency of each variable
        for var in unassigned:
            count = 0
            for clause in self.clauses:
                if var in [abs(lit) for lit in clause]: #if variable appears in the clause
                    #check if clause is NOT YET SATISFIED 
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

    def solve_astar(self) -> Optional[Dict[int, bool]]:
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

            #check if already explored 
            state_key = frozenset(assignment.items()) 
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            #goal test: check if all clauses are satisfied
            if current_node.h_val == 0:
                solution = self._interpret_solution([var if val else -var for var, val in assignment.items()])
                if dfs_check_connectivity(self.islands[0][:2], set(), self.island_positions, solution['bridges']):
                    return solution  
                else:
                   self._add_clause_to_forbid_solution(solution)  
                   continue  
                
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
                    solution = self._interpret_solution([var if val else -var for var, val in assignment.items()])
                    if any(v > 0 for v in self.island_positions.values()) and not solution['bridges']:
                        continue
                    if dfs_check_connectivity(self.islands[0][:2], set(), self.island_positions, solution['bridges']):
                        return solution
                    else:
                        self._add_clause_to_forbid_solution(solution)
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
        return None





        

           
