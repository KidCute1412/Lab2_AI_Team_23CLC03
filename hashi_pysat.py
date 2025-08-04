from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, product
from pysat.formula import CNF
from pysat.solvers import Glucose3
from hashi_core import HashiwokakeroSolver
from typing import Optional

def dfs_check_connectivity(island: Tuple[int, int], visited: Set[Tuple[int, int]],
                          island_positions: Dict[Tuple[int, int], int],
                          bridges: List[Dict]) -> bool:
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
    return len(visited) == len(island_positions)

class HashiSATSolver(HashiwokakeroSolver):
    def solve_with_connectivity_check(self, max_iterations: int = 10) -> Optional[Dict]:
        self.generate_cnf_constraints()
        for _ in range(max_iterations):
            cnf = CNF()
            for clause in self.clauses:
                cnf.append(clause)
            solver = Glucose3()
            solver.append_formula(cnf)
            if not solver.solve():
                solver.delete()
                return None
            model = solver.get_model()
            solver.delete()
            solution = self._interpret_solution(model)
            start_island = self.islands[0][:2]
            visited = set()
            if dfs_check_connectivity(start_island, visited, self.island_positions, solution['bridges']):
                return solution
            self._add_clause_to_forbid_solution(solution)
        return None