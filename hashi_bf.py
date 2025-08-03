from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, product
from hashi_core import HashiwokakeroSolver
from typing import Optional
from itertools import product

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

class HashiBruteSolver(HashiwokakeroSolver):
    def solve_brute_force(self, max_iterations: int = 1000) -> Optional[Dict]:
        self.generate_cnf_constraints()
        num_vars = self.var_counter - 1
        tried = set()
        for _ in range(max_iterations):
            for assignment in product([False, True], repeat=num_vars):
                model = [i + 1 if val else -(i + 1) for i, val in enumerate(assignment)]
                if tuple(model) in tried:
                    continue
                if not self._satisfies_clauses(self.clauses, model):
                    continue
                solution = self._interpret_solution(model)
                start_island = self.islands[0][:2]
                visited = set()
                if dfs_check_connectivity(start_island, visited, self.island_positions, solution['bridges']):
                    return solution
                self._add_clause_to_forbid_solution(solution)
                tried.add(tuple(model))
                break
        return None