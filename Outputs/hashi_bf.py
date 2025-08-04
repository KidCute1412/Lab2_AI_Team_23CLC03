from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, product
from hashi_core import HashiwokakeroSolver
from hashi_core import dfs_check_connectivity
from typing import Optional
from itertools import product


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