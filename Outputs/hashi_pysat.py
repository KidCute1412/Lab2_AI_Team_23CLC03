from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, product
from pysat.formula import CNF
from pysat.solvers import Glucose3
from hashi_core import HashiwokakeroSolver
from hashi_core import dfs_check_connectivity
from typing import Optional



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