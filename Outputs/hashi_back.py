from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, product
from hashi_core import HashiwokakeroSolver
from hashi_core import dfs_check_connectivity
from typing import Optional

class HashiBacktrackSolver(HashiwokakeroSolver):
    def solve_backtracking(self) -> Optional[Dict]:
        bridge_state = {b: 0 for b in self.possible_bridges}
        island_counts = { (r, c): 0 for r, c, _ in self.islands }

        def is_valid_bridge(i1, i2, count):
            for i in [i1, i2]:
                if island_counts[i] + count > self.island_positions[i]:
                    return False
            for other, c in bridge_state.items():
                if c > 0 and self._bridges_intersect((i1, i2), other):
                    return False
            return True

        def backtrack(index):
            self.backtrack_counter += 1
            if index == len(self.possible_bridges):
                for i, req in self.island_positions.items():
                    if island_counts[i] != req:
                        return None
                start = self.islands[0][:2]
                visited = set()
                bridges = [{'from': b[0], 'to': b[1], 'count': c} for b, c in bridge_state.items() if c > 0]
                if dfs_check_connectivity(start, visited, self.island_positions, bridges):
                    return {
                        'bridges': bridges,
                        'bridge_counts': island_counts.copy(),
                        'total_bridges': sum(bridge_state.values())
                    }
                return None

            i1, i2 = self.possible_bridges[index]
            for count in [0, 1, 2]:
                if count == 0 or is_valid_bridge(i1, i2, count):
                    bridge_state[(i1, i2)] = count
                    for i in [i1, i2]:
                        island_counts[i] += count
                    result = backtrack(index + 1)
                    if result:
                        return result
                    for i in [i1, i2]:
                        island_counts[i] -= count
                    bridge_state[(i1, i2)] = 0
            return None

        return backtrack(0)