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
        count = 0
        for clause in clauses:
            if not any((lit > 0 and assignment.get(abs(lit), False)) #literal dương và chưa được gán giá trị True
                       or (lit < 0 and not assignment.get(abs(lit), False))  #literal âm và chưa được gán giá trị False
                       for lit in clause):
                count += 1
        return count


class AStarSolver:
    def __init__(self, grid_size: int, islands: List[Tuple[int, int, int]]):
        self.grid_size = grid_size
        self.islands = islands
        self.cnf_converter = hashi(grid_size, islands)
        self.clauses = self.cnf_converter.generate_cnf_constraints()
        # Lấy tất cả biến (set of int)
        self.variables = set(abs(v) for v in self.cnf_converter.variables.values())
        # Node bắt đầu chưa gán gì
        self.start_node = Node(current_assignment={}, clauses=self.clauses, g_val=0)

    def solve(self) -> Optional[Dict[int, bool]]:
        open_list = []
        hq.heappush(open_list, self.start_node)
        closed_set = set()

        while open_list:
            current_node = hq.heappop(open_list)
            assignment = current_node.current_assignment

            # Nếu không còn clause nào chưa thỏa mãn thì return
            if current_node.h_val == 0:
                return assignment

            if current_node in closed_set:
                continue
            closed_set.add(current_node)

            assigned_vars = set(assignment.keys()) # Các biến đã được gán
            unassigned = self.variables - assigned_vars  # Các biến chưa được gán
            if not unassigned: # Nếu không còn biến nào chưa được gán
                continue

            next_var = min(unassigned)  # chọn biến nhỏ nhất để gán

            for value in [True, False]: 
                new_assignment = assignment.copy() #tạo bản sao để tránh thay đổi assignment gốc
                new_assignment[next_var] = value
                new_node = Node(new_assignment, self.clauses, g_val=current_node.g_val + 1)
                hq.heappush(open_list, new_node)

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
if __name__ == "__main__":
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



        

           
