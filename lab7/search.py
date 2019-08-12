import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return '%'.join(modl)
    
    def convertHashtoState(key):
        st = key.split("%")
        a = []
        b = []
        for i in st:
            a.append(i[0:2])
            b.append(i[2:])
        return dict(zip(a,b))

    st = util.Stack()
    st.push(convertStateToHash(problem.getStartState()))
    while(not st.isEmpty()):
        ps = convertHashtoState(st.pop())
        # print(ps)
        if(problem.isGoalState(ps)):
            return ps
        for (a,b,c) in problem.getSuccessors(ps):
            # print(b)
            st.push(convertStateToHash(a))
    ## YOUR CODE HERE
    # util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.
    return util.points2distance(((problem.G.node[state]['x'],0,0),(problem.G.node[state]['y'],0,0)),((problem.G.node[problem.end_node]['x'],0,0),(problem.G.node[problem.end_node]['y'],0,0)))

def AStar_search(problem, heuristic=nullHeuristic):

    """Search the node that has the lowest combined cost and heuristic first."""
    def get_path(node):
        ans = []
        while(node != -1):
            ans.append(node.state)
            node = node.parent_node
        # ans.append(node.state)
        ans.reverse()
        print(len(ans))
        return ans
    
    visited = dict(((i,0) for i in problem.G.nodes))
    dist = dict(((i, float("inf")) for i in problem.G.nodes))
    dist[problem.getStartState()] = 0
    pq = util.PriorityQueue()
    pq.push(Node(problem.getStartState(), None, 0, -1, 0), heuristic(problem.getStartState(), problem))
    while(1):
        cur_nd = pq.pop()
        cur_st = cur_nd.state
        if(problem.isGoalState(cur_st)):
            return get_path(cur_nd)
        if (visited[cur_st] == 1):
            continue
        else:
            visited[cur_st] = 1
            for (a,b,c) in problem.getSuccessors(cur_st):
                # if (not(visited[a]) and (dist[a] < dist[cur_st] + c)):
                #     dist[a] = dist[cur_st] + c
                temp = Node(a, b, cur_nd.path_cost + c, cur_nd, cur_nd.depth + 1)
                pq.push(temp, temp.path_cost + heuristic(a, problem))