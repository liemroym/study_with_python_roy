class Edge():
    def __init__(self, node1, node2, value) -> None:
        self.nodes = [node1, node2]
        self.weight = value


def djikstra(graph : list[list[Edge]], start, end):
    nodes = []
    i = 0
    for edges in graph: 
        nodes.append((edges[0].nodes[0], i))
        i += 1

    weight = [0]*len(nodes)
    
    curr_node = start
    while (curr_node != end):
         
        for edge in graph[curr_node[1]]:
            weight[]

if __name__ == "__main__":
    djikstra([[Edge('A', 'B', 4), Edge('A', 'C', 2)], 
              [Edge('B', 'C', 1), Edge('B', 'E', 5)],
              [Edge('C', 'D', 8), Edge('C', 'E', 10)],
              [Edge('D', 'E', 2), Edge('D', 'F', 1)],
              [Edge('E', 'F', 4)]],
              
              'A', 'F')