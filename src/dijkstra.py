import numpy as np

from heapq import heappop, heappush
from itertools import count
from sys import maxsize


def dijkstra(graph, origins, **kwargs):

    destinations = kwargs.get('destinations', [])
    objective = kwargs.get('objective', 'objective')
    return_paths = kwargs.get('return_paths', True)
    terminate_at_destinations = kwargs.get('terminate_at_destinations', True)
    maximum_cost = kwargs.get('maximum_cost', np.inf)


    nodes = graph._node
    edges = graph._adj

    costs = {} # dictionary of objective values for paths

    visited = {} # dictionary of costs-to-reach for nodes

    terminal = {k: True for k in graph.nodes}

    terminals = []

    if terminate_at_destinations:

        terminals = [d for d in destinations if d not in origins]

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        # Source is seen at the start of iteration and at 0 cost
        visited[origin] = np.inf

        heappush(heap, (0, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, source = heappop(heap)

        if source in costs:

            continue  # already searched this node.

        costs[source] = cost

        if source in terminals:

            continue

        for target, edge in edges[source].items():

            # Updating states for edge traversal
            cost_target = cost + edge.get(objective, 1)

            # Updating the weighted cost for the path
            savings = cost_target <= visited.get(target, np.inf)

            feasible = cost_target <= maximum_cost

            if savings & feasible:
               
                visited[target] = cost_target
                terminal[source] = False
                # terminal[target] = True

                heappush(heap, (cost_target, next(c), target))

    terminal = {k: terminal[k] for k in costs.keys()}

    return costs, terminal