from utils import read_edges, get_path
from collections import deque


def dfs(start, end):
    """
    dfs is done with stack.
    In each time, we update stack when we find neighbors of node now
    , putting neighbor v into stack
    Then we take out the arbitrary neighbor of now from stack
    until we find destination

    """
    # Begin your code (Part 2)
    graph = read_edges()
    stack = deque()
    open, close, last = {}, {}, {}

    now, dfs_visited = start, 0
    open[now], open[start]["dist"] = {}, 0

    while now != end:
        dfs_visited += 1
        update(now, graph, stack, open, close, last)
        close[now] = open[now]
        now = stack.pop()

    dfs_path = get_path(last, start, end)
    dfs_dist = open[end]['dist']

    return dfs_path, dfs_dist, dfs_visited
    # End your code (Part 2)


def update(now, graph, stack, open, close, last):
    if now not in graph:
        return

    for v, dist, _ in graph[now]:
        if v in close:
            continue
        elif v not in open:
            open[v] = {}

        open[v]["dist"] = open[now]["dist"] + dist
        last[v] = now
        stack.append(v)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
