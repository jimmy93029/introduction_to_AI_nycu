from utils import read_edges, get_heuristic, get_path, Nearest


def astar(start, end):
    # Begin your code (Part 4)
    """
    astar is done with dictionary (open).
    In each time, we update open if dist of path to v < open[v]["g"]
    , putting neighbor v into open
    Then we take out the node with smallest f from open
    until we find destination

    """
    graph = read_edges()
    h = get_heuristic(end)
    open, close, last = {}, {}, {}

    now, open[start] = start, {}
    open[start]["f"], open[start]["g"] = h[start], 0

    while now != end:
        update(now, graph, last, h, open, close)
        close[now] = open[now]
        open.pop(now)
        now = Nearest(open)

    astar_path = get_path(last, start, end)
    astar_dist = open[end]["g"]
    astar_visited = len(close) + 1

    return astar_path, astar_dist, astar_visited
    # End your code (Part 4)


def update(now, graph, last, h, open, close):
    if now not in graph:
        return

    g_now = open[now]["g"]
    for v, dist, _ in graph[now]:
        if v in close:
            continue
        elif v not in open:
            open[v] = {}
            open[v]["g"] = float('inf')

        if dist + g_now < open[v]["g"]:
            open[v]["g"] = dist + g_now
            open[v]["f"] = open[v]["g"] + h[v]
            last[v] = now


if __name__ == '__main__':
    path, dist, num_visited = astar(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
