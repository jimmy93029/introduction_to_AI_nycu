from utils import read_edges, get_heuristic, get_path, Nearest


def astar_time(start, end):
    # Begin your code (Part 6)
    """
    astar time is done with dictionary (open).
    In each time, we update open if dist of path to v < open[v]["g"]
    , putting neighbor v into open
    Then we take out the node with smallest f from open
    until we find destination
    Note that : we set heuristic function = 0
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

    time_path = get_path(last, start, end)
    time = open[end]["f"]
    time_visited = len(close) + 1

    return time_path, time, time_visited
    # End your code (Part 6)


def update(now, graph, last, h, open, close):
    if now not in graph:
        return

    g_now = open[now]["g"]
    for v, dist, speed_limit in graph[now]:
        if v in close:
            continue
        elif v not in open:
            open[v] = {}
            open[v]["g"] = float('inf')

        # set heuristic function = 0
        # h(x) = 0 is admissible because 0 smaller than travel time definitely
        if g_now + dist * 36 / (10 * speed_limit) < open[v]["g"]:
            open[v]["g"] = g_now + dist * 36 / (10 * speed_limit)  # turn km/hr to m/s
            open[v]["f"] = open[v]["g"] + 0
            last[v] = now


if __name__ == '__main__':
    path, time, num_visited = astar_time(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
