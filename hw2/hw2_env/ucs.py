from utils import read_edges, get_path
import heapq


def ucs(start, end):
    # Begin your code (Part 3)
    """
    ufs is done with priority queue (pq)
    In each time, we update pq if dist of path to v < open[v]]["dist"]
    , putting neighbor v into pq
    Then we take out the node with smallest dist from pq
    until we find destination

    """
    graph = read_edges()
    pq = []
    open, close, last = {}, {}, {}

    now, ucs_visited = start, 0
    open[start], open[start]["dist"] = {}, 0

    while now != end:
        ucs_visited += 1
        if now not in close:
            update(now, graph, last, pq, open, close)
            close[now] = open[now]
        now = heapq.heappop(pq)[-1]

    ucs_path = get_path(last, start, end)
    ucs_dist = open[end]["dist"]

    return ucs_path, ucs_dist, ucs_visited
    # End your code (Part 3)


def update(now, graph, last, pq, open, close):
    if now not in graph:
        return

    for v, dist, _ in graph[now]:
        if v in close:
            continue
        elif v not in open:
            open[v] = {}
            open[v]["dist"] = float('inf')

        if open[now]["dist"] + dist < open[v]["dist"]:
            open[v]["dist"] = open[now]["dist"] + dist
            heapq.heappush(pq, (open[v]['dist'], v))
            last[v] = now


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
