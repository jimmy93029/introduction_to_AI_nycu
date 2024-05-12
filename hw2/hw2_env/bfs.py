from utils import read_edges, get_path
import queue


def bfs(start, end):
    # Begin your code (Part 1)
    """
    bfs is done with queue.
    In each time, we update queue if nodes of path to v < open[v]]["nodes"]
    , putting neighbor v into queue
    Then we take out the nearest node from queue
    until we find destination

    """
    graph = read_edges()
    que = queue.Queue()
    open, last = {}, {}

    now, bfs_visited = start, 0
    open[now], open[start]["nodes"], open[start]["dist"] = {}, 0, 0

    while now != end:
        bfs_visited += 1
        update(now, graph, que, open, last)
        now = que.get()

    bfs_path = get_path(last, start, end)
    bfs_dist = open[end]["dist"]

    return bfs_path, bfs_dist, bfs_visited
    # End your code (Part 1)


def update(now, graph, que, open, last):
    if now not in graph:
        return

    nodes = open[now]["nodes"]
    dist_now = open[now]["dist"]
    for v, dist, _ in graph[now]:
        if v not in open:
            open[v] = {}
            open[v]["nodes"] = float('inf')

        if nodes + 1 < open[v]["nodes"]:
            open[v]["nodes"] = nodes + 1
            open[v]["dist"] = dist_now + dist
            last[v] = now
            que.put(v)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
