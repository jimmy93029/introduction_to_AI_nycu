import csv
edgeFile = 'edges.csv'
hFile = "heuristic.csv"


def read_edges():
    """
    read edges from edges.csv
    """
    graph = {}
    with open(edgeFile, "r") as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            u = int(row["start"])
            if u not in graph:
                graph[u] = []
            graph[u].append((int(row["end"]), float(row["distance"]), float(row["speed limit"])))

    return graph


def get_heuristic(end):
    """
    read edges from heuristic.csv
    """
    h = {}
    with open(hFile, "r") as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            id = int(row["node"])
            h[id] = float(row[f"{end}"])

    return h


def get_path(last, start, end):
    """
    get the path from start to end with dictionary last
    """
    reverse_path = []
    now = end
    while now != start:
        reverse_path.append(now)
        now = last[now]
    reverse_path.append(start)

    return reverse_path[::-1]


def Nearest(open):
    """
    get the node with smallest f
    """
    nearest = 0
    f_min, g_min = float('Inf'), float('Inf')
    for key, value in open.items():
        if value['f'] < f_min:
            f_min = value['f']
            g_min = value['g']
            nearest = key
        elif value['f'] == f_min and value['g'] < g_min:
            g_min = value['g']
            nearest = key

    return nearest
