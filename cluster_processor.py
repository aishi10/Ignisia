from collections import defaultdict

def group_by_cluster(data):
    clusters = defaultdict(list)

    for row in data:
        cluster_key = (row["question_id"], row["cluster_id"])
        clusters[cluster_key].append(row)

    return clusters
