"""### graphs"""

# pip install pyvis

import networkx as nx

from pyvis.network import Network

from models.emb_model import get_most_similar_sentences


def get_graphs_and_cmp_sv(sentences_to_comp, rate):
    graph = nx.Graph()
    graph.add_nodes_from(sentences_to_comp)
    edjes = get_most_similar_sentences(sentences_to_comp, rate)
    graph.add_edges_from(edjes)

    dict_res = {
        "comp_sv": list(nx.connected_components(graph)),
        "amount_of_comp_sv": nx.number_connected_components(graph),
        "graph_nx": graph
    }

    net = Network(notebook=True, cdn_resources='remote')
    net.add_nodes(sentences_to_comp)
    net.add_edges(edjes)

    dict_res['graph_pyvis'] = net

    return dict_res


# pip install community
# pip install python-louvain

import community.community_louvain as lo


def get_clusters(graph_nx):
    partition = lo.best_partition(graph_nx)

    clusters = []

    for cluster_id in set(partition.values()):
        nodes = [nodes for nodes in partition.keys() if partition[nodes] == cluster_id]
        if (len(nodes) > 2):
            clusters.append(nodes)
    return clusters
