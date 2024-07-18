"""### graphs"""

# pip install pyvis

import networkx as nx
from networkx import Graph

from pyvis.network import Network

from app_dir.models.emb_model import get_most_similar_sentences


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


def get_clusters_and_colorized_graph(graph_nx: Graph):
    partition = lo.best_partition(graph_nx)

    color_map = []
    for node in graph_nx:
        if node < 'р':
            color_map.append('blue')
        else:
            color_map.append('green')
    nx.draw(graph_nx, node_color=color_map, with_labels=True)

    clusters = []
    for cluster_id in set(partition.values()):
        nodes = [nodes for nodes in partition.keys() if partition[nodes] == cluster_id]

        # min amount of nodes in cluster is 2
        if len(nodes) > 1:
            clusters.append(nodes)
    return (clusters, graph_nx)
