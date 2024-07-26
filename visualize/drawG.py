import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm  # 导入tqdm库

def draw_datasets(datasets, draw_config):
    if draw_config["test"]:
        traversed_father_domains = set([])  # 使用set类型来记录已经遍历过的each_graph.parentDomain
        # 假设datasets是包含每个对象each_graph的列表
        for each_graph in tqdm(datasets, desc="Drawing graphs"):  # 使用tqdm包装datasets来显示进度条
            # 检查each_graph.parentDomain是否已经遍历过
            if each_graph.parentDomain in traversed_father_domains:
                continue  # 如果已经遍历过，则跳过当前对象，继续下一个对象的遍历
            else:
                traversed_father_domains.add(each_graph.parentDomain)  # 将当前对象的parentDomain添加到已遍历set中
                draw_homo_from_het(each_graph, draw_config)  # 执行绘图函数
    else:
        for each_graph in tqdm(datasets, desc="Drawing graphs"):  # 使用tqdm包装datasets来显示进度条
            print(each_graph)
            draw_homo_from_het(each_graph, draw_config)  # 执行绘图函数


def draw_homo_from_het(data, draw_config):
    plt.clf()
    plt.rcParams['figure.figsize'] = draw_config["figsize"]
    # 将 PyG 异构图数据转换为 NetworkX 图对象
    G = nx.MultiDiGraph()

    # 添加节点和节点类型
    for each_type in data.node_types:
        for index, row in enumerate(data[each_type].x):
            G.add_node('[' + each_type + '] ' + str(index), type=each_type, attr=row, subset=each_type)

    for each_type in data.edge_types:
        each_relation = each_type[1]
        for i in range(data[each_type]["edge_index"][0].numel()):
            source = '[' + each_type[0] + '] ' + str(int(data[each_type]["edge_index"][0][i]))
            target = '[' + each_type[2] + '] ' + str(int(data[each_type]["edge_index"][1][i]))
            G.add_edge(source, target, type=each_relation, len=draw_config["edge_length"], width=draw_config["edge_width"], )
            
    # nodes_by_type = get_nodes_by_type(G.nodes)
    # edges_by_type = get_edges_by_type(G.nodes, G.edges)

    # 绘制 NetworkX 图
    if draw_config["layout"] == "spring":
        pos = nx.spring_layout(G, **draw_config["spring"]) # k越大越散，越小越聚集
    elif draw_config["layout"] == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, **draw_config["kamada_kawai"])
    elif draw_config["layout"] == "multipartite": # 最清晰
        pos = nx.multipartite_layout(G)


    for node_type, color in draw_config["node_colors"].items():
        nodes = [node for node, data in G.nodes(data=True) if data["type"] == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=250)

    for edge_type, color in draw_config["edge_colors"].items():
        edges = [
            (source, target)
            for source, target, data in G.edges(data=True)
            if data["type"] == edge_type
        ]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color)

    if draw_config["font_is_enabled"]:
        nx.draw_networkx_labels(G, pos, font_size=8, font_color=draw_config["font_color"])
    title = data.parentDomain + "/" + data.time.replace(":", "-")
    plt.title(title)
    
    if draw_config["show"]:
        plt.show()
    
    if draw_config["save"]:
        os.makedirs(draw_config["draw_path"] + data.parentDomain, exist_ok=True)
        plt.savefig(draw_config["draw_path"] + title + ".pdf", dpi=draw_config["dpi"], )

    plt.close()


def get_nodes_by_type(nodes):
    nodes_by_type = {}
    for node in nodes:
        node_type = nodes[node]['type']

        # 将节点添加到对应类型的节点列表中
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append(node)


def get_edges_by_type(nodes, edges):
    edges_by_type = {}
    for each_edge in edges:
        u, v = each_edge[0], each_edge[1]
        node_u_type = nodes[u]['type']
        node_v_type = nodes[v]['type']

        # 根据节点类型构建边的类型字符串
        edge_type = f"{node_u_type}-{node_v_type}"

        # 将边添加到对应类型的边列表中
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        edges_by_type[edge_type].append((u, v))
        

def get_round_pos(G): 
    # 定义节点的位置，按照类型分布在不同的圆周上
    pos = {}
    node_types = set(nx.get_node_attributes(G, 'type').values())
    num_types = len(node_types)
    radius = 2.0

    for i, node_type in enumerate(node_types):
        nodes_of_type = [node for node in G.nodes if G.nodes[node]['type'] == node_type]
        angle = 2 * i * np.pi / num_types
        for j, node in enumerate(nodes_of_type):
            # 添加随机扰动来调整节点位置，以避免节点完全重合
            r = radius + 0.1 * np.random.rand()
            theta = angle + 0.1 * np.random.rand() * np.pi
            pos[node] = (r * np.cos(theta), r * np.sin(theta))

    return pos
