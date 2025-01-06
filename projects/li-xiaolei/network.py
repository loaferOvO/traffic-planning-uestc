import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
import numpy as np


def blue_noise_sampling(width, height, radius, k=30):
    """
    使用蓝噪声算法生成样本点。
    :param width: 平面宽度
    :param height: 平面高度
    :param radius: 点之间的最小距离
    :param k: 每个点的放置尝试次数
    :return: 生成的样本点列表
    """
    def in_bounds(point):
        return 0 <= point[0] < width and 0 <= point[1] < height

    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    points = []
    active_list = []

    # 随机生成初始点
    initial_point = (random.uniform(0, width), random.uniform(0, height))
    points.append(initial_point)
    active_list.append(initial_point)

    grid_x, grid_y = int(initial_point[0] / cell_size), int(initial_point[1] / cell_size)
    grid[grid_x][grid_y] = initial_point

    while active_list:
        current_point = random.choice(active_list)
        found = False
        for _ in range(k):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(radius, 2 * radius)
            new_point = (
                current_point[0] + distance * np.cos(angle),
                current_point[1] + distance * np.sin(angle)
            )
            if in_bounds(new_point):
                new_grid_x, new_grid_y = int(new_point[0] / cell_size), int(new_point[1] / cell_size)
                # 检查新点是否与周围点的距离足够远
                too_close = False
                for i in range(max(0, new_grid_x - 2), min(grid_width, new_grid_x + 3)):
                    for j in range(max(0, new_grid_y - 2), min(grid_height, new_grid_y + 3)):
                        neighbor = grid[i][j]
                        if neighbor and np.linalg.norm(np.array(new_point) - np.array(neighbor)) < radius:
                            too_close = True
                            break
                    if too_close:
                        break
                if not too_close:
                    points.append(new_point)
                    active_list.append(new_point)
                    grid[new_grid_x][new_grid_y] = new_point
                    found = True
                    break
        if not found:
            active_list.remove(current_point)

    return points


class TrafficNetwork:
    def __init__(self, num_nodes, num_edges, width=10, height=10, radius=2):
        """
        初始化网络，使用蓝噪声采样生成节点位置，并随机生成边。
        :param num_nodes: 网络中的节点数
        :param num_edges: 网络中的边数
        :param width: 平面宽度
        :param height: 平面高度
        :param radius: 蓝噪声采样的最小距离
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.width = width
        self.height = height
        self.radius = radius
        self.graph = self.generate_network()

    def generate_network(self):
        """
        使用蓝噪声采样生成网络的节点，并随机生成边。
        """
        G = nx.DiGraph()
        # 生成蓝噪声样本点
        points = blue_noise_sampling(self.width, self.height, self.radius)
        if len(points) < self.num_nodes:
            raise ValueError("无法生成足够的蓝噪声点，请调整参数！")
        points = points[:self.num_nodes]

        # 添加节点及其位置
        for i, point in enumerate(points):
            G.add_node(i, pos=point)

        # 随机连接边
        edges = set()
        while len(edges) < self.num_edges:
            u = random.choice(range(self.num_nodes))
            v = random.choice(range(self.num_nodes))
            if u != v and (u, v) not in edges:  # 避免自环和重复边
                weight = random.randint(1, 10)
                G.add_edge(u, v, weight=weight)
                edges.add((u, v))

        return G

    def draw_network(self):
        """
        绘制网络图。
        """
        pos = nx.get_node_attributes(self.graph, 'pos')
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color='lightblue')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.title("Traffic Network with Blue Noise Sampling")
        plt.show()

    def dijkstra_shortest_path(self, origin, destination):
        """
        使用 Dijkstra 算法计算两点之间的最短路径。
        :param origin: 起点
        :param destination: 终点
        :return: 最短路径和路径的总权重
        """
        try:
            path = nx.dijkstra_path(self.graph, source=origin, target=destination, weight='weight')
            cost = nx.dijkstra_path_length(self.graph, source=origin, target=destination, weight='weight')
            return path, cost
        except nx.NetworkXNoPath:
            return None, float('inf')
        
    def update_edge_weights(self, q_values):
        """
        更新边的权重，模拟流量对成本的影响。
        :param q_values: 边的流量字典
        """
        for edge in self.graph.edges:
            base_weight = self.graph.edges[edge]['weight']
            flow = q_values[edge]
            # 成本函数：基础成本 + 流量的非线性增长
            self.graph.edges[edge]['weight'] = base_weight * (1 + (flow / 1000) ** 2)


def update_edge_flow(q_values, network, origin, destination, base_flow):
    path, _ = network.dijkstra_shortest_path(origin, destination)
    for edge in q_values.keys():
        q_values[edge] *= 0.9
    if path:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            q_values[edge] += base_flow * 0.1  # 均匀分配流量

def visualize_traffic(network, od_pair, base_flow, frames=10):
    """
    可视化交通流量的更新过程。
    :param network: TrafficNetwork 对象
    :param od_pair: OD 对 (origin, destination)
    :param total_flow: 总流量
    :param frames: 动画帧数
    """
    origin, destination = od_pair
    pos = nx.get_node_attributes(network.graph, 'pos')  # 节点位置
    fig, ax = plt.subplots(figsize=(8, 6))

    q_values = {edge: 0 for edge in network.graph.edges}  # 初始化流量为 0

    def update(frame):
        """
        更新函数：每帧分配一部分流量，并更新可视化。
        """

        # 清除当前绘图
        ax.clear()

        # 分配流量
        update_edge_flow(q_values, network, origin, destination, base_flow)

        # 更新边的权重
        network.update_edge_weights(q_values)

        # 绘制网络
        edge_colors = [q_values[edge] for edge in network.graph.edges]
        nx.draw(network.graph, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_edges(
            network.graph, pos, edge_color=edge_colors, edge_cmap=plt.cm.Reds, width=2, ax=ax
        )

        # 显示流量值
        flow_labels = {edge: f"{q_values[edge]:.1f}" for edge in network.graph.edges}
        nx.draw_networkx_edge_labels(network.graph, pos, edge_labels=flow_labels, ax=ax)

        ax.set_title(f"Traffic Visualization (Frame {frame + 1}/{frames})")

    # 创建动画
    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()


if __name__ == "__main__":
    # 创建一个蓝噪声采样生成的随机网络
    num_nodes = 6
    num_edges = 10
    network = TrafficNetwork(num_nodes=num_nodes, num_edges=num_edges, width=8, height=6, radius=2)

    # 使用 Dijkstra 算法计算最短路径
    origin, destination = 0, 5
    path, cost = network.dijkstra_shortest_path(origin, destination)

    #初始化流量
    q_values = {edge: 0 for edge in network.graph.edges}  # 初始化流量为 0
    if path:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                q_values[edge] = 1000  # 均匀分配流量

    # 动态可视化流量分布
    base_flow = 1000  # 基础流量更新值
    visualize_traffic(network, od_pair=(origin, destination), base_flow=base_flow, frames=100)