import torch
import numpy as np
from scipy.stats import multivariate_normal

from torch_geometric.utils import erdos_renyi_graph, dense_to_sparse


def generate_attribute_graph(num_nodes=3600, p=0.01, dim=32):
    edge_index = erdos_renyi_graph(num_nodes, p)

    num_class = 5
    num_per_class = int(num_nodes / num_class)
    
    cov_matrix = np.diag([1 for _ in range(dim)])

    centers, datas = [], []
    for i in range(num_class):
        centers.append(2.5 * np.random.random(size=dim) - 1)
        datas.append(multivariate_normal.rvs(centers[i], cov_matrix, num_per_class))

    data = np.vstack(datas)

    label = np.hstack([np.zeros([num_per_class]), np.ones([num_per_class]), np.ones([num_per_class]) + 1, np.ones([num_per_class]) + 2, np.ones([num_per_class]) + 3])

    permutation = np.random.permutation(label.shape[0])
    data = data[permutation, :]
    label = label[permutation]

    x = torch.from_numpy(data).float()
    y = torch.from_numpy(label).long()
    
    torch.save(edge_index, './data/synthetic/attribute/edge_index.pt')
    torch.save(x, './data/synthetic/attribute/x.pt')
    torch.save(y, './data/synthetic/attribute/y.pt')


def generate_topology_graph(num_nodes=3600, p=0.03, q=0.0001, dim=32):
    adj = np.zeros([num_nodes, num_nodes])
    
    num_communities = 5
    num_per_community = int(num_nodes / num_communities)

    community_index = []

    # intra-cluster
    for c in range(num_communities):
        community_index.append([c*num_per_community, (c + 1)*num_per_community])
        st, ed = community_index[c]
        for i in range(st, ed):
            for j in range(i + 1, ed):
                if np.random.random() < p:
                    adj[i, j] = adj[j, i] = 1
    
    # inter-cluster
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            st1, ed1 = community_index[c1]
            st2, ed2 = community_index[c2]
            
            for i in range(st1, ed1):
                for j in range(st2, ed2):
                    if np.random.random() < q:
                        adj[i, j] = adj[j, i] = 1
                
    adj = torch.from_numpy(adj).long()
    edge_index = dense_to_sparse(adj)[0]

    cov_matrix = np.diag([1 for _ in range(dim)])

    center = 2.5 * np.random.random(size=dim) - 1

    data = multivariate_normal.rvs(center, cov_matrix, num_nodes)
    
    label = np.hstack([np.zeros([num_per_community]), np.ones([num_per_community]), np.ones([num_per_community]) + 1, np.ones([num_per_community]) + 2, np.ones([num_per_community]) + 3])

    x = torch.from_numpy(data).float()
    y = torch.from_numpy(label).long()

    torch.save(edge_index, './data/sythetic/topology/edge_index.pt')
    torch.save(x, './data/synthetic/topology/x.pt')
    torch.save(y, './data/synthetic/topology/y.pt')
    
def generate_tl_40_graph():
    # TL-40: 2 classes related to topology and 3 classes related to attributes
    num_nodes=3600
    p=0.03
    q=0.0001
    dim=32

    adj = np.zeros([num_nodes, num_nodes])

    num_class = 5
    num_per_class = int(num_nodes / num_class)

    num_communities = 2
    num_per_community = int(num_nodes / num_communities)

    community_index = []

    # intra-cluster
    for c in range(num_communities):
        community_index.append([c*num_per_community, (c + 1)*num_per_community])
        st, ed = community_index[c]
        for i in range(st, ed):
            for j in range(i + 1, ed):
                if np.random.random() < p:
                    adj[i, j] = adj[j, i] = 1

    # inter-cluster
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            st1, ed1 = community_index[c1]
            st2, ed2 = community_index[c2]            
            for i in range(st1, ed1):
                for j in range(st2, ed2):
                    if np.random.random() < q:
                        adj[i, j] = adj[j, i] = 1
                
    adj = torch.from_numpy(adj).long()
    edge_index = dense_to_sparse(adj)[0]

    label = np.repeat(np.arange(num_communities), num_per_community)

    nums = int(num_per_class / num_communities)  # Each community takes nums of nodes
    dis1_index = np.hstack([np.arange(c[0], c[0] + nums) for c in community_index])  # The index of Distribution 1
    label[dis1_index] = num_class - 3  # Distribution 1 constitutes category 3

    dis2_index = np.hstack([np.arange(c[0]+nums, c[0] + nums*2) for c in community_index])  # The index of Distribution 2
    label[dis2_index] = num_class - 2  # Distribution 2 constitutes category 4

    dis3_index = np.hstack([np.arange(c[0]+nums*2, c[0] + nums*3) for c in community_index])  # The index of Distribution 3
    label[dis3_index] = num_class - 1  # Distribution 3 constitutes category 4

    all_index = np.arange(num_nodes)
    remain_index = all_index[~np.isin(all_index, np.hstack([dis1_index, dis2_index, dis3_index]))]  # Indexes outside of distribution 1 2 3

    cov_matrix = np.diag([1 for _ in range(dim)])

    center1 = 2.5 * np.random.random(size=dim) - 1
    center2 = 2.5 * np.random.random(size=dim) - 1
    center3 = 2.5 * np.random.random(size=dim) - 1
    center4 = 2.5 * np.random.random(size=dim) - 1

    data1 = multivariate_normal.rvs(center1, cov_matrix, num_per_class)
    data2 = multivariate_normal.rvs(center2, cov_matrix, num_per_class)
    data3 = multivariate_normal.rvs(center3, cov_matrix, num_per_class)
    data4 = multivariate_normal.rvs(center4, cov_matrix, num_nodes - num_per_class*3)
    data = np.zeros([num_nodes, dim])
    data[dis1_index] = data1
    data[dis2_index] = data2
    data[dis3_index] = data3
    data[remain_index] = data4

    x = torch.from_numpy(data).float()
    y = torch.from_numpy(label).long()

    torch.save(edge_index, './data/synthetic/tl-40/edge_index.pt')
    torch.save(x, './data/synthetic/tl-40/x.pt')
    torch.save(y, './data/synthetic/tl-40/y.pt')

def generate_tl_60_graph():
    #TL-60: 3 classes related to topology and 2 classes related to attributes
    num_nodes=3600
    p=0.03
    q=0.0001
    dim=32

    adj = np.zeros([num_nodes, num_nodes])

    num_class = 5
    num_per_class = int(num_nodes / num_class)

    num_communities = 3
    num_per_community = int(num_nodes / num_communities)

    community_index = []

    # intra-cluster
    for c in range(num_communities):
        community_index.append([c*num_per_community, (c + 1)*num_per_community])
        st, ed = community_index[c]
        for i in range(st, ed):
            for j in range(i + 1, ed):
                if np.random.random() < p:
                    adj[i, j] = adj[j, i] = 1

    # inter-cluster
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            st1, ed1 = community_index[c1]
            st2, ed2 = community_index[c2]
            
            for i in range(st1, ed1):
                for j in range(st2, ed2):
                    if np.random.random() < q:
                        adj[i, j] = adj[j, i] = 1
                
    adj = torch.from_numpy(adj).long()
    edge_index = dense_to_sparse(adj)[0]

    label = np.repeat(np.arange(num_communities), num_per_community)

    nums = int(num_per_class / num_communities)  
    dis1_index = np.hstack([np.arange(c[0], c[0] + nums) for c in community_index])  # The index of Distribution 1
    label[dis1_index] = num_class - 2  # Distribution 1 constitutes category 3

    dis2_index = np.hstack([np.arange(c[0]+nums, c[0] + nums*2) for c in community_index])  # The index of Distribution 2
    label[dis2_index] = num_class - 1  # Distribution 1 constitutes category 4

    all_index = np.arange(num_nodes)
    remain_index = all_index[~np.isin(all_index, np.hstack([dis1_index, dis2_index]))]  # Indexes outside of distribution 1 2

    cov_matrix = np.diag([1 for _ in range(dim)])

    center1 = 2.5 * np.random.random(size=dim) - 1
    center2 = 2.5 * np.random.random(size=dim) - 1
    center3 = 2.5 * np.random.random(size=dim) - 1

    data1 = multivariate_normal.rvs(center1, cov_matrix, num_per_class)
    data2 = multivariate_normal.rvs(center2, cov_matrix, num_per_class)
    data3 = multivariate_normal.rvs(center3, cov_matrix, num_nodes - num_per_class*2)
    data = np.zeros([num_nodes, dim])
    data[dis1_index] = data1
    data[dis2_index] = data2
    data[remain_index] = data3

    x = torch.from_numpy(data).float()
    y = torch.from_numpy(label).long()

    torch.save(edge_index, './data/sythetic/tl-60/edge_index.pt')
    torch.save(x, './data/sythetic/tl-60/x.pt')
    torch.save(y, './data/sythetic/tl-60/y.pt')


def generate_tl_80_graph():
    # TL-80: 4 classes related to topology and 1 classes related to attributes
    num_nodes=3600
    p=0.03
    q=0.0001
    dim=32

    adj = np.zeros([num_nodes, num_nodes])

    num_class = 5
    num_per_class = int(num_nodes / num_class)

    num_communities = 4
    num_per_community = int(num_nodes / num_communities)

    community_index = []

    # intra-cluster
    for c in range(num_communities):
        community_index.append([c*num_per_community, (c + 1)*num_per_community])
        st, ed = community_index[c]
        for i in range(st, ed):
            for j in range(i + 1, ed):
                if np.random.random() < p:
                    adj[i, j] = adj[j, i] = 1

    # inter-cluster
    for c1 in range(num_communities):
        for c2 in range(c1 + 1, num_communities):
            st1, ed1 = community_index[c1]
            st2, ed2 = community_index[c2]
            
            for i in range(st1, ed1):
                for j in range(st2, ed2):
                    if np.random.random() < q:
                        adj[i, j] = adj[j, i] = 1
                
    adj = torch.from_numpy(adj).long()
    edge_index = dense_to_sparse(adj)[0]

    label = np.repeat(np.arange(num_communities), num_per_community)

    nums = int(num_per_class / num_communities) 
    dis1_index = np.hstack([np.arange(c[0], c[0] + nums) for c in community_index])  # The index of Distribution 1
    label[dis1_index] = num_class - 1  # Distribution 1 constitutes category 5

    all_index = np.arange(num_nodes)
    remain_index = all_index[~np.isin(all_index, dis1_index)]  # Indexes outside of distribution 1

    cov_matrix = np.diag([1 for _ in range(dim)])

    center1 = 2.5 * np.random.random(size=dim) - 1
    center2 = 2.5 * np.random.random(size=dim) - 1

    data1 = multivariate_normal.rvs(center2, cov_matrix, num_per_class)
    data2 = multivariate_normal.rvs(center1, cov_matrix, num_nodes - num_per_class)
    data = np.zeros([num_nodes, dim])
    data[dis1_index] = data1
    data[remain_index] = data2

    x = torch.from_numpy(data).float()
    y = torch.from_numpy(label).long()

    torch.save(edge_index, './data/synthetic/tl-80/edge_index.pt')
    torch.save(x, './data/synthetic/tl-80/x.pt')
    torch.save(y, './data/synthetic/tl-80/y.pt')



generate_attribute_graph()
generate_tl_40_graph()
generate_tl_60_graph()
generate_tl_80_graph()
generate_topology_graph()
print("generate synthetic graph complete.")