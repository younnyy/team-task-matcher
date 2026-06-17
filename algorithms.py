# -*- coding: utf-8 -*-
"""
algorithms.py
图论大作业核心算法实现：
1. 匈牙利算法 (Hungarian Algorithm) - 用于求解无权偶图的最大匹配。
2. KM 算法 (Kuhn-Munkres Algorithm) - 用于求解带权偶图的最大权完美匹配。
这两个算法都包含了步进式的状态记录，方便前端进行可视化和求解过程分析。
"""

from collections import deque

def hungarian_maximum_matching_stepwise(left_nodes, right_nodes, edges):
    """
    无权偶图最大匹配（匈牙利算法）- 步进式实现
    :param left_nodes: 左侧顶点集合 X (例如学生姓名列表)
    :param right_nodes: 右侧顶点集合 Y (例如任务名称列表)
    :param edges: 边集，元素为元组 (u, v)，其中 u 属于 left_nodes, v 属于 right_nodes
    :return: (matching, steps)
             matching: 最终匹配字典 {right_node: left_node}
             steps: 详细步骤记录列表，每步是一个字典，包含当前状态信息
    """
    # 建立邻接表：算法只需要从 X 侧顶点快速找到可连接的 Y 侧顶点。
    adj = {u: [] for u in left_nodes}
    for u, v in edges:
        if u in adj:
            adj[u].append(v)
            
    # 匹配状态：match_y[v] = u 表示 Y 中的 v 与 X 中的 u 匹配；-1 表示未匹配
    match_y = {v: None for v in right_nodes}
    match_x = {u: None for u in left_nodes}
    
    steps = []
    
    # 辅助函数：记录当前状态的快照
    def record_step(log_msg, root, S=None, T=None, tree_edges=None, aug_path=None, found_path=False):
        steps.append({
            'matching': {k: v for k, v in match_y.items() if v is not None},
            'root': root,
            'S': list(S) if S else [],
            'T': list(T) if T else [],
            'tree_edges': list(tree_edges) if tree_edges else [],
            'log': log_msg,
            'augmenting_path': list(aug_path) if aug_path else [],
            'found_path': found_path
        })

    # 对 X 中的每个顶点，尝试寻找 M-可扩路
    for u in left_nodes:
        if match_x[u] is not None:
            continue
            
        record_step(f"【新阶段】开始为未匹配学生【{u}】寻找可行任务分配。", root=u)
        
        # 初始化以 u 为根的 M-交错树
        S = {u}
        T = set()
        parent = {u: None} # 用于回溯可扩路
        tree_edges = []
        
        # 使用 BFS 寻找最短的 M-可扩路；deque 能避免 list.pop(0) 的线性搬移成本。
        queue = deque([u])
        found_augmenting_path = False
        target_y = None
        
        record_step(f"初始化 M-交错树，根节点为【{u}】。当前 S = {S}, T = ∅。", root=u, S=S, T=T)
        
        while queue and not found_augmenting_path:
            x = queue.popleft()
            
            # 探索 x 的所有邻接顶点
            for y in adj[x]:
                if y in T:
                    continue
                
                # 记录交错树边 (x, y)
                tree_edges.append((x, y))
                parent[y] = x
                
                # 情况 A：y 未被匹配，找到了 M-可扩路
                if match_y[y] is None:
                    found_augmenting_path = True
                    target_y = y
                    
                    # 重构 M-可扩路
                    path = []
                    curr = y
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse() # 路径从 u 到 y
                    
                    record_step(
                        log_msg=f"找到从【{u}】到【{y}】的 M-可扩路: {' -> '.join(path)}。准备更新匹配关系。",
                        root=u, S=S, T=T, tree_edges=tree_edges, aug_path=path, found_path=True
                    )
                    break
                
                # 情况 B：y 已经被匹配，继续扩展交错树
                else:
                    z = match_y[y]
                    parent[z] = y
                    T.add(y)
                    S.add(z)
                    tree_edges.append((y, z))
                    queue.append(z)
                    
                    record_step(
                        log_msg=f"从【{x}】探索到任务【{y}】。该任务已分配给【{z}】。将任务【{y}】加入 T，学生【{z}】加入 S。交错树继续生长。",
                        root=u, S=S, T=T, tree_edges=tree_edges
                    )
            
        if found_augmenting_path:
            # 沿着可扩路交替更新匹配（异或运算 M = M Δ P）。
            # parent 在 X/Y 两侧交替记录父节点，因此可以从终点任务一路回溯到根学生。
            curr = target_y
            while curr is not None:
                p_x = parent[curr] # X 中的父节点
                next_y = parent[p_x] # X 的父节点对应的旧匹配 Y 节点（如果有）
                
                # 将 (p_x, curr) 加入匹配
                match_y[curr] = p_x
                match_x[p_x] = curr
                
                curr = next_y
                
            record_step(
                log_msg=f"成功为学生【{u}】分配任务【{target_y}】。当前匹配规模已增加。",
                root=u, S=S, T=T, tree_edges=tree_edges
            )
        else:
            record_step(
                log_msg=f"未能找到从【{u}】出发的 M-可扩路。学生【{u}】在此轮中暂时无法匹配。",
                root=u, S=S, T=T, tree_edges=tree_edges
            )
            
    final_matching = {k: v for k, v in match_y.items() if v is not None}
    return final_matching, steps


def km_optimal_matching_stepwise(left_nodes, right_nodes, capability_matrix):
    """
    带权偶图最大权完美匹配（KM 算法）- 步进式实现
    假设输入为两组节点和它们之间的能力/意愿矩阵。
    :param left_nodes: 左侧顶点 X (学生)
    :param right_nodes: 右侧顶点 Y (任务)
    :param capability_matrix: 字典 {student: {task: weight}}，代表意愿值或熟练度权值
    :return: (matching, total_weight, steps)
    """
    # 1. 补齐偶图，使之左右顶点数相等（若不相等则添加虚拟顶点，边权为0）。
    # KM 标准形式求完美匹配；虚拟点让“真实节点数不相等”的作业输入也能运行。
    orig_left_len = len(left_nodes)
    orig_right_len = len(right_nodes)
    n = max(orig_left_len, orig_right_len)
    
    # 复制顶点列表，并用虚拟节点补齐
    X = list(left_nodes)
    for i in range(n - orig_left_len):
        X.append(f"虚拟学生_{i+1}")
        
    Y = list(right_nodes)
    for j in range(n - orig_right_len):
        Y.append(f"虚拟任务_{j+1}")
        
    # 构建完整的权值矩阵 W[i][j]
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        u = X[i]
        for j in range(n):
            v = Y[j]
            # 如果是原图中的节点且有连边，则取权重；否则设为 0
            if i < orig_left_len and j < orig_right_len:
                W[i][j] = float(capability_matrix.get(u, {}).get(v, 0.0))
            else:
                W[i][j] = 0.0
                
    # 2. 初始化顶标 (Labels)
    # l_x[i] = max_j W[i][j], l_y[j] = 0
    l_x = [max(W[i]) for i in range(n)]
    l_y = [0.0] * n
    
    # 匹配数组：match_y[j] = i 表示 Y[j] 匹配给 X[i]
    match_y = [-1] * n
    match_x = [-1] * n
    
    steps = []
    
    def get_current_matching_dict():
        # 仅返回真实节点之间的匹配，排除虚拟节点
        m = {}
        for j in range(n):
            i = match_y[j]
            if i != -1 and i < orig_left_len and j < orig_right_len:
                m[Y[j]] = X[i]
        return m

    def record_step(log_msg, root_idx=None, S_indices=None, T_indices=None, slack=None, alpha=None):
        steps.append({
            'matching': get_current_matching_dict(),
            'l_x': {X[i]: l_x[i] for i in range(orig_left_len)},
            'l_y': {Y[j]: l_y[j] for j in range(orig_right_len)},
            'root': X[root_idx] if root_idx is not None and root_idx < orig_left_len else None,
            'S': [X[i] for i in S_indices if i < orig_left_len] if S_indices else [],
            'T': [Y[j] for j in T_indices if j < orig_right_len] if T_indices else [],
            'log': log_msg,
            'alpha': alpha,
            # 相等子图的边
            'equality_edges': [(X[i], Y[j]) for i in range(orig_left_len) for j in range(orig_right_len) if abs(l_x[i] + l_y[j] - W[i][j]) < 1e-9]
        })

    record_step("初始化KM算法顶标：左侧顶标设为对应最大边权，右侧顶标设为0。构建初始相等子图。")

    # 对 X 中的每个节点，寻找增广路
    for root in range(n):
        # 仅对真实节点（或为了完美匹配补齐的虚拟节点）寻找匹配
        # 初始化以 root 为根的交错树相关变量
        S = {root}
        T = set()
        
        # parent 用于重建可扩路
        parent = [-1] * n
        
        # slack 用于计算顶标调整值 alpha。
        # 它记录当前交错树 S 到每个右侧点还差多少才能成为相等子图边。
        # slack[j] = min_{i in S} {l_x[i] + l_y[j] - W[i][j]}
        slack = [float('inf')] * n
        slack_x = [-1] * n  # 记录 slack[j] 对应在 S 中的 x 索引
        
        def update_slack(i):
            for j in range(n):
                val = l_x[i] + l_y[j] - W[i][j]
                if val < slack[j]:
                    slack[j] = val
                    slack_x[j] = i
                    
        update_slack(root)
        
        if root < orig_left_len:
            record_step(f"【新阶段】开始为学生【{X[root]}】寻找最优匹配。初始化交错树。", root_idx=root, S_indices=S, T_indices=T)
            
        found_augmenting_path = False
        while not found_augmenting_path:
            # 1. 尝试在当前相等子图 $G_L$ 中寻找未访问的可用右侧节点 y
            # 也就是 slack[j] == 0 的节点
            target_j = -1
            for j in range(n):
                if j not in T and abs(slack[j]) < 1e-9:
                    target_j = j
                    break
                    
            # 2. 如果在相等子图里找到了可用边 S -> target_j
            if target_j != -1:
                # 记录 parent 关系
                parent[target_j] = slack_x[target_j]
                
                # 情况 A：target_j 未匹配，找到 M-可扩路
                if match_y[target_j] == -1:
                    found_augmenting_path = True
                    
                    # 沿着可扩路交替更新匹配关系
                    curr_j = target_j
                    while curr_j != -1:
                        prev_i = parent[curr_j]
                        next_j = match_x[prev_i]
                        
                        match_y[curr_j] = prev_i
                        match_x[prev_i] = curr_j
                        
                        curr_j = next_j
                        
                    if root < orig_left_len:
                        record_step(
                            log_msg=f"在相等子图中找到 M-可扩路。成功为学生【{X[root]}】匹配任务【{Y[target_j]}】。更新当前匹配。",
                            root_idx=root, S_indices=S, T_indices=T
                        )
                    break
                    
                # 情况 B：target_j 已经匹配给 match_y[target_j]，扩展交错树
                else:
                    matched_i = match_y[target_j]
                    T.add(target_j)
                    S.add(matched_i)
                    update_slack(matched_i)
                    
                    if root < orig_left_len:
                        record_step(
                            log_msg=f"探索相等子图边【{X[slack_x[target_j]]}】->【{Y[target_j]}】。该任务已被【{X[matched_i]}】分配。扩展交错树并更新松弛值 slack。",
                            root_idx=root, S_indices=S, T_indices=T
                        )
            
            # 3. 如果在相等子图里没有可用边了（即所有 j 不在 T 中的 slack[j] 都大于 0）
            # 我们必须调整顶标以扩充相等子图
            else:
                # 计算 alpha = min_{j not in T} slack[j]
                alpha = float('inf')
                for j in range(n):
                    if j not in T:
                        alpha = min(alpha, slack[j])
                        
                if alpha == float('inf') or alpha < 1e-9:
                    # 避免死循环
                    break
                    
                # 更新顶标
                for i in range(n):
                    if i in S:
                        l_x[i] -= alpha
                for j in range(n):
                    if j in T:
                        l_y[j] += alpha
                    else:
                        # 调整其余节点的 slack 值
                        slack[j] -= alpha
                        
                if root < orig_left_len:
                    record_step(
                        log_msg=f"在相等子图中没有可用的增广方向。计算顶标调整值 alpha = {alpha:.1f}。左侧 S 中顶标减去 alpha，右侧 T 中顶标加上 alpha。相等子图扩充，引入新边。",
                        root_idx=root, S_indices=S, T_indices=T, alpha=alpha
                    )

    # 3. 计算最终的总权值
    total_weight = 0.0
    final_matching = get_current_matching_dict()
    for v, u in final_matching.items():
        total_weight += capability_matrix.get(u, {}).get(v, 0.0)
        
    return final_matching, total_weight, steps
