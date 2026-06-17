# -*- coding: utf-8 -*-
"""Streamlit 交互界面：偶图匹配课程项目任务分配系统。"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from algorithms import hungarian_maximum_matching_stepwise, km_optimal_matching_stepwise
from reporting import generate_assignment_export, save_report
from sample_data import DEFAULT_STUDENTS, DEFAULT_TASKS, DEFAULT_UNWEIGHTED_EDGES, DEFAULT_WEIGHTS


PROJECT_DIR = Path(__file__).resolve().parent
EXPORT_PATH = PROJECT_DIR / "小组分工导出.md"


st.set_page_config(
    page_title="偶图匹配任务分配系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Matplotlib 默认字体不一定支持中文，显式配置能避免图中节点乱码。
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang HK", "Heiti TC", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


st.markdown(
    """
    <style>
    .block-container { padding-top: 3.3rem; padding-bottom: 2.5rem; }
    .main-title { font-size: 2.05rem; font-weight: 760; line-height: 1.18; margin-bottom: .3rem; }
    .subtitle { color: #4b5563; font-size: 1rem; margin-bottom: 1.1rem; }
    .section-note { color: #6b7280; font-size: .9rem; margin-top: -.25rem; }
    .status-pill {
        display: inline-flex; align-items: center; gap: .35rem;
        padding: .28rem .55rem; border: 1px solid #d1d5db; border-radius: 999px;
        color: #374151; background: #ffffff; font-size: .84rem; margin-right: .35rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb; border-radius: 8px; padding: .72rem .84rem; background: #ffffff;
    }
    div[data-testid="stMetric"] label { color: #6b7280; }
    .stTabs [data-baseweb="tab-list"] { gap: .3rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding-left: 1rem; padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_names(raw_text):
    """把逗号分隔的输入转成去重后的节点列表，保留用户输入顺序。"""
    names = []
    for item in raw_text.replace("，", ",").split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    return names


def build_default_edge_matrix(students, tasks, edges):
    matrix = pd.DataFrame(False, index=students, columns=tasks)
    for student, task in edges:
        if student in matrix.index and task in matrix.columns:
            matrix.loc[student, task] = True
    return matrix


def build_weight_matrix(students, tasks, source_weights=None):
    matrix = pd.DataFrame(1.0, index=students, columns=tasks)
    for i, student in enumerate(students):
        for j, task in enumerate(tasks):
            if source_weights and student in source_weights and task in source_weights[student]:
                matrix.loc[student, task] = float(source_weights[student][task])
            elif i == j:
                matrix.loc[student, task] = 5.0
    return matrix


def dataframe_to_edges(matrix):
    edges = []
    for student in matrix.index:
        for task in matrix.columns:
            if bool(matrix.loc[student, task]):
                edges.append((student, task))
    return edges


def dataframe_to_weights(matrix):
    weights = {}
    for student in matrix.index:
        weights[student] = {}
        for task in matrix.columns:
            weights[student][task] = float(matrix.loc[student, task])
    return weights


def task_for_student(final_matching, student):
    for task, matched_student in final_matching.items():
        if matched_student == student:
            return task
    return None


def draw_bipartite_graph(
    left_nodes,
    right_nodes,
    edges,
    matching,
    root=None,
    S=None,
    T=None,
    tree_edges=None,
    aug_path=None,
    labels_x=None,
    labels_y=None,
    weights=None,
    equality_edges=None,
):
    """绘制左右对齐的偶图，并高亮当前算法步骤中的关键状态。"""
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_xlim(-0.62, 1.62)

    n_left = len(left_nodes)
    n_right = len(right_nodes)
    max_nodes = max(n_left, n_right, 1)
    spacing_left = (max_nodes - 1) / (n_left - 1) if n_left > 1 else 0
    spacing_right = (max_nodes - 1) / (n_right - 1) if n_right > 1 else 0

    coords = {}
    for i, node in enumerate(left_nodes):
        coords[node] = (0.0, max_nodes - i * spacing_left - 1 if n_left > 1 else (max_nodes - 1) / 2)
    for j, node in enumerate(right_nodes):
        coords[node] = (1.0, max_nodes - j * spacing_right - 1 if n_right > 1 else (max_nodes - 1) / 2)

    set_S = set(S or [])
    set_T = set(T or [])
    tree_edge_set = {frozenset(edge) for edge in (tree_edges or [])}
    equality_edge_set = {frozenset(edge) for edge in (equality_edges or [])}

    if weights:
        edges_to_draw = [
            (student, task)
            for student in left_nodes
            for task in right_nodes
            if weights.get(student, {}).get(task, 0.0) > 0
        ]
    else:
        edges_to_draw = edges

    aug_path_edges = set()
    if aug_path and len(aug_path) > 1:
        for idx in range(len(aug_path) - 1):
            aug_path_edges.add(frozenset((aug_path[idx], aug_path[idx + 1])))

    for student, task in edges_to_draw:
        key = frozenset((student, task))
        is_matching = matching.get(task) == student
        if key in aug_path_edges:
            color, width, style, alpha = "#d97706", 4.0, "solid", 1.0
        elif is_matching:
            color, width, style, alpha = "#dc2626", 3.0, "solid", 0.92
        elif key in tree_edge_set:
            color, width, style, alpha = "#2563eb", 2.1, "dashed", 0.86
        elif key in equality_edge_set:
            color, width, style, alpha = "#059669", 1.7, "solid", 0.68
        else:
            color, width, style, alpha = "#cbd5e1", 1.0, "dotted", 0.42

        ax.plot(
            [coords[student][0], coords[task][0]],
            [coords[student][1], coords[task][1]],
            color=color,
            linewidth=width,
            linestyle=style,
            alpha=alpha,
            zorder=1,
        )

        if weights and weights.get(student, {}).get(task, 0.0) > 0:
            mid_x = (coords[student][0] + coords[task][0]) / 2
            mid_y = (coords[student][1] + coords[task][1]) / 2
            ax.text(
                mid_x,
                mid_y + 0.055,
                f"{weights[student][task]:.1f}",
                color="#334155",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.4),
            )

    node_list = left_nodes + right_nodes
    node_colors = []
    for node in node_list:
        if node == root:
            node_colors.append("#ef4444")
        elif node in set_S:
            node_colors.append("#93c5fd")
        elif node in set_T:
            node_colors.append("#86efac")
        else:
            node_colors.append("#f8fafc")

    nx.draw_networkx_nodes(
        nx.Graph(),
        coords,
        nodelist=node_list,
        node_color=node_colors,
        node_size=760,
        edgecolors="#334155",
        linewidths=1.3,
        ax=ax,
    )

    for node in left_nodes:
        label = node
        if labels_x and node in labels_x:
            label += f"\n[l(x)={labels_x[node]:.1f}]"
        ax.text(coords[node][0] - 0.065, coords[node][1], label, fontsize=10, fontweight="bold", ha="right", va="center")

    for node in right_nodes:
        label = node
        if labels_y and node in labels_y:
            label += f"\n[l(y)={labels_y[node]:.1f}]"
        ax.text(coords[node][0] + 0.065, coords[node][1], label, fontsize=10, fontweight="bold", ha="left", va="center")

    ax.text(0.0, max_nodes - 0.25, "小组成员 X", ha="center", va="bottom", color="#475569", fontsize=11)
    ax.text(1.0, max_nodes - 0.25, "项目任务 Y", ha="center", va="bottom", color="#475569", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


def render_metric_strip(students, tasks, final_matching, algorithm_mode, total_weight):
    match_count = len(final_matching)
    assignment_rate = match_count / len(tasks) * 100 if tasks else 0.0
    cols = st.columns(4)
    cols[0].metric("小组成员", f"{len(students)} 人")
    cols[1].metric("待分配任务", f"{len(tasks)} 项")
    cols[2].metric("成功匹配", f"{match_count} 项", f"{assignment_rate:.1f}%")
    if "最大权" in algorithm_mode:
        cols[3].metric("团队总得分", f"{total_weight:.1f}")
    else:
        cols[3].metric("算法步骤", f"{st.session_state.get('step_count', 0)} 步")


def render_result_table(students, final_matching, capability_matrix=None):
    rows = []
    for student in students:
        task = task_for_student(final_matching, student)
        if capability_matrix is not None and task is not None:
            rows.append({"成员": student, "最终分工": task, "得分": f"{capability_matrix[student][task]:.1f}"})
        else:
            rows.append({"成员": student, "最终分工": task or "未分配"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


st.markdown('<div class="main-title">基于偶图匹配的课程项目任务分配系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">用匈牙利算法求最大任务覆盖，用 KM 算法求最大权分工，并导出小组成员任务分配结果。</div>',
    unsafe_allow_html=True,
)

st.sidebar.header("控制台")
algorithm_mode = st.sidebar.radio(
    "匹配模式",
    ["无权最大匹配 (匈牙利算法)", "最大权最优匹配 (KM 算法)"],
    help="无权模式追求匹配数量最多；带权模式追求总得分最高。",
)
data_mode = st.sidebar.radio("数据来源", ["使用内置默认案例", "自定义录入数据"])
show_raw_report = st.sidebar.checkbox("显示分工导出预览", value=True)

is_weighted = "最大权" in algorithm_mode
students = list(DEFAULT_STUDENTS)
tasks = list(DEFAULT_TASKS)
edges = []
capability_matrix = {}

st.markdown(
    f"""
    <span class="status-pill">模式：{algorithm_mode}</span>
    <span class="status-pill">数据：{data_mode}</span>
    <span class="status-pill">导出文件：{EXPORT_PATH.name}</span>
    """,
    unsafe_allow_html=True,
)

tab_data, tab_steps, tab_report = st.tabs(["数据建模", "算法推演", "分工导出"])

with tab_data:
    left_col, right_col = st.columns([0.92, 1.08])
    with left_col:
        st.subheader("问题规模")
        if data_mode == "自定义录入数据":
            students_raw = st.text_area("小组成员 X（逗号分隔）", "张三, 李四, 王五, 赵六, 钱七", height=86)
            tasks_raw = st.text_area("项目任务 Y（逗号分隔）", "前端开发, 后端接口, 数据库设计, 算法模块, 文档测试", height=86)
            students = parse_names(students_raw)
            tasks = parse_names(tasks_raw)
        else:
            st.dataframe(pd.DataFrame({"小组成员 X": students}), width="stretch", hide_index=True)
            st.dataframe(pd.DataFrame({"项目任务 Y": tasks}), width="stretch", hide_index=True)

        if not students or not tasks:
            st.error("请至少输入 1 名成员和 1 项任务。")

    with right_col:
        st.subheader("能力关系")
        if is_weighted:
            if data_mode == "使用内置默认案例":
                weight_df = build_weight_matrix(students, tasks, DEFAULT_WEIGHTS)
                st.dataframe(weight_df, width="stretch")
            else:
                weight_df = build_weight_matrix(students, tasks)
                weight_df = st.data_editor(
                    weight_df,
                    width="stretch",
                    key=f"weights_{len(students)}_{len(tasks)}",
                    num_rows="fixed",
                )
            capability_matrix = dataframe_to_weights(weight_df)
            st.caption("分值越高表示越适合承担该任务，KM 算法会最大化总得分。")
        else:
            if data_mode == "使用内置默认案例":
                edge_df = build_default_edge_matrix(students, tasks, DEFAULT_UNWEIGHTED_EDGES)
                st.dataframe(edge_df, width="stretch")
            else:
                edge_df = build_default_edge_matrix(students, tasks, [])
                edge_df = st.data_editor(
                    edge_df,
                    width="stretch",
                    key=f"edges_{len(students)}_{len(tasks)}",
                    num_rows="fixed",
                )
            edges = dataframe_to_edges(edge_df)
            st.caption("勾选表示该成员具备承担该任务的能力。")
            if not edges:
                st.warning("当前没有任何能力边，算法只能得到空匹配。")

if not students or not tasks:
    st.stop()

if is_weighted:
    final_matching, total_weight, steps = km_optimal_matching_stepwise(students, tasks, capability_matrix)
else:
    final_matching, steps = hungarian_maximum_matching_stepwise(students, tasks, edges)
    total_weight = 0.0

st.session_state["step_count"] = len(steps)
export_text = generate_assignment_export(
    students=students,
    tasks=tasks,
    algorithm_mode=algorithm_mode,
    final_matching=final_matching,
    steps=steps,
    edges=edges,
    capability_matrix=capability_matrix if is_weighted else None,
    total_weight=total_weight if is_weighted else None,
)

with tab_steps:
    st.subheader("算法推演")
    render_metric_strip(students, tasks, final_matching, algorithm_mode, total_weight)
    st.markdown('<div class="section-note">拖动步骤条可以查看交错树扩展、增广路发现、匹配边更新和 KM 顶标变化。</div>', unsafe_allow_html=True)

    if steps:
        step_idx = st.slider("推演步骤", 0, len(steps) - 1, len(steps) - 1)
        curr_step = steps[step_idx]
        graph_col, log_col = st.columns([1.55, 0.9])

        with graph_col:
            fig = draw_bipartite_graph(
                left_nodes=students,
                right_nodes=tasks,
                edges=edges,
                matching=curr_step["matching"],
                root=curr_step.get("root"),
                S=curr_step.get("S"),
                T=curr_step.get("T"),
                tree_edges=curr_step.get("tree_edges"),
                aug_path=curr_step.get("augmenting_path"),
                labels_x=curr_step.get("l_x"),
                labels_y=curr_step.get("l_y"),
                weights=capability_matrix if is_weighted else None,
                equality_edges=curr_step.get("equality_edges"),
            )
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with log_col:
            st.markdown(f"#### 第 {step_idx + 1} / {len(steps)} 步")
            st.info(curr_step["log"])
            st.markdown("**图例**")
            st.markdown(
                """
                - 红色节点：当前交错树根
                - 蓝色节点：已访问成员集合 S
                - 绿色节点：已访问任务集合 T
                - 红色实线：当前匹配边
                - 蓝色虚线：交错树边
                - 橙色粗线：本步发现的可扩路
                """
            )
            if is_weighted:
                st.markdown("- 绿色细线：KM 相等子图边")
    else:
        st.warning("当前输入无法生成算法步骤。")

with tab_report:
    result_col, report_col = st.columns([0.88, 1.12])
    with result_col:
        st.subheader("最终分工")
        render_result_table(students, final_matching, capability_matrix if is_weighted else None)
        if is_weighted:
            avg_score = total_weight / len(final_matching) if final_matching else 0.0
            st.metric("最大权分配总分", f"{total_weight:.1f}", f"平均 {avg_score:.2f}")
        else:
            st.metric("成功分配任务数", f"{len(final_matching)} 项", f"{len(final_matching) / len(tasks) * 100:.1f}%")

        if len(final_matching) < min(len(students), len(tasks)):
            st.warning("当前不是完美匹配，报告会自动加入瓶颈分析。")
        else:
            st.success("当前得到完美匹配，所有成员与任务均完成一对一分工。")

    with report_col:
        st.subheader("小组分工导出")
        if st.button("保存分工到工作区", type="primary", width="stretch"):
            saved_path = save_report(export_text, EXPORT_PATH)
            st.success(f"已写入：{saved_path}")

        st.download_button(
            "下载 Markdown 分工表",
            data=export_text,
            file_name="小组分工导出.md",
            mime="text/markdown",
            width="stretch",
        )

        if show_raw_report:
            st.text_area("分工导出预览", export_text, height=420)
        else:
            st.markdown("分工导出预览已在侧边栏关闭，可直接保存或下载。")
