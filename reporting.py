# -*- coding: utf-8 -*-
"""生成系统内小组分工导出文件的纯函数。

Streamlit 页面只负责展示和触发保存；导出正文在这里集中生成，方便测试，
也能避免页面刷新时无意覆盖提交用的正式作业报告。
"""

from pathlib import Path


def _is_weighted_mode(algorithm_mode):
    return "最大权" in algorithm_mode or "KM" in algorithm_mode


def _task_for_student(final_matching, student):
    for task, matched_student in final_matching.items():
        if matched_student == student:
            return task
    return None


def _ordered_unassigned_students(students, final_matching):
    matched_students = set(final_matching.values())
    return [student for student in students if student not in matched_students]


def _ordered_unassigned_tasks(tasks, final_matching):
    return [task for task in tasks if task not in final_matching]


def build_match_table(students, final_matching, capability_matrix=None):
    """按学生原始顺序生成最终匹配表，避免 set 导致报告顺序随机变化。"""
    weighted = capability_matrix is not None
    header = "| 学生成员 (X) | 分配任务 (Y) |"
    divider = "| :--- | :--- |"
    if weighted:
        header += " 熟练度/意愿得分 |"
        divider += " :--- |"

    lines = [header, divider]
    for student in students:
        task = _task_for_student(final_matching, student)
        if task is None:
            continue
        if weighted:
            weight = capability_matrix.get(student, {}).get(task, 0.0)
            lines.append(f"| {student} | {task} | {weight:.1f} |")
        else:
            lines.append(f"| {student} | {task} |")

    return "\n".join(lines)


def build_member_work_notes(students, final_matching, capability_matrix=None):
    """生成每个小组成员的分工说明，满足作业中“结果分析”部分的可读性要求。"""
    lines = ["### 2. 小组成员分工说明", ""]
    for student in students:
        task = _task_for_student(final_matching, student)
        if task is None:
            lines.append(f"* **{student}**：未分配到任务。该成员位于未饱和顶点集合中，需要增加可承担任务边。")
            continue

        if capability_matrix is None:
            lines.append(
                f"* **{student}**：负责 **{task}**。该边存在于能力关系图中，"
                "并在匈牙利算法的增广过程中进入最终匹配。"
            )
        else:
            weight = capability_matrix.get(student, {}).get(task, 0.0)
            lines.append(
                f"* **{student}**：负责 **{task}**，熟练度/意愿得分为 **{weight:.1f}**。"
                "KM 算法在相等子图中选择该匹配，使团队总效能最大化。"
            )
    return "\n".join(lines)


def build_input_summary(students, tasks, edges=None, capability_matrix=None):
    lines = ["### 1. 输入数据说明", ""]
    lines.append(f"* 学生成员集合 $X$：`{', '.join(students)}`。")
    lines.append(f"* 待分配任务集合 $Y$：`{', '.join(tasks)}`。")

    if capability_matrix is None:
        edge_text = "、".join(f"{student}-{task}" for student, task in (edges or []))
        lines.append(f"* 能力边集合 $E$：{edge_text if edge_text else '空集'}。")
    else:
        lines.append("* 权值矩阵含义：每个数值表示成员对任务的熟练度或主观意愿得分。")
        for student in students:
            row = capability_matrix.get(student, {})
            row_text = "，".join(f"{task}:{row.get(task, 0.0):.1f}" for task in tasks)
            lines.append(f"  * {student}：{row_text}")
    return "\n".join(lines)


def build_bottleneck_analysis(students, tasks, final_matching, steps, algorithm_mode):
    unassigned_students = _ordered_unassigned_students(students, final_matching)
    unassigned_tasks = _ordered_unassigned_tasks(tasks, final_matching)
    lines = ["### 3. 团队任务分配瓶颈与结构分析", ""]

    if len(final_matching) == min(len(students), len(tasks)) and not unassigned_students and not unassigned_tasks:
        lines.append("本次匹配成功实现完美匹配：每名成员都被分配到一项任务，每项任务也都恰好由一名成员负责。")
        lines.append("这说明当前能力边分布满足 Hall 定理要求，团队技能覆盖较均衡。")
        return "\n".join(lines)

    lines.append(
        f"本次分配没有达到完全匹配。未匹配学生：`{', '.join(unassigned_students) or '无'}`；"
        f"未匹配任务：`{', '.join(unassigned_tasks) or '无'}`。"
    )

    if not _is_weighted_mode(algorithm_mode):
        hall_s = []
        hall_t = []
        for step in reversed(steps or []):
            if not step.get("found_path") and step.get("S"):
                hall_s = step.get("S", [])
                hall_t = step.get("T", [])
                break
        if hall_s:
            lines.append("")
            lines.append("根据 Hall 定理，偶图存在饱和 $X$ 的匹配，当且仅当任意 $S \\subseteq X$ 都满足 $|N(S)| \\ge |S|$。")
            lines.append(f"系统在交错树搜索中识别到瓶颈子集 $S={hall_s}$，对应邻集 $N(S)={hall_t}$。")
            lines.append(f"由于 $|N(S)|={len(hall_t)} < |S|={len(hall_s)}$，因此当前能力图不存在完美匹配。")
    else:
        lines.append("在 KM 模式下，未匹配通常意味着真实节点数不相等，或部分任务只能通过虚拟节点/零权边完成。")

    return "\n".join(lines)


def generate_assignment_export(
    students,
    tasks,
    algorithm_mode,
    final_matching,
    steps,
    edges=None,
    capability_matrix=None,
    total_weight=None,
):
    """生成系统内导出的“小组分工表”，区别于提交用的完整作业报告。"""
    weighted = _is_weighted_mode(algorithm_mode)
    matrix_for_report = capability_matrix if weighted else None
    match_count = len(final_matching)
    assignment_rate = match_count / len(tasks) * 100 if tasks else 0.0
    score_line = ""
    if weighted:
        score_line = f"\n* 最大权分配总分：{total_weight or 0.0:.1f}"

    return f"""# 小组分工导出

## 一、导出摘要

* 算法模式：{algorithm_mode}
* 小组成员数：{len(students)}
* 项目任务数：{len(tasks)}
* 成功分配任务数：{match_count}
* 任务分配率：{assignment_rate:.1f}%{score_line}
* 算法推演步骤数：{len(steps or [])}

## 二、输入数据

{build_input_summary(students, tasks, edges=edges, capability_matrix=matrix_for_report)}

## 三、最终分工

{build_match_table(students, final_matching, capability_matrix=matrix_for_report)}

{build_member_work_notes(students, final_matching, capability_matrix=matrix_for_report)}

{build_bottleneck_analysis(students, tasks, final_matching, steps, algorithm_mode)}
""".strip() + "\n"


def generate_assignment_report(
    students,
    tasks,
    algorithm_mode,
    final_matching,
    steps,
    edges=None,
    capability_matrix=None,
    total_weight=None,
):
    """返回完整 Markdown 报告文本，不产生文件写入副作用。"""
    weighted = _is_weighted_mode(algorithm_mode)
    matrix_for_report = capability_matrix if weighted else None
    match_count = len(final_matching)
    assignment_rate = match_count / len(tasks) * 100 if tasks else 0.0

    if weighted:
        objective = "寻找一种一人一任务的分配方案，使团队整体熟练度/意愿得分之和达到最大。"
        score_sentence = f"最大权分配总分为 **{total_weight or 0.0:.1f}**，平均得分为 **{(total_weight or 0.0) / match_count:.2f}**。" if match_count else "最大权分配总分为 **0.0**。"
    else:
        objective = "在成员能力资质符合的前提下，使尽可能多的任务被成功分配，实现最大基数任务覆盖。"
        score_sentence = f"任务分配率为 **{assignment_rate:.1f}%**。"

    input_summary = build_input_summary(students, tasks, edges=edges, capability_matrix=matrix_for_report)
    match_table = build_match_table(students, final_matching, capability_matrix=matrix_for_report)
    member_notes = build_member_work_notes(students, final_matching, capability_matrix=matrix_for_report)
    bottleneck = build_bottleneck_analysis(students, tasks, final_matching, steps, algorithm_mode)

    if weighted:
        algo_part = r"""
### 算法名称：KM 算法 (Kuhn-Munkres Algorithm)

KM 算法用于求解带权偶图的最大权完美匹配。它为左右两侧顶点设置可行顶标 $l(x)$ 与 $l(y)$，并始终保持 $l(x)+l(y)\ge w(x,y)$。算法先在满足 $l(x)+l(y)=w(x,y)$ 的相等子图中寻找增广路；若无法增广，则通过最小松弛量 $\alpha$ 调整顶标，使相等子图逐步扩大，直到得到最大权匹配。
"""
    else:
        algo_part = r"""
### 算法名称：匈牙利方法 (Hungarian Method)

匈牙利算法用于求解无权偶图的最大基数匹配。算法从空匹配开始，反复从未匹配的左侧顶点出发建立 $M$-交错树。如果找到一条连接到未匹配右侧顶点的 $M$-可扩路，就沿该路径执行异或更新 $M \leftarrow M \Delta E(P)$，使匹配数增加 1；如果不存在可扩路，则说明当前顶点无法在现有能力边下继续增广。
"""

    report = f"""# 基于偶图匹配的课程项目任务分配系统分析与求解报告

## 一、问题描述与背景叙述

在软件工程、团队课程设计等协同开发场景中，团队常常面临“任务多、人手杂、分工难”的问题。如何科学、合理地把开发任务分配给团队成员，是保障项目按时保质交付的关键。

本团队共包含 {len(students)} 名成员（{', '.join(students)}），共有 {len(tasks)} 项待开发任务（{', '.join(tasks)}）。本次建模目标是：{objective}

## 二、偶图匹配图论模型构建

将该任务分配问题建模为偶图 $G=(X,Y,E)$：

1. 左侧顶点集 $X$ 表示小组成员。
2. 右侧顶点集 $Y$ 表示待完成任务。
3. 边 $(x,y)$ 表示成员 $x$ 能够承担任务 $y$；在带权模式下，边权 $w(x,y)$ 表示熟练度或意愿得分。
4. 匹配 $M\\subseteq E$ 表示最终分工方案，要求任意两条匹配边不共享端点，因此满足“一个成员最多负责一个任务、一个任务最多由一名成员负责”的实际约束。

{input_summary}

## 三、图论算法原理分析

{algo_part}

定理支撑：

* Hall 定理：偶图存在饱和 $X$ 的匹配，当且仅当对任意 $S\\subseteq X$，都有 $|N(S)|\\ge |S|$。
* Konig 定理：在偶图中，最大匹配的边数等于最小顶点覆盖的顶点数，为匹配问题提供了重要的对偶理论基础。

## 四、求解过程、结果与分析

系统共记录了 **{len(steps or [])}** 个算法推演步骤。通过逐步观察交错树扩展、增广路发现与匹配更新，可以清楚看到最终分工方案如何从空匹配逐步形成。

### 1. 最终分配方案结果

{match_table}

在本次分配中，共成功分配了 **{match_count}** 项任务。{score_sentence}

{member_notes}

{bottleneck}

## 五、核心程序实现说明

本系统使用 Python 编写，核心代码由两部分组成：

* `algorithms.py`：实现匈牙利算法和 KM 算法，并记录可视化所需的每一步状态。
* `app.py`：基于 Streamlit 构建交互界面，提供数据录入、图形展示、步骤推演、结果分析与报告导出。

核心求解流程如下：

```python
for u in left_nodes:
    if u 已经匹配:
        continue
    从 u 出发建立 M-交错树
    如果找到 M-可扩路:
        沿可扩路执行匹配翻转
```

## 六、报告总结

通过本作业的建模与计算，团队任务分配问题被转化为图论中的偶图匹配问题。算法不仅能给出当前条件下的最优分工方案，还能解释为什么该方案满足“一人一任务”的约束，并在匹配不完整时指出造成瓶颈的成员或任务集合。这说明图论方法不仅适用于抽象数学问题，也可以直接服务于课程项目管理、人员排班和资源分配等现实场景。
"""
    return report.strip() + "\n"


def save_report(report_text, output_path):
    path = Path(output_path)
    path.write_text(report_text, encoding="utf-8")
    return path
