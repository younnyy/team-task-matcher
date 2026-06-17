from algorithms import hungarian_maximum_matching_stepwise, km_optimal_matching_stepwise
from reporting import generate_assignment_export, generate_assignment_report
from sample_data import DEFAULT_STUDENTS, DEFAULT_TASKS, DEFAULT_UNWEIGHTED_EDGES, DEFAULT_WEIGHTS


def test_default_unweighted_report_lists_each_member_assignment():
    matching, steps = hungarian_maximum_matching_stepwise(
        DEFAULT_STUDENTS,
        DEFAULT_TASKS,
        DEFAULT_UNWEIGHTED_EDGES,
    )

    report = generate_assignment_report(
        students=DEFAULT_STUDENTS,
        tasks=DEFAULT_TASKS,
        algorithm_mode="无权最大匹配 (匈牙利算法)",
        final_matching=matching,
        steps=steps,
        edges=DEFAULT_UNWEIGHTED_EDGES,
    )

    assert "共成功分配了 **5** 项任务" in report
    assert "| 张三 | 前端开发 |" in report
    assert "| 李四 | 后端接口 |" in report
    assert "| 王五 | 算法模块 |" in report
    assert "| 赵六 | 文档测试 |" in report
    assert "| 钱七 | 数据库设计 |" in report
    assert "小组成员分工说明" in report
    assert "任务分配 0 项" not in report


def test_default_weighted_report_includes_total_score_and_member_work():
    matching, total_weight, steps = km_optimal_matching_stepwise(
        DEFAULT_STUDENTS,
        DEFAULT_TASKS,
        DEFAULT_WEIGHTS,
    )

    report = generate_assignment_report(
        students=DEFAULT_STUDENTS,
        tasks=DEFAULT_TASKS,
        algorithm_mode="最大权最优匹配 (KM 算法)",
        final_matching=matching,
        steps=steps,
        capability_matrix=DEFAULT_WEIGHTS,
        total_weight=total_weight,
    )

    assert "最大权分配总分为 **25.0**" in report
    assert "| 张三 | 前端开发 | 5.0 |" in report
    assert "| 李四 | 后端接口 | 5.0 |" in report
    assert "| 王五 | 算法模块 | 5.0 |" in report
    assert "| 赵六 | 文档测试 | 5.0 |" in report
    assert "| 钱七 | 数据库设计 | 5.0 |" in report
    assert "小组成员分工说明" in report


def test_assignment_export_is_group_work_output_not_project_report():
    matching, steps = hungarian_maximum_matching_stepwise(
        DEFAULT_STUDENTS,
        DEFAULT_TASKS,
        DEFAULT_UNWEIGHTED_EDGES,
    )

    export = generate_assignment_export(
        students=DEFAULT_STUDENTS,
        tasks=DEFAULT_TASKS,
        algorithm_mode="无权最大匹配 (匈牙利算法)",
        final_matching=matching,
        steps=steps,
        edges=DEFAULT_UNWEIGHTED_EDGES,
    )

    assert export.startswith("# 小组分工导出")
    assert "| 张三 | 前端开发 |" in export
    assert "成功分配任务数：5" in export
    assert "基于偶图匹配的课程项目任务分配系统分析与求解报告" not in export
