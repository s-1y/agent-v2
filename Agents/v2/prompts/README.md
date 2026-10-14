# V2 Prompts

当前 V2 使用 workspace dispatch 模型：节点先经过更细的阶段自动机，再写不可变 plan、创建节点组、最后 dispatch。progress 已取消，状态由 workspace / group / questions / child_reports 渲染。

`position` 是 harness 从 workspace plan 派生出来的短路标，只在新成员首次开工时作为上下文出现；agent 不负责维护它。

## 主模板

| 文件 | 用途 |
| --- | --- |
| `phases/*.md` | 每个 phase 的详细阶段提示 |

## 工具文档

新工具包括：`finish_orient`、`choose_work_mode`、`choose_org_mode`、`write_plan`、`create_parallel_group`、`create_sequence_group`、`dispatch`、`continue_child`、`ask_parent`、`answer_child_question`、`change_plan`、`return_to_parent`。

## 阶段提示

任务节点每次开工时，`context.py` 会根据 `node.phase()` 自动加载。新编号阶段包括：

- `phases/0_system.md`
- `phases/1_orient.md`
- `phases/2_choose_work_mode.md`
- `phases/3A_self_work.md`
- `phases/3A2_summarize.md`（写代码停工后只用文字概括，这段话交给上级）
- `phases/3B_choose_org_mode.md`
- `phases/4A1_design_sequence.md`（只用文字回答设计问题，不调用决策工具）
- `phases/4A2_write_sequence_plan.md`（根据上一步回答调用 `write_plan`）
- `phases/4B1_design_parallel.md`
- `phases/4B2_write_parallel_plan.md`

设计步结束时，节点不交决定即被外层视为答完，推进到对应的写 plan 步；两步共用同一个会话，所以设计回答就在上下文里。
- `phases/5_create_group.md`
- `phases/6_dispatch.md`
- `phases/7_answer_question.md`
- `phases/8_review.md`

这些内容会由 `context.py` 自动贴进任务节点回合消息。工具文档只负责简短解释工具参数和回执；阶段文档负责详细说明当前阶段应该怎么想。

## 可用变量

阶段提示可以使用：

- `{{position}}`
- `{{project_root}}`
- `{{from_parent}}`

回合消息不再使用单独模板文件，由 `context.py` 拼接状态信封并插入当前 phase 文档。
