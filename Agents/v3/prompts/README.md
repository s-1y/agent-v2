# V3 Prompts

阶段文档在 `phases/<角色>/`。`context.py` 按 `node.phase()` 读进回合消息，位置、项目根目录和上级交代由它直接拼好，阶段文件里不再写占位符。

工具说明在 `tools/`。观察阶段把话写完就结束，没有单独的结束工具。

## 角色

- `phases/manager/`：根。看要求，派设计，派开发，用法没说清就让同一个人补，说清了派验收，再把筛选后的建议交回设计。
- `phases/designer/`：第一轮是补全要求、写第一版手册。之后每一轮是对接验收、只在新版本文件夹写改进手册。
- `phases/worker/`：开发。认任务之后自己做，或分工、建组、下发、收回。做完后系统追问怎样使用或怎样查看结果。
- `phases/reviewer/leader/`：验收负责人。看交付，想划分，`send_review_plan` 后系统按编号同时启动队员。收齐后筛选。
- `phases/reviewer/member/`：验收队员。先检查分到的部分，再按三类归类交回。不再往下派。

## 工具

`choose_work_mode`、`choose_org_mode`、`write_plan`、`create_parallel_group`、`create_sequence_group`、`dispatch`、`send_review_plan`、`send_next`、`finish_design`、`continue_child`、`ask_parent`、`answer_child_question`、`change_plan`、`return_to_parent`。
