"""决定工具。调用即把一个决定交给外层 scheduler 并暂停本回合。

V2 workspace 设计里，工具只表达当前阶段的一个动作；时机检查和状态流转
统一放在 scheduler.apply_decision 里。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from Agents.engine import (
    Action,
    Observation,
    Tool,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from Agents.v2.tool_prompts import load_tool_doc

_DECISIONS: dict[str, "Decision"] = {}

_FINISH_ORIENT = load_tool_doc("finish_orient")
_CHOOSE_WORK_MODE = load_tool_doc("choose_work_mode")
_CHOOSE_ORG_MODE = load_tool_doc("choose_org_mode")
_WRITE_PLAN = load_tool_doc("write_plan")
_CREATE_PARALLEL = load_tool_doc("create_parallel_group")
_CREATE_SEQUENCE = load_tool_doc("create_sequence_group")
_DISPATCH = load_tool_doc("dispatch")
_CONTINUE_CHILD = load_tool_doc("continue_child")
_ASK_PARENT = load_tool_doc("ask_parent")
_ANSWER_CHILD = load_tool_doc("answer_child_question")
_CHANGE_PLAN = load_tool_doc("change_plan")
_RETURN = load_tool_doc("return_to_parent")


class Decision(BaseModel):
    kind: Literal[
        "finish_orient",
        "choose_work_mode",
        "choose_org_mode",
        "write_plan",
        "create_parallel_group",
        "create_sequence_group",
        "dispatch",
        "continue_child",
        "ask_parent",
        "answer_child_question",
        "change_plan",
        "return_to_parent",
    ]
    payload: dict[str, Any]


def _stash(conversation, decision: Decision) -> None:
    key = str(getattr(conversation, "id", "default"))
    _DECISIONS[key] = decision
    if conversation is not None:
        conversation.pause()


def take_decision(conversation) -> Optional[Decision]:
    return _DECISIONS.pop(str(getattr(conversation, "id", "default")), None)


class _DecisionObservation(Observation):
    """回执。"""


# --- stage choices ---


class FinishOrientAction(Action):
    pass


class _FinishOrientExecutor(ToolExecutor):
    def __call__(
        self, action: FinishOrientAction, conversation=None
    ) -> _DecisionObservation:
        _stash(conversation, Decision(kind="finish_orient", payload={}))
        return _DecisionObservation.from_text(_FINISH_ORIENT["observation"])


class FinishOrientTool(ToolDefinition[FinishOrientAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["FinishOrientTool"]:
        return [
            cls(
                name="finish_orient",
                description=_FINISH_ORIENT["tool"],
                action_type=FinishOrientAction,
                observation_type=_DecisionObservation,
                executor=_FinishOrientExecutor(),
            )
        ]


class ChooseWorkModeAction(Action):
    mode: Literal["self", "delegate"] = Field(description=_CHOOSE_WORK_MODE["mode"])


class _ChooseWorkModeExecutor(ToolExecutor):
    def __call__(
        self, action: ChooseWorkModeAction, conversation=None
    ) -> _DecisionObservation:
        _stash(
            conversation,
            Decision(kind="choose_work_mode", payload=action.model_dump()),
        )
        return _DecisionObservation.from_text(_CHOOSE_WORK_MODE["observation"])


class ChooseWorkModeTool(
    ToolDefinition[ChooseWorkModeAction, _DecisionObservation]
):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ChooseWorkModeTool"]:
        return [
            cls(
                name="choose_work_mode",
                description=_CHOOSE_WORK_MODE["tool"],
                action_type=ChooseWorkModeAction,
                observation_type=_DecisionObservation,
                executor=_ChooseWorkModeExecutor(),
            )
        ]


class ChooseOrgModeAction(Action):
    mode: Literal["sequence", "parallel"] = Field(
        description=_CHOOSE_ORG_MODE["mode"]
    )


class _ChooseOrgModeExecutor(ToolExecutor):
    def __call__(
        self, action: ChooseOrgModeAction, conversation=None
    ) -> _DecisionObservation:
        _stash(
            conversation,
            Decision(kind="choose_org_mode", payload=action.model_dump()),
        )
        return _DecisionObservation.from_text(_CHOOSE_ORG_MODE["observation"])


class ChooseOrgModeTool(ToolDefinition[ChooseOrgModeAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ChooseOrgModeTool"]:
        return [
            cls(
                name="choose_org_mode",
                description=_CHOOSE_ORG_MODE["tool"],
                action_type=ChooseOrgModeAction,
                observation_type=_DecisionObservation,
                executor=_ChooseOrgModeExecutor(),
            )
        ]


# --- write_plan ---


class WritePlanAction(Action):
    plan: str = Field(description=_WRITE_PLAN["plan"])


class _WritePlanExecutor(ToolExecutor):
    def __call__(
        self, action: WritePlanAction, conversation=None
    ) -> _DecisionObservation:
        _stash(conversation, Decision(kind="write_plan", payload=action.model_dump()))
        return _DecisionObservation.from_text(_WRITE_PLAN["observation"])


class WritePlanTool(ToolDefinition[WritePlanAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["WritePlanTool"]:
        return [
            cls(
                name="write_plan",
                description=_WRITE_PLAN["tool"],
                action_type=WritePlanAction,
                observation_type=_DecisionObservation,
                executor=_WritePlanExecutor(),
            )
        ]


# --- create groups ---


class CreateNodeGroupAction(Action):
    count: int = Field(description=_CREATE_PARALLEL["count"])


class _CreateParallelGroupExecutor(ToolExecutor):
    def __call__(
        self, action: CreateNodeGroupAction, conversation=None
    ) -> _DecisionObservation:
        if action.count < 1:
            return _DecisionObservation.from_text("count 必须大于 0。", is_error=True)
        _stash(
            conversation,
            Decision(
                kind="create_parallel_group",
                payload=action.model_dump(),
            ),
        )
        return _DecisionObservation.from_text(_CREATE_PARALLEL["observation"])


class CreateParallelNodeGroupTool(
    ToolDefinition[CreateNodeGroupAction, _DecisionObservation]
):
    @classmethod
    def create(
        cls, conv_state=None, **params
    ) -> Sequence["CreateParallelNodeGroupTool"]:
        return [
            cls(
                name="create_parallel_group",
                description=_CREATE_PARALLEL["tool"],
                action_type=CreateNodeGroupAction,
                observation_type=_DecisionObservation,
                executor=_CreateParallelGroupExecutor(),
            )
        ]


class _CreateSequenceGroupExecutor(ToolExecutor):
    def __call__(
        self, action: CreateNodeGroupAction, conversation=None
    ) -> _DecisionObservation:
        if action.count < 1:
            return _DecisionObservation.from_text("count 必须大于 0。", is_error=True)
        _stash(
            conversation,
            Decision(kind="create_sequence_group", payload=action.model_dump()),
        )
        return _DecisionObservation.from_text(_CREATE_SEQUENCE["observation"])


class CreateSequenceNodeGroupTool(
    ToolDefinition[CreateNodeGroupAction, _DecisionObservation]
):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["CreateSequenceNodeGroupTool"]:
        return [
            cls(
                name="create_sequence_group",
                description=_CREATE_SEQUENCE["tool"],
                action_type=CreateNodeGroupAction,
                observation_type=_DecisionObservation,
                executor=_CreateSequenceGroupExecutor(),
            )
        ]

# --- dispatch / sequence ---


class DispatchAction(Action):
    pass


class _DispatchExecutor(ToolExecutor):
    def __call__(self, action: DispatchAction, conversation=None) -> _DecisionObservation:
        _stash(conversation, Decision(kind="dispatch", payload={}))
        return _DecisionObservation.from_text(_DISPATCH["observation"])


class DispatchTool(ToolDefinition[DispatchAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["DispatchTool"]:
        return [
            cls(
                name="dispatch",
                description=_DISPATCH["tool"],
                action_type=DispatchAction,
                observation_type=_DecisionObservation,
                executor=_DispatchExecutor(),
            )
        ]


class ContinueChildAction(Action):
    child_id: str = Field(description=_CONTINUE_CHILD["child_id"])
    message: str = Field(description=_CONTINUE_CHILD["message"])


class _ContinueChildExecutor(ToolExecutor):
    def __call__(
        self, action: ContinueChildAction, conversation=None
    ) -> _DecisionObservation:
        _stash(
            conversation,
            Decision(kind="continue_child", payload=action.model_dump()),
        )
        return _DecisionObservation.from_text(_CONTINUE_CHILD["observation"])


class ContinueChildTool(ToolDefinition[ContinueChildAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ContinueChildTool"]:
        return [
            cls(
                name="continue_child",
                description=_CONTINUE_CHILD["tool"],
                action_type=ContinueChildAction,
                observation_type=_DecisionObservation,
                executor=_ContinueChildExecutor(),
            )
        ]


# --- questions ---


class AskParentAction(Action):
    question: str = Field(description=_ASK_PARENT["question"])


class _AskParentExecutor(ToolExecutor):
    def __call__(
        self, action: AskParentAction, conversation=None
    ) -> _DecisionObservation:
        _stash(conversation, Decision(kind="ask_parent", payload=action.model_dump()))
        return _DecisionObservation.from_text(_ASK_PARENT["observation"])


class AskParentTool(ToolDefinition[AskParentAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["AskParentTool"]:
        return [
            cls(
                name="ask_parent",
                description=_ASK_PARENT["tool"],
                action_type=AskParentAction,
                observation_type=_DecisionObservation,
                executor=_AskParentExecutor(),
            )
        ]


class AnswerChildQuestionAction(Action):
    question_id: str = Field(description=_ANSWER_CHILD["question_id"])
    answer: str = Field(description=_ANSWER_CHILD["answer"])


class _AnswerChildQuestionExecutor(ToolExecutor):
    def __call__(
        self, action: AnswerChildQuestionAction, conversation=None
    ) -> _DecisionObservation:
        _stash(
            conversation,
            Decision(kind="answer_child_question", payload=action.model_dump()),
        )
        return _DecisionObservation.from_text(_ANSWER_CHILD["observation"])


class AnswerChildQuestionTool(
    ToolDefinition[AnswerChildQuestionAction, _DecisionObservation]
):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["AnswerChildQuestionTool"]:
        return [
            cls(
                name="answer_child_question",
                description=_ANSWER_CHILD["tool"],
                action_type=AnswerChildQuestionAction,
                observation_type=_DecisionObservation,
                executor=_AnswerChildQuestionExecutor(),
            )
        ]


# --- change plan / return ---


class ChangePlanAction(Action):
    failure_summary: str = Field(description=_CHANGE_PLAN["failure_summary"])
    plan: str = Field(description=_CHANGE_PLAN["plan"])


class _ChangePlanExecutor(ToolExecutor):
    def __call__(
        self, action: ChangePlanAction, conversation=None
    ) -> _DecisionObservation:
        _stash(conversation, Decision(kind="change_plan", payload=action.model_dump()))
        return _DecisionObservation.from_text(_CHANGE_PLAN["observation"])


class ChangePlanTool(ToolDefinition[ChangePlanAction, _DecisionObservation]):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ChangePlanTool"]:
        return [
            cls(
                name="change_plan",
                description=_CHANGE_PLAN["tool"],
                action_type=ChangePlanAction,
                observation_type=_DecisionObservation,
                executor=_ChangePlanExecutor(),
            )
        ]


class ReturnToParentAction(Action):
    message: str = Field(description=_RETURN["message"])


class _ReturnToParentExecutor(ToolExecutor):
    def __call__(
        self, action: ReturnToParentAction, conversation=None
    ) -> _DecisionObservation:
        _stash(
            conversation,
            Decision(kind="return_to_parent", payload=action.model_dump()),
        )
        return _DecisionObservation.from_text(_RETURN["observation"])


class ReturnToParentTool(
    ToolDefinition[ReturnToParentAction, _DecisionObservation]
):
    @classmethod
    def create(cls, conv_state=None, **params) -> Sequence["ReturnToParentTool"]:
        return [
            cls(
                name="return_to_parent",
                description=_RETURN["tool"],
                action_type=ReturnToParentAction,
                observation_type=_DecisionObservation,
                executor=_ReturnToParentExecutor(),
            )
        ]


register_tool("FinishOrientTool", FinishOrientTool)
register_tool("ChooseWorkModeTool", ChooseWorkModeTool)
register_tool("ChooseOrgModeTool", ChooseOrgModeTool)
register_tool("WritePlanTool", WritePlanTool)
register_tool("CreateParallelNodeGroupTool", CreateParallelNodeGroupTool)
register_tool("CreateSequenceNodeGroupTool", CreateSequenceNodeGroupTool)
register_tool("DispatchTool", DispatchTool)
register_tool("ContinueChildTool", ContinueChildTool)
register_tool("AskParentTool", AskParentTool)
register_tool("AnswerChildQuestionTool", AnswerChildQuestionTool)
register_tool("ChangePlanTool", ChangePlanTool)
register_tool("ReturnToParentTool", ReturnToParentTool)


def plan_decision_tools(_phase: str) -> list[Tool]:
    return [
        Tool(name="FinishOrientTool"),
        Tool(name="ChooseWorkModeTool"),
        Tool(name="ChooseOrgModeTool"),
        Tool(name="WritePlanTool"),
        Tool(name="CreateParallelNodeGroupTool"),
        Tool(name="CreateSequenceNodeGroupTool"),
        Tool(name="DispatchTool"),
        Tool(name="ContinueChildTool"),
        Tool(name="AskParentTool"),
        Tool(name="AnswerChildQuestionTool"),
        Tool(name="ChangePlanTool"),
        Tool(name="ReturnToParentTool"),
    ]


def language_decision_tools() -> list[Tool]:
    """兼容旧名。"""
    return plan_decision_tools("write_plan")


def self_work_decision_tools() -> list[Tool]:
    return [Tool(name="AskParentTool")]
