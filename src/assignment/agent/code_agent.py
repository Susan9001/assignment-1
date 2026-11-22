"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
  DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
  DEFAULT_COMPACTION_MAX_TOKENS,
  Agent,
  format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
  """An agent that fixes a software issue and submits a git patch."""

  def __init__(
    self,
    task: str,
    environment: Environment,
    model: str | None = None,
    logs_save_path: str | None = None,
    step_limit: int = 100,
    skills_path: str | None = None,
    auto_stop_environment: bool = True,
    compact_threshold_tokens: int | None = None,
    compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
  ):
    super().__init__(
      environment=environment,
      model=model,
      logs_save_path=logs_save_path,
      step_limit=step_limit,
      skills_path=skills_path,
      auto_stop_environment=auto_stop_environment,
      compact_threshold_tokens=compact_threshold_tokens,
      compaction_keep_recent_steps=compaction_keep_recent_steps,
      compaction_max_tokens=compaction_max_tokens,
    )
    self.task = task
    self.submitted_patch = ""

    # 1.3: Advertise the coding tools to the model.
    self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

    # 1.1.b: Construct the system prompt and task_prompt. These
    # should be usable by the `Agent.build_prompt` method.
    system_information = json.dumps(
      {
        "machine": environment.machine,
        "release": environment.release,
        "system": environment.system,
        "version": environment.version,
      },
      indent=2,
    )
    self.system_prompt = (
      "You are a coding agent working on a software task in a repository. "
      "Inspect the relevant code and behavior before making changes. "
      "Fix the underlying issue with focused changes that follow the "
      "repository's conventions. Run relevant checks when possible, "
      "and report what you changed and what you verified.\n\n"
      f"<system_information>\n{system_information}\n</system_information>"
    )
    self.task_prompt = f"Complete the following task:\n\n{task}"
    # 1.4: Advertise skill metadata without revealing full instructions.
    if self.skills:
      catalog = "\n\n".join(skill["metadata"] for skill in self.skills.values())
      self.system_prompt += (
        f"\n\n<available_skills>\n{catalog}\n</available_skills>\n"
        "Call invoke_skill with a skill name to read its full instructions "
        "before using it."
      )

  # 1.3: Dispatch each tool call and return a linked observation.
  def execute_tool_calls(
    self, tool_calls: list[dict[str, Any]]
  ) -> list[dict[str, str]]:
    """Dispatch coding tools and return one observation per call."""

    handlers = {"execute": self._execute, "send_message": self._send_message}
    if self.skills:
      handlers["invoke_skill"] = self._invoke_skill
    observations: list[dict[str, str]] = []
    for call in tool_calls:
      call_id = call.get("id", "") if isinstance(call, dict) else ""
      function = call.get("function") if isinstance(call, dict) else None

      if self.finished:
        content = "Error: the task has already been submitted."
      elif not isinstance(function, dict):
        content = "Error: malformed tool call; missing function."
      else:
        name = function.get("name")
        handler = handlers.get(name) if isinstance(name, str) else None
        if handler is None:
          content = f"Error: unknown tool {name!r}."
        else:
          try:
            arguments = json.loads(function.get("arguments", ""))
          except (json.JSONDecodeError, TypeError) as exc:
            content = f"Error: invalid JSON arguments for {name}: {exc}"
          else:
            if not isinstance(arguments, dict):
              content = f"Error: {name} arguments must be a JSON object."
            else:
              content = handler(arguments)

      observations.append(
        {"role": "tool", "tool_call_id": call_id, "content": content}
      )
    return observations

  def _execute(self, arguments: dict[str, Any]) -> str:
    """Run a validated command in the task environment."""

    allowed = {"command", "shell", "cwd", "timeout", "env"}
    unexpected = set(arguments) - allowed
    command = arguments.get("command")
    if unexpected:
      return f"Error: unexpected execute argument(s): {sorted(unexpected)}."
    if not (
      isinstance(command, str)
      or (
        isinstance(command, list)
        and command
        and all(isinstance(item, str) for item in command)
      )
    ):
      return "Error: execute requires a string or string-list command."
    if (
      "shell" in arguments
      and arguments["shell"] is not None
      and not isinstance(arguments["shell"], bool)
    ):
      return "Error: shell must be a boolean or null."
    if (
      "cwd" in arguments
      and arguments["cwd"] is not None
      and not isinstance(arguments["cwd"], str)
    ):
      return "Error: cwd must be a string or null."
    if "timeout" in arguments and arguments["timeout"] is not None and (
      isinstance(arguments["timeout"], bool)
      or not isinstance(arguments["timeout"], (int, float))
    ):
      return "Error: timeout must be a number or null."
    if "env" in arguments and arguments["env"] is not None and not (
      isinstance(arguments["env"], dict)
      and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in arguments["env"].items()
      )
    ):
      return "Error: env must map strings to strings or be null."
    if isinstance(command, list) and arguments.get("shell", True) is not False:
      return "Error: an argv-list command requires shell=false."
    if isinstance(command, str) and arguments.get("shell") is False:
      return "Error: shell=false requires an argv-list command."

    options = {
      key: arguments[key]
      for key in ("shell", "cwd", "timeout", "env")
      if key in arguments
    }
    return format_tool_output(self.env.execute(command, **options))

  def _send_message(self, arguments: dict[str, Any]) -> str:
    """Report completion and stop the agent."""

    summary = arguments.get("summary")
    if set(arguments) != {"summary"} or not isinstance(summary, str):
      return "Error: send_message requires a summary string."
    print(summary, flush=True)
    self.finished = True
    return "Message sent to the user."

  # 1.4: Reveal one skill's instructions only when invoked.
  def _invoke_skill(self, arguments: dict[str, Any]) -> str:
    """Reveal the complete instructions for one loaded skill."""

    skill_name = arguments.get("name")
    if set(arguments) != {"name"} or not isinstance(skill_name, str):
      return "Error: invoke_skill requires a name string."
    skill = self.skills.get(skill_name)
    if skill is None:
      return f"Error: unknown skill {skill_name!r}."
    return skill["content"]
