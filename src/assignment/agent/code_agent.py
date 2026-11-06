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
    # TODO(1.4): If any skills are available to the agent, make their
    # descriptions/metadata available to the agent in the prompt.

  def execute_tool_calls(
    self, tool_calls: list[dict[str, Any]]
  ) -> list[dict[str, str]]:
    """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

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
        if not isinstance(name, str) or name not in {"execute", "send_message"}:
          content = f"Error: unknown tool {name!r}."
        else:
          try:
            arguments = json.loads(function.get("arguments", ""))
          except (json.JSONDecodeError, TypeError) as exc:
            content = f"Error: invalid JSON arguments for {name}: {exc}"
          else:
            if not isinstance(arguments, dict):
              content = f"Error: {name} arguments must be a JSON object."
            elif name == "execute":
              allowed = {"command", "shell", "cwd", "timeout", "env"}
              unexpected = set(arguments) - allowed
              command = arguments.get("command")
              if unexpected:
                content = f"Error: unexpected execute argument(s): {sorted(unexpected)}."
              elif not (
                isinstance(command, str)
                or (
                  isinstance(command, list)
                  and command
                  and all(isinstance(item, str) for item in command)
                )
              ):
                content = "Error: execute requires a string or string-list command."
              elif (
                "shell" in arguments
                and arguments["shell"] is not None
                and not isinstance(arguments["shell"], bool)
              ):
                content = "Error: shell must be a boolean or null."
              elif (
                "cwd" in arguments
                and arguments["cwd"] is not None
                and not isinstance(arguments["cwd"], str)
              ):
                content = "Error: cwd must be a string or null."
              elif "timeout" in arguments and arguments["timeout"] is not None and (
                isinstance(arguments["timeout"], bool)
                or not isinstance(arguments["timeout"], (int, float))
              ):
                content = "Error: timeout must be a number or null."
              elif "env" in arguments and arguments["env"] is not None and not (
                isinstance(arguments["env"], dict)
                and all(
                  isinstance(key, str) and isinstance(value, str)
                  for key, value in arguments["env"].items()
                )
              ):
                content = "Error: env must map strings to strings or be null."
              elif isinstance(command, list) and arguments.get("shell", True) is not False:
                content = "Error: an argv-list command requires shell=false."
              elif isinstance(command, str) and arguments.get("shell") is False:
                content = "Error: shell=false requires an argv-list command."
              else:
                options = {
                  key: arguments[key]
                  for key in ("shell", "cwd", "timeout", "env")
                  if key in arguments
                }
                content = format_tool_output(self.env.execute(command, **options))
            else:
              summary = arguments.get("summary")
              if (
                set(arguments) != {"summary"}
                or not isinstance(summary, str)
              ):
                content = "Error: send_message requires a summary string."
              else:
                print(summary, flush=True)
                self.finished = True
                content = "Message sent to the user."

      observations.append(
        {"role": "tool", "tool_call_id": call_id, "content": content}
      )
    return observations
