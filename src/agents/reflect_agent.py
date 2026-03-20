from pydantic import BaseModel
from pydantic_ai import Agent
from typing import List

from .utils import model
from .worker import TaskSpec


class ReflectionOutput(BaseModel):
    objective_complete: bool
    confidence:         float
    next_tasks:         List[TaskSpec]


reflect_agent = Agent(model=model, output_type=ReflectionOutput)


@reflect_agent.system_prompt
def _reflect_prompt() -> str:
    return """
Given current findings and uncertainties decide:

  - objective_complete: true if the question can be answered confidently
  - confidence: 0.0 to 1.0
  - next_tasks: concrete follow-ups if incomplete; empty if done

Avoid repeating tasks already listed as completed.
"""
