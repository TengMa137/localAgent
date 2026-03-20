from pydantic import BaseModel
from pydantic_ai import Agent

from .utils import model

class SynthesisOutput(BaseModel):
    report: str


synthesis_agent = Agent(model=model, output_type=SynthesisOutput)


@synthesis_agent.system_prompt
def _synth_prompt() -> str:
    return """
Produce a final well-structured research report.

Requirements:
  - Clear conclusion up front
  - Key findings grouped logically
  - Uncertainties stated explicitly
  - Evidence-backed tone

Do not hallucinate citations or sources.
"""