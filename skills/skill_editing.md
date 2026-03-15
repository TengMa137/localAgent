---
name: "Skill Improvement"
description: "Guidelines for improving and creating new skills"
version: "1.0.0"
tags: ["meta", "learning"]
author: "Matt"
readonly: true
---

# Skill Improvement Guidelines

## When to Improve a Skill
- You repeatedly give similar advice not captured in any skill
- User asks about something not covered by existing skills
- You find contradictory information in web research
- User provides feedback that a skill is incomplete

## How to Propose Changes
1. Identify which skill needs improvement (use `list_skills` first)
2. Load current version with `load_skill(skill_name)`
3. Formulate specific improvement
4. Explain rationale clearly
5. Suggest the change to user for approval

## Creating New Skills
If no existing skill covers a topic:
1. Identify the gap (e.g., "sleep optimization" not covered)
2. Research the topic thoroughly (use web_search if needed)
3. Propose new skill structure to user
4. Let user create the skill file, then you can refine it

## Quality Standards
- Skills should be actionable, not just information dumps
- Include examples where helpful
- Reference authoritative sources
- Keep skills focused (one topic per skill)
- Cross-reference related skills