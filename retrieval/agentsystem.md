---
title: 'AI Agents for Production - lessons learned'
date: 2025-12-30
tags:
  - Agent system
  - AI in prodcution
---

Over the past year, I’ve seen a lot of impressive AI agent demos, also built some for personal use. They reason, call tools, coordinate with other agents. Most of them look great in a notebook or a recorded demo. Very few survive their first few weeks in production.

This post isn’t a checklist or a guide. It’s a reflection on what I’ve learned while trying to run AI agents as real systems in production, not experiments.



## The Difference Between “It Works” and “It Holds Up”

When people talk about AI agents, the conversation usually starts with prompts, tools, memory, or reasoning loops. Those things matter, but they’re rarely what decides success or failure once users show up.

In production, the questions shift. You start worrying about where the data comes from, how often it changes, and what happens when the model is slow, expensive, or simply wrong. You discover that most failures aren’t dramatic, they’re subtle, recurring, and hard to explain.

I’ve rarely seen a system fail because the prompt was bad. I’ve seen many fail because nobody thought carefully about the system surrounding the model.


## Designing the System Changes Everything

One of the biggest shifts in how I think about agents was moving away from asking how “smart” an agent is and toward asking how the system behaves end-to-end.

A production agent isn’t an LLM call. It’s a pipeline that starts long before inference and continues long after a response is generated. Data has to be ingested, transformed, stored, retrieved, and served reliably. Each step introduces its own constraints and failure modes.

In one project, the agent logic itself was solid. The real issue was upstream: embeddings were being regenerated far more often than expected. Cold starts became common, latency spiked, and a small schema change triggered a full reprocessing job. Nothing about the agent reasoning was wrong, but the system design made it unusable at scale.

That experience changed how I approach new projects. Most production problems show up before the model ever sees a token.

## Cost and Latency Are Design Inputs, Not Afterthoughts

In demos, cost is invisible. In production, it becomes impossible to ignore.

Token usage accumulates faster than intuition suggests. A slightly longer prompt here, an extra tool call there, and suddenly the system is both slow and expensive. Bigger models can improve quality, but only until latency becomes unacceptable. More reasoning steps can help, but they compound unpredictably.

There isn’t a single correct tradeoff, but there is a wrong approach: not thinking about tradeoffs at all. Estimating cost before shipping, separating latency-sensitive paths from offline processing, and being intentional about caching made a bigger difference for me than any prompt tweak.

From a leadership perspective, unpredictability turns into risk. From an engineering perspective, it usually turns into rushed rewrites.


## Sometimes the Best Agent Uses Fewer Models

One uncomfortable realization I’ve had more than once is that some problems don’t need an agent at all.

Some of the most stable systems I’ve worked on leaned heavily on classical information retrieval, deterministic ranking, and simple heuristics. LLMs were used sparingly, often only at the final step where language generation actually added value.

In one case, replacing a multi-step agent loop with structured retrieval followed by a single constrained generation reduced cost and latency significantly. Users reported better results, not worse. The system became easier to debug and easier to evolve.

LLMs are powerful tools, but production systems benefit from restraint. Predictability often matters more than cleverness.


## Data Work Dominates Everything Else

Fine-tuning sounds exciting. Dataset construction rarely does.

In practice, user behavior is noisy, feedback is biased, and logs are incomplete. Before any training begins, there’s a long stretch of work that involves defining what “good” even means and deciding which failures are worth optimizing for.

When fine-tuning does make sense, it introduces a new set of production concerns. Models need to be served reliably, compared against previous versions, and rolled back when things go wrong. This is where MLOps stops being optional, even in LLM-heavy systems.

The hard part isn’t training. It’s trusting the data enough to act on it.


## Observability Is What Turns Guessing Into Learning

One of the most painful lessons I’ve learned is that without observability, teams end up guessing. Was the failure caused by retrieval? By the prompt? By the model? By the data changing underneath?

Production systems need visibility into latency, cost, retrieval quality, outputs, and user feedback - not to chase perfect metrics, but to understand reality. When something breaks, you want to know where and why, not speculate.

If you can’t explain a failure, you can’t fix it. And you definitely can’t scale the system. In another system, I shipped a small prompt change that tested well offline. Nothing crashed after the rollout. Latency stayed flat. Costs didn’t spike. It looked like a harmless improvement, but its not. It turned out the change subtly increased fallback behavior under specific input patterns. Without tracing or structured output metrics, I couldn’t immediately tell whether the issue was retrieval, prompting, or the model itself. The system was “working,” but it had stopped telling us when quality degraded.



## Change Is Inevitable - Make It Safe

Another thing I underestimated early on was how hard it is to change AI systems safely.

Prompts evolve. Models get replaced. Embeddings need to be regenerated. Retrieval logic improves. Every one of those changes can silently degrade user experience if it’s rolled out carelessly.

The ability to deploy gradually, compare versions, and roll back quickly often matters more than the improvement itself. In production, undoing a change is sometimes the most important feature you have.


## Determinism Is Underrated

At first, I accepted non-determinism in LLM driven Agent system as a given. Over time, I started treating determinism as a design goal.

That doesn’t mean eliminating creativity entirely. It means being deliberate about where variability is allowed and where it isn’t. Critical paths benefit from constrained outputs, versioned prompts, and predictable behavior. Creative generation can live at the edges.

The closer a component is to core functionality, the less freedom it should have.


## Real Users Will Surprise You

Once a system is live, users will interact with it in ways you didn’t anticipate. Some will push boundaries accidentally. Others will do it on purpose.

Prompt injection, malformed inputs, and unexpected usage patterns aren’t edge cases, they’re normal. Handling them isn’t just about security; it’s about reliability. Guardrails and validation are part of making the system trustworthy, not bureaucratic overhead.


## Humans Never Fully Disappear

Even systems designed to be fully automated end up involving people. Someone reviews failures, handles escalations, or corrects outputs. Designing those human touchpoints intentionally is far better than discovering them during an incident.

Ignoring this reality doesn’t remove humans from the loop. It just makes their involvement chaotic.


## Production Is a Long Game

One last lesson I wish I’d learned earlier: AI systems decay.

Data drifts. User expectations change. Prompts accumulate complexity. Edge cases multiply. A system that can’t evolve safely eventually becomes brittle, no matter how impressive it was at launch.

Shipping is only the beginning. Surviving change is the real challenge.

The most effective engineers constantly asking what happens when something breaks, how we know it’s working, and whether we can change it safely months from now.

AI agents are easy to build.
Resilient AI systems are hard to operate.

---

In the next post, I’ll share my thoughts on how to choose between agent frameworks, RAG approaches and context engineering, and when each approach actually makes sense in production.

Tools change quickly.
Production constraints don’t.
