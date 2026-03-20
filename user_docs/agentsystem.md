---
title: 'Designing AI Agent Systems Under Real-World Constraints'
date: 2025-12-30
tags:
  - Agent systems
  - AI systems
---

Over the past year I’ve seen many impressive AI agent demos, and built a few myself. They reason, call tools, and coordinate with other agents. Most of them work well in notebooks or recorded demos.

Very few hold up once they become part of a real system.

This post isn’t a guide or checklist. It’s a reflection on what I’ve learned while trying to run agent-based systems under real-world constraints rather than controlled experiments.


### The Gap Between “Working” and “Holding Up”

Discussions about AI agents often start with prompts, tools, memory, or reasoning loops. Those pieces matter, but they’re rarely what determines whether a system survives real usage.

Once users arrive, the questions change.  
Where does the data come from? How often does it change? What happens when the model is slow, expensive, or simply wrong?

Most failures aren’t dramatic. They’re subtle, recurring, and hard to explain. I’ve seen many fail because the surrounding system wasn’t designed carefully.


### System Design Matters More Than Agent Intelligence

One shift that changed how I approach agents was moving from asking *how smart the agent is* to asking *how the whole system behaves*.

An agent system is rarely just an LLM call. It’s a pipeline that starts long before inference and continues after a response is generated. Data needs to be ingested, transformed, stored, retrieved, and served reliably. Each stage introduces its own constraints.

In one project, the agent logic worked well. The problem appeared upstream: embeddings were regenerated far more frequently than expected. Cold starts became common, latency spiked, and a small schema change triggered a full reprocessing job. Nothing was wrong with the reasoning loop. The system around it simply wasn’t designed for the workload.

That experience reshaped how I think about agent systems.  
Most problems appear before the model ever sees a token.


### Cost and Latency Are Design Inputs

In demos, cost is invisible. In real systems, it quickly becomes impossible to ignore.

Token usage accumulates faster than intuition suggests. A slightly longer prompt, an additional tool call, and the system becomes both slower and more expensive. Larger models may improve quality, but only until latency becomes unacceptable. Additional reasoning steps may help, but they also compound unpredictably.

There isn’t a single correct tradeoff, but ignoring the tradeoff entirely is usually the mistake.

Estimating cost early, separating latency-sensitive paths from offline work, and being intentional about caching often mattered more than prompt tweaks.


### Sometimes the Best Agent Uses Fewer Models

One uncomfortable realization I’ve had more than once is that some problems don’t need an agent at all.

Some of the most stable systems I’ve seen and worked with relied heavily on classical information retrieval, deterministic ranking, and simple heuristics. LLMs were used sparingly, often only at the final step where language generation actually added value, e.g. replacing a multi-step agent loop with structured retrieval followed by a constrained generation step reduced both cost and latency, and the system became much easier to debug.

LLMs are powerful tools. Production systems often benefit from restraint.


### Data Work Dominates Everything Else

Fine-tuning sounds exciting. Dataset construction rarely does.

User behavior is noisy, feedback is biased, and logs are incomplete. Before any training begins, there is usually a long phase of defining what “good” even means and deciding which failures matter.

When fine-tuning is introduced, it also adds operational complexity: models must be served reliably, compared against previous versions, and rolled back when necessary.

The difficult part isn’t training. It’s trusting the data enough to act on it.


### Observability Turns Guessing Into Learning

Without observability, teams end up guessing.

Was a failure caused by retrieval? By prompting? By the model? Or by the underlying data changing?

Real systems need visibility into latency, cost, retrieval quality, outputs, and user feedback. Not to chase perfect metrics, but to understand what is actually happening. In my local agent project, I made a prompt change that looked harmless. Latency and costs stayed stable, but the change subtly increased fallback behavior for certain inputs. Without tracing and structured output metrics, it took time to understand what had happened.

The system was still “working”, but it had stopped telling us when quality degraded.


### Change Is Inevitable — Make It Safe

AI systems evolve constantly. Prompts change, models are replaced, embeddings are regenerated, and retrieval logic improves. Each of these changes can silently degrade user experience if rolled out carelessly. The ability to deploy gradually, compare versions, and roll back quickly often matters more than the improvement itself.


### Determinism Is Underrated

Early on, I accepted non-determinism in LLM systems as unavoidable. Over time, I started treating determinism as a design goal. That doesn’t mean removing creativity. It means deciding carefully where variability belongs.

Core system paths benefit from constrained outputs, versioned prompts, and predictable behavior. Creative generation can live at the edges. The closer a component is to core functionality, the less freedom it should have.


### Users Will Surprise You

Once a system is live, people will use it in ways you didn’t expect. Prompt injection, malformed inputs, and unusual usage patterns are normal, not edge cases. Guardrails and validation aren’t bureaucratic overhead — they’re part of building a reliable system.


### Humans Never Fully Disappear

Even highly automated systems involve people somewhere in the loop. Someone reviews failures, handles escalations, or corrects outputs.

Designing those human touchpoints intentionally is far better than discovering them during an incident.


### Building AI Systems Is a Long Game

AI systems change over time. Data drifts, user expectations evolve, and edge cases accumulate.

A system that can’t evolve safely eventually becomes brittle, no matter how impressive it looked at launch. Shipping is only the beginning. The real challenge is operating and evolving the system over time.

Building an AI agent is easy. Building a resilient AI system is much harder.