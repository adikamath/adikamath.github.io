---
layout: post
title: "Weekly Notes: Vector Search and AI Startups"
date: 2026-04-07 23:30 -0700
modified: 2026-04-07 23:30 -0700
description: A weekly reflection on billion-scale vector search infrastructure, AI agents, and how company building may be changing.
tags:
  - learning-notes
  - machine-learning
  - vector-search
  - startups
  - ai-agents
---

This week split pretty cleanly across two questions. First, what does it actually take to make AI systems work at production scale? Second, what happens when AI stops being a feature and starts changing how companies are built?

One item was very grounded in infrastructure. The other was broader and a bit more unsettling. Together, they made the same point: the interesting part now is less about novelty and more about the operating choices underneath it.

**Themes this week**

- Search at scale
- AI and startup leverage

## Search at scale

### Powering Billion-Scale Vector Search with OpenSearch

*Article from [Uber](https://www.uber.com/blog/powering-billion-scale-vector-search-with-opensearch/){:target="_blank" rel="noopener noreferrer"}*

- **TL;DR:** Uber used Amazon OpenSearch to support vector search across billion-scale items. What stood out was not just the scale, but the number of practical tuning levers available, including better vector search algorithms and GPU acceleration.
- **Why it mattered:** This was a useful reminder that production vector search is an infrastructure problem as much as a model problem. Relevance matters, but so do latency, cost, and the ability to tune the system without rebuilding everything from scratch.
- **My take:** I liked seeing the logic behind moving from classic keyword search on Apache Lucene toward semantic search across Uber's family of apps. What I still wanted, though, was a clearer comparison against the alternatives. The OpenSearch choice is interesting, but the trade-offs would have been even more useful than the final answer.
- **Practical takeaway:** I want to pressure-test whether this same setup can also support RAG and agent memory, instead of treating those as separate systems by default.

## AI and startup leverage

### The New Way To Build A Startup

*Video on [YouTube](https://youtu.be/rWUWfj_PqmM?si=sJ-PQIjZDVW-T7By){:target="_blank" rel="noopener noreferrer"}*

- **TL;DR:** The argument is that "20x companies" will automate most internal functions with AI agents, letting very small teams punch far above their weight. The two models that stood out were one AI super-employee handling many tasks, or one agent per employee built from that person's workflows and documentation.
- **Why it mattered:** This feels like more than a productivity upgrade. It reads like a rewrite of the startup playbook, where leverage delays hiring, lowers payroll, and lets a company stay small and cohesive for longer.
- **My take:** I think this shift is real, and I think a lot of larger companies are underestimating how fast young startups can use it to move past them. At the same time, the labor side of this is still deeply unresolved. If employee data is used to train systems that can replace parts of their work, that changes the meaning of career growth in a pretty serious way.
- **Practical takeaway:** Any team going deep on AI agents should think early about the employee and career implications, not just the automation upside.

## Patterns across this week's learnings

- The hard part of AI is moving from prototype value to operating value.
- Infrastructure choices and org choices are starting to matter as much as model choices.
- The biggest open questions are now about trade-offs, not raw capability.

## On my radar next

- Whether a vector search stack like OpenSearch can cleanly extend into RAG and long-term agent memory.
- What the best early-stage org design looks like when AI agents are treated as part of the team structure.
- How companies should handle trust, ownership, and career risk when employee workflows become training data.
