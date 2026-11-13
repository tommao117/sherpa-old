# Documentation

## Structure

- `README.md`: project purpose, features, quick start, and documentation links.
- `docs/design/overview.*`: goals, architecture, workflow, module boundaries, and rationale.
- `docs/design/<module>.*`: the design of one module or workflow.
- `docs/plan/overview.*`: roadmap, dependencies, and milestone status.
- `docs/plan/<module>.*`: local tasks, acceptance criteria, progress, and development agreements.

Keep global relationships in overviews and local details in module documents. Use matching module names across design and plan. Create documents only when needed.

## Writing

1. Place content by purpose. Keep implementation progress and experiment records out of design documents; link to their owning plan.
2. Write for readers with no chat history. Define necessary terms and context; remove conversational references, writing-process narration, and dialogue remnants.
3. Explain what the system does, how it works, and why. State essential boundaries once where they matter; avoid repeated disclaimers.
4. Make every paragraph add information. Use short, direct sentences and stable terms. Remove filler, redundant summaries, and repeated content; link instead.
5. Match presentation to the subject: diagrams for relationships, tables for comparisons, lists for steps, and short paragraphs for reasoning. Avoid decorative structure and empty template sections.
6. Choose Markdown or HTML for clarity, not convention alone. Use the simplest format that communicates the subject well. Keep one maintained version of each document.

Before editing, read the relevant overview and module documents. After editing, check scope, standalone readability, concision, links, and diagram consistency. Distinguish target design, implemented behavior, verified results, and open proposals without mixing their document roles.

## Development flow

Proceed along the agreed roadmap to a testable integration, not one helper at a time. Verify and commit coherent stages. Pause for unresolved design choices or expanded permissions; live GPU/model work requires explicit approval.

## Stage commits

Commit each coherent development stage after its relevant checks pass. Review the staged diff and include only intended changes; exclude credentials, local environments, and generated data. Keep commit messages scoped to the stage. Do not push unless explicitly requested.
