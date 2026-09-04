# Agent API boundary

The agent-facing service is scheduled after the deterministic editor and retrieval catalog are
working. Its contract will operate on immutable content-addressed motion IDs, compact diagnostic
events, operator parameters, protected metrics, and artifact paths. It will not expose dense joint
transforms as the normal LLM control surface.

The planned operations are:

```text
retrieve_reference
grade_motion
apply_edit
optimize_edit
make_loop
export_motion
get_history
```

Until that stage, the CLI is the stable integration boundary. Current commands are listed by:

```bash
motionlab --help
```

HTTP schemas must later reuse the Python service schemas, validate all paths inside a configured
workspace, and reject path traversal. No HTTP or LLM orchestration module is created before those
service contracts have real implementations.
