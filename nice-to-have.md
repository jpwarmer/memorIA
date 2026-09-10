# Nice to Have

Improvements and known gaps to address as memorIA matures beyond MVP.

## Memory Lifecycle

- [x] `memory_delete` tool — allow agents and users to remove outdated or wrong memories
- [x] `memory_update` / `memory_supersede` — in-place edit vs atomic replace that keeps the old vector for audit
- [ ] TTL / expiration — especially for `session` and `cache` scopes, auto-expire after N days
- [ ] Relevance decay — older memories should rank lower in search unless explicitly pinned
- [x] Duplicate detection — `memory_dedupe` reports clusters; does not auto-merge
- [ ] Memory compaction — periodically merge related memories into a single, updated entry

## Quality & Reliability

- [ ] Validation layer on save — reject empty, too-short, or low-signal memories before they hit Qdrant
- [ ] Confidence scoring — tag memories with a confidence level so agents can weigh them appropriately
- [ ] Source tracking — link memories back to the conversation/session that created them for auditability
- [ ] Conflict detection — flag when a new memory contradicts an existing one in the same scope

## Search

- [x] Parent + chunk vectors — search chunks, return de-duplicated parents
- [x] `min_score` on search — stop returning the least-bad unmatched hit
- [ ] Hybrid search — combine semantic (vector) with keyword (BM25) to catch cases where intent doesn't match wording
- [ ] Search filters by date range — "what did I learn last week" without scanning everything
- [ ] Relevance feedback loop — let agents mark search results as helpful/not to improve ranking over time

## Observability

- [x] Memory stats endpoint — counts, token histogram, truncation_exposure, duplicate candidates
- [x] `memory_reindex` — repeatable re-chunk/re-embed after model or window changes
- [ ] Dashboard or CLI to browse/edit/prune memories manually
- [ ] Log when an agent uses a memory so you can track actual utility vs. dead weight

## Integration

- [ ] Sync selected memories to CLAUDE.md automatically (e.g., confirmed conventions)
- [ ] Import from external sources — pull decisions from Notion, Slack threads, or git commit messages
- [ ] Multi-user support — scope memories per user for shared repos
