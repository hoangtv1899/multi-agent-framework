# Test suite

A **tiered** suite. Plain `pytest` is fast, offline and deterministic; the heavy
or external tiers are opt-in behind flags, so the default run never touches the
network, the LLM, or SLURM.

```bash
cd ~/RCSFA/multi-agent
module load pytorch/2.8.0          # runtime env (openai, mcp, numpy, xarray…)
pytest                             # tier 1 only — offline unit tests
```

## Tiers

| Run | Adds | Needs |
|-----|------|-------|
| `pytest` | offline unit tests (mocked LLM + MCP) | nothing |
| `pytest --runlive` | live MCP data-source tests | network |
| `pytest --runllm` | real reception/planner round-trips | `PNNL_API_KEY` |
| `pytest --runcompute` | ELM build/run on SLURM | an `salloc` node |
| `pytest --runlegacy` | retired PFLOTRAN-era tests (reference) | — (some fail by design) |

Flags compose: `pytest --runlive --runllm` runs offline + live + LLM. The
`live`/`llm`/`compute` markers and the `--run*` flags live in
[conftest.py](../conftest.py) and [pytest.ini](../pytest.ini).

## What's covered (tier 1, offline)

| File | Covers |
|------|--------|
| `test_tool_loop.py` | `ToolLoopAgent.run()` agentic loop: dispatch, `tools=` every round, ask_user, unknown-tool, max-rounds (mocked LLM) |
| `test_expander.py` | `expand()` materialize: grid → bands → allocation → spread → Fan/soil enrich → columns, polygon clip, empty-grid guard (mocked MCP) |
| `test_orchestration.py` | `reception.process` message/brief assembly; `run_pipeline`/`run_session` `_parse_json`, plan printers, `context_from`, `run_planner` wiring |
| `test_agentic.py` | `_build_tools` schemas + allowlist; reception `_parse`; ask_user schema |
| `test_mcp_tools.py` | each MCP server's pure parse logic + expander band/allocate helpers |
| `test_validate.py` | brief/plan deterministic validator |
| `test_columns_adapter.py` | Tier-2 → Tier-3 `columns_to_elm_plan` adapter |
| `test_elm_wrapper.py`, `test_elm_clone_routing.py`, `test_elm_exp_manager_structure.py`, `test_elm_setup_plotting.py`, `test_elm_integration.py` | ELM wrapper, `--keepexe` clone routing, exp-manager structure, plotting, integration glue |

## What's covered (opt-in tiers)

- **live** — `test_live_mcp.py`: all 6 MCP servers start + advertise tools; terrain
  elevation in CONUS range; `resolve_watershed`; Fan WTD; soil; USGS wells.
- **llm** — `test_live_llm.py`: endpoint round-trip; a `tools=` request succeeds for
  the Bedrock-routed Claude model; full reception → parseable brief (also needs `--runlive`).
- **compute** — `test_elm_e2e_minimal.py`: `ELMExpManager.execute_plan()` build+run
  on a node (self-skips off-node even with `--runcompute`).
  `smoke_test_elm.py` is a standalone single-column build→run→analyze script
  (`python tests/smoke_test_elm.py` on an `salloc` node), not collected by pytest.

## Adding a test

- Offline by default — mock the LLM (`SimpleLLMClient`) and MCP clients
  (`call_tool_json` / `list_tools_detailed`); see `test_tool_loop.py` for the fakes.
- Touches the network / LLM / SLURM? Mark it `@pytest.mark.live` / `llm` / `compute`
  so it stays opt-in.

## Legacy

`tests/legacy/` holds the retired PFLOTRAN-era reception/planner/validator tests
and old manual driver scripts — superseded by `reception_llm.py`, the planner
capability probe, `validate.py`, and `tools/run_pipeline.py`. Auto-marked
`legacy` and skipped unless `--runlegacy`. Kept for reference only.
