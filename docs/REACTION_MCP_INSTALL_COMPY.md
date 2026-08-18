# Installing the reaction sandbox MCP (`pflotran_mcp`) on Compy

> **Superseded (2026-08-18).** This recipe describes the original port: an
> unzipped copy of the upstream repository (`reaction_sandbox_mcp-main/`),
> served through a launcher and a `fastmcp` shim under
> `mcp/reaction-sandbox-mcp/`. That launcher and shim are **deleted**. The live
> server is the git checkout `reaction_sandbox_mcp-upstream/` on branch
> `compy-port`, installed **editable** into the `ideas` env, so `mcp_config.json`
> launches it directly as the console script `pflotran-mcp` — see
> `docs/CLAUDE_CODE_MCP_GUIDE.md`, "Which tree is live". Kept for the record of
> how the port was done, not as instructions to follow.


A standalone recipe for setting this server up in **your own account** on
`compy01.pnl.gov`. It does not install or assume the IDEAS framework — when
you are done you have a working MCP server that any MCP client can talk to.

Naming, because three names refer to the same thing: the upstream repository is
`reaction_sandbox_mcp`, its Python package is `pflotran_mcp`, and it is
conventionally registered with a client under the short name **`reaction`**.

Verified on Compy against PFLOTRAN v7.0 / PETSc 3.21.6.

---

## 0. What you are building

Four pieces. Only the first is upstream code:

| piece | why it is needed |
|---|---|
| PFLOTRAN + PETSc | the server shells out to a PFLOTRAN binary; the LAMBDA sandbox must be compiled in |
| upstream `reaction_sandbox_mcp` | defines the 39 tools |
| a `fastmcp` shim | upstream imports a package that is not installed |
| a launcher `main.py` | upstream has **no entry point** — it never calls `mcp.run()` |

The shim and launcher live outside the upstream tree so upstream stays
**unmodified and pullable**. Do not edit anything inside it.

Pick your locations once and use them throughout:

```bash
export SRC=$HOME/reaction_sandbox_mcp-main         # upstream, unmodified
export MCPDIR=$HOME/mcp/reaction-sandbox-mcp       # the shim + launcher you write
mkdir -p $MCPDIR
```

Budget: ~1 hour, almost all of it PETSc.

---

## 1. Toolchain

```bash
module load gcc/10.2.0 openmpi/4.0.1
```

Compy is **CentOS 7 with glibc 2.17**. Before any `pip install`:

```bash
export PIP_ONLY_BINARY=":all:"
```

Without it, modern `manylinux_2_28` wheels refuse to install and the sdist
fallback fails to compile against gcc 4.8.5.

```bash
conda create -y -n ideas python=3.12
conda activate ideas
pip install mcp
python3 -c "from mcp.server.fastmcp import FastMCP; print('FastMCP ok')"
```

`mcp` supplies FastMCP. You do **not** need the standalone `fastmcp`
distribution — step 4a bridges that gap.

---

## 2. PETSc 3.21.6

The long step. PFLOTRAN v7.0 needs PETSc >= 3.21.4.

```bash
cd $HOME
git clone -b v3.21.6 https://gitlab.com/petsc/petsc.git petsc-pflotran
cd petsc-pflotran
export PETSC_DIR=$PWD
export PETSC_ARCH=arch-compy-opt
```

**Download the HDF5 tarball by hand first.** PETSc's downloader fails on
Compy's outdated TLS stack, and the failure surfaces deep in `configure` as an
unrelated-looking error:

```bash
mkdir -p $HOME/petsc-pkgs && cd $HOME/petsc-pkgs
wget https://web.cels.anl.gov/projects/petsc/download/externalpackages/hdf5-1.14.3-p1.tar.bz2
```

Then configure against the local tarball:

```bash
cd $PETSC_DIR
./configure PETSC_ARCH=arch-compy-opt \
  --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
  --with-debugging=0 COPTFLAGS=-O2 CXXOPTFLAGS=-O2 FOPTFLAGS=-O2 \
  --download-hdf5=$HOME/petsc-pkgs/hdf5-1.14.3-p1.tar.bz2 \
  --with-hdf5-fortran-bindings=yes \
  --download-fblaslapack=yes --download-metis=yes --download-parmetis=yes
make all
```

---

## 3. PFLOTRAN v7.0

```bash
cd $HOME
git clone https://bitbucket.org/pflotran/pflotran.git
cd pflotran && git checkout v7.0
cd src/pflotran && make -j8 pflotran
```

**The LAMBDA sandbox ships stock in v7.0 — no patching.** Confirm all three:

```bash
ls $HOME/pflotran/src/pflotran/reaction_sandbox_pnnl_lambda.F90
ls $HOME/pflotran/database/lambda.dat
ls $HOME/pflotran/regression_tests/default/reaction_sandbox/reaction_sandbox_lambda.in
```

Smoke-test against the build's own gold standard:

```bash
cd $HOME/pflotran/regression_tests/default/reaction_sandbox
$HOME/pflotran/src/pflotran/pflotran -pflotranin reaction_sandbox_lambda.in
echo "exit=$?"        # expect 0
```

---

## 4. Upstream repo + the two missing pieces

```bash
cd $HOME
# unzip the release, or clone it, so that $SRC/server.py exists
ls $SRC/server.py
```

### 4a. The shim

Upstream does `from fastmcp import FastMCP`. **The failure mode is silent**:
upstream ships a fallback `mcp_server.SimpleMCP` whose `run()` *prints*
instead of speaking MCP, so without the shim the server appears to start and
is dead.

```bash
mkdir -p $MCPDIR/_shim
cat > $MCPDIR/_shim/fastmcp.py <<'EOF'
"""Shim: reaction_sandbox_mcp does `from fastmcp import FastMCP`.

The standalone `fastmcp` distribution is not installed, and the server's
bundled fallback (mcp_server.SimpleMCP) is a placeholder whose run() does not
speak the MCP protocol. The official `mcp` package ships an equivalent
FastMCP, so re-export it under the name the server expects. This keeps the
upstream repository unmodified and pullable.
"""
from mcp.server.fastmcp import FastMCP  # noqa: F401
EOF
```

### 4b. The launcher

`server.py` defines 39 tools and **never calls `mcp.run()`**, so as shipped it
cannot be launched at all.

> **Set every path here, with a default — do not rely on your shell.**
> An MCP client does not pass your environment to the server. The MCP SDK's
> `get_default_environment()` forwards only `HOME`, `LOGNAME`, `PATH`, `SHELL`
> and `USER`. Anything you `export` in a profile script is invisible to the
> server unless the client happens to forward the full environment.
>
> This bites hardest on `LAMBDA_PFLOTRAN_DIR`: upstream's
> `tools/lambda_pipeline.py` reads it to put the Lambda package on `sys.path`
> and otherwise falls back to its author's home directory. Miss it and every
> lambda tool returns
> `"Lambda-PFLOTRAN not available: No module named 'preprocessing'"`
> even though the package is sitting right there.

```bash
cat > $MCPDIR/main.py <<EOF
#!/usr/bin/env python3
"""Launcher for the reaction_sandbox_mcp server (PFLOTRAN reactive transport).

Upstream has no entry point and imports a \`fastmcp\` package that is absent
here. This supplies both without editing upstream code.
"""
import os
import sys
from pathlib import Path

SERVER_DIR = Path(os.getenv("REACTION_MCP_DIR", "$SRC"))
SHIM_DIR = Path(__file__).resolve().parent / "_shim"

# Defaults, because an MCP client forwards almost no environment.
os.environ.setdefault("PFLOTRAN_EXECUTABLE",
                      "$HOME/pflotran/src/pflotran/pflotran")
os.environ.setdefault("MPI_COMMAND", "mpirun")
os.environ.setdefault("LAMBDA_PFLOTRAN_DIR",
                      str(SERVER_DIR / "lambda_pflotran_refactor"))

if not SERVER_DIR.is_dir():
    sys.exit(f"reaction MCP: server dir not found: {SERVER_DIR}")

sys.path.insert(0, str(SHIM_DIR))       # \`fastmcp\` -> mcp.server.fastmcp
sys.path.insert(0, str(SERVER_DIR))     # upstream \`server\` and \`tools\` package
os.chdir(SERVER_DIR)                    # upstream resolves some paths relatively

import server  # noqa: E402

if __name__ == "__main__":
    server.mcp.run(transport="stdio")
EOF
```

Three details that are load-bearing:

- the shim goes on `sys.path` **before** upstream, or upstream's placeholder wins;
- the `chdir` is required — upstream resolves several paths relative to its own directory;
- `MPI_COMMAND` is `mpirun` on Compy. On a Slurm-native site (e.g. NERSC) use `srun`.

---

## 5. Verify — before wiring it to any client

Save this as `$HOME/verify_reaction_mcp.py`. It speaks stdio to the server
directly and needs nothing else installed.

```python
#!/usr/bin/env python3
"""Standalone check that the reaction MCP serves over stdio."""
import asyncio, json, sys, tempfile
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LAUNCHER = sys.argv[1]

async def main():
    params = StdioServerParameters(command=sys.executable, args=[LAUNCHER])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = [t.name for t in (await s.list_tools()).tools]
            print("tools:", len(names))
            print("lambda:", [n for n in names if "lambda" in n.lower()])

            out = await s.call_tool("run_lambda_binning",
                {"sample_id": "SPS_0001", "binning_method": "uniform",
                 "n_bins": 5, "output_dir": tempfile.mkdtemp()})
            d = json.loads(out.content[0].text)
            print("binning:", d.get("validation_status"),
                  d.get("error") or d.get("reaction_network_file"))

asyncio.run(main())
```

Run it:

```bash
conda activate ideas
python3 $HOME/verify_reaction_mcp.py $MCPDIR/main.py
```

Expected:

```
tools: 39
lambda: ['run_lambda_preprocessing', 'run_lambda_binning', 'generate_lambda_reaction_database']
binning: success /tmp/…/binning/SPS_0001/uniform_mean/reaction_network.txt
```

Note this script does **not** forward your environment — that is deliberate.
If it passes, the launcher's defaults are correct and the server will work
under any client.

Reading the failures:

| output | meaning |
|---|---|
| fewer than 39 tools, or a hang | the shim is not first on `sys.path`; upstream's `SimpleMCP` placeholder won |
| `binning: failed … No module named 'preprocessing'` | `LAMBDA_PFLOTRAN_DIR` default is wrong or missing in the launcher |
| `server dir not found` | `$SRC` is wrong |

---

## 6. Register it with your MCP client

The server speaks stdio, so any MCP client can use it. Give it the **conda
interpreter by absolute path** — not a bare `python3`, which will not be the
env you built.

```json
"pflotran": {
  "command": "/people/<you>/.conda/envs/ideas/bin/python3",
  "args": ["/people/<you>/mcp/reaction-sandbox-mcp/main.py"],
  "timeout": 300.0
}
```

The generous timeout is deliberate: binning and simulation tools are far
slower than typical HTTP-backed MCP servers.

---

## 7. Two behaviours to know before relying on it

Both measured on this install, neither documented upstream.

**`binning_method` has no effect.** `uniform`, `cumulative`, `class_based` and
`bulk` return byte-identical networks at every bin count tested. The server
routes to a reconstructed binning package that implements neither those four
methods nor their semantics. Selecting one is not a choice.

**Use `n_bins=5`.** Donor species are named for each bin's mean carbon number,
*rounded*, so bins whose means round to the same integer collapse into one
species — which silently merges two distinct organic-matter pools. For sample
`SPS_0001`: `n=3` gives `C21, C24, C21`; `n=4` gives two collisions; `n=5` is
distinct. This is a property of one sample's carbon distribution, not a
general rule — check any new sample by reading the donor names out of the
generated `reaction_network.txt`.

---

## Troubleshooting

| symptom | cause |
|---|---|
| server "starts" but no tools appear | `fastmcp` shim not first on `sys.path` |
| `No module named 'preprocessing'` | `LAMBDA_PFLOTRAN_DIR` not set **in the launcher** — the client did not forward your shell env |
| `server dir not found` | `REACTION_MCP_DIR` / `$SRC` wrong |
| PETSc `configure` fails downloading HDF5 | Compy's TLS is too old — supply the tarball locally (step 2) |
| pip installs fail to build | `export PIP_ONLY_BINARY=":all:"` |
| tools time out | raise `timeout` in the client config |
| MPI errors when a tool runs a simulation | `MPI_COMMAND` is `mpirun` on Compy, `srun` on Slurm-native sites |
