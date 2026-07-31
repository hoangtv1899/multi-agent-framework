# Installing the reaction sandbox MCP (`pflotran_mcp`) on Compy

A standalone recipe for setting this up in **your own account** on
`compy01.pnl.gov`. Nothing here depends on anyone else's home directory.

Naming, because three names refer to the same thing: the upstream repository is
`reaction_sandbox_mcp`, its Python package is `pflotran_mcp`, and the framework
registers it in `mcp_config.json` as **`reaction`**.

Everything below was verified on Compy against PFLOTRAN v7.0 / PETSc 3.21.6.

---

## 0. What you are building

Four pieces. Only the first is upstream code:

| piece | why it is needed |
|---|---|
| PFLOTRAN + PETSc | the MCP shells out to a PFLOTRAN binary; the LAMBDA sandbox must be compiled in |
| upstream `reaction_sandbox_mcp` | defines the 39 tools |
| a `fastmcp` shim | upstream imports a package that is not installed |
| a launcher `main.py` | upstream has **no entry point** — it never calls `mcp.run()` |

The shim and launcher exist so upstream stays **unmodified and pullable**. Do
not edit anything inside the upstream tree.

Pick a root and use it throughout:

```bash
export IDEAS=$HOME/IDEAS          # or wherever you want this to live
mkdir -p $IDEAS
```

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

Create the environment (Python 3.12):

```bash
conda create -y -n ideas python=3.12
conda activate ideas
pip install mcp
python3 -c "from mcp.server.fastmcp import FastMCP; print('FastMCP ok')"
```

`mcp` is what supplies FastMCP. You do **not** need the standalone `fastmcp`
distribution — step 4 bridges that gap.

---

## 2. PETSc 3.21.6

This is the longest step (~30–45 min). PFLOTRAN v7.0 needs PETSc >= 3.21.4.

```bash
cd $HOME
git clone -b v3.21.6 https://gitlab.com/petsc/petsc.git petsc-pflotran
cd petsc-pflotran
export PETSC_DIR=$PWD
export PETSC_ARCH=arch-compy-opt
```

**Download the HDF5 tarball by hand first.** PETSc's own downloader fails on
Compy's outdated TLS stack, and the failure appears deep in `configure` as an
unrelated-looking error:

```bash
mkdir -p $HOME/petsc-pkgs && cd $HOME/petsc-pkgs
# fetch hdf5-1.14.3-p1.tar.bz2 from a machine with working TLS, or:
wget https://web.cels.anl.gov/projects/petsc/download/externalpackages/hdf5-1.14.3-p1.tar.bz2
```

Then configure with the local tarball:

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
ls src/pflotran/reaction_sandbox_pnnl_lambda.F90     # the sandbox source
ls database/lambda.dat                               # its thermodynamic database
ls regression_tests/default/reaction_sandbox/reaction_sandbox_lambda.in
```

Smoke-test the build against its own gold standard:

```bash
cd $HOME/pflotran/regression_tests/default/reaction_sandbox
$HOME/pflotran/src/pflotran/pflotran -pflotranin reaction_sandbox_lambda.in
echo "exit=$?"        # expect 0
```

---

## 4. Upstream repo + the two missing pieces

```bash
cd $IDEAS
# unzip the release, or clone it, as reaction_sandbox_mcp-main/
ls reaction_sandbox_mcp-main/server.py   # must exist
```

### 4a. The shim

Upstream does `from fastmcp import FastMCP`. **The failure mode here is
silent**: upstream ships a fallback `mcp_server.SimpleMCP` whose `run()`
*prints* instead of speaking MCP, so without the shim the server appears to
start and is dead.

```bash
mkdir -p $IDEAS/multi-agent-framework/mcp/reaction-sandbox-mcp/_shim
cat > $IDEAS/multi-agent-framework/mcp/reaction-sandbox-mcp/_shim/fastmcp.py <<'EOF'
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
cannot be launched at all. Upstream's config also points at its author's WSL
machine (`/mnt/c/Users/.../OneDrive/...`) with `PFLOTRAN_EXECUTABLE`
defaulting to `/usr/local/bin/pflotran`.

Edit the two paths at the top to match your account:

```bash
cat > $IDEAS/multi-agent-framework/mcp/reaction-sandbox-mcp/main.py <<EOF
#!/usr/bin/env python3
"""Launcher for the reaction_sandbox_mcp server (PFLOTRAN reactive transport).

Upstream has no entry point and imports a \`fastmcp\` package that is absent
here. This supplies both without editing upstream code.

Registered in mcp_config.json as \`reaction\`.
"""
import os
import sys
from pathlib import Path

SERVER_DIR = Path(os.getenv(
    "REACTION_MCP_DIR", "$IDEAS/reaction_sandbox_mcp-main"))
SHIM_DIR = Path(__file__).resolve().parent / "_shim"
PFLOTRAN = "$HOME/pflotran/src/pflotran/pflotran"

os.environ.setdefault("PFLOTRAN_EXECUTABLE", PFLOTRAN)
os.environ.setdefault("MPI_COMMAND", "mpirun")

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

- the shim goes on `sys.path` **before** upstream, or upstream's own fallback wins;
- the `chdir` is required — upstream resolves several paths relative to its own directory;
- `MPI_COMMAND` is `mpirun` on Compy. On a Slurm-native site (e.g. NERSC) it is `srun`.

---

## 5. Environment

Add to your `env_compy.sh` (or equivalent):

```bash
module load gcc/10.2.0 openmpi/4.0.1
conda activate ideas

export PFLOTRAN_DIR=$HOME/pflotran
export PFLOTRAN_EXECUTABLE=$PFLOTRAN_DIR/src/pflotran/pflotran
export MPI_COMMAND=mpirun
export REACTION_MCP_DIR=$IDEAS/reaction_sandbox_mcp-main
export LAMBDA_PFLOTRAN_DIR=$REACTION_MCP_DIR/lambda_pflotran_refactor
export PIP_ONLY_BINARY=":all:"
```

---

## 6. Register it with the framework

In `multi-agent-framework/mcp_config.json`, under `mcp_servers`. **Absolute
paths, and the interpreter from your conda env** — not a bare `python3`:

```json
"reaction": {
  "command": "/path/to/your/.conda/envs/ideas/bin/python3",
  "args": ["/path/to/IDEAS/multi-agent-framework/mcp/reaction-sandbox-mcp/main.py"],
  "timeout": 300.0
}
```

The 300 s timeout is deliberate: the binning and simulation tools are much
slower than the other servers, which are HTTP data lookups.

---

## 7. Verify

Run from the **framework repo root** — the servers assume it as the working
directory.

```bash
cd $IDEAS/multi-agent-framework
source $IDEAS/env_compy.sh

python3 -c "
import sys; sys.path.insert(0,'src')
from core.mcp_manager import MCPManager
c = MCPManager('mcp_config.json').get_all_clients()['reaction']
t = c.list_tools()
print('tools:', len(t))
print('lambda:', [n for n in t if 'lambda' in n.lower()])
"
```

Expected:

```
tools: 39
lambda: ['run_lambda_preprocessing', 'run_lambda_binning',
         'generate_lambda_reaction_database', 'visualize_binning_results']
```

**Fewer than 39 tools, or a hang, means the shim is not being picked up** —
that is the silent-fallback failure from step 4a.

Then exercise a real tool:

```bash
python3 -c "
import sys, tempfile; sys.path.insert(0,'src')
from core.mcp_manager import MCPManager
c = MCPManager('mcp_config.json').get_all_clients()['reaction']
r = c.call_tool_json('run_lambda_binning',
    {'sample_id':'SPS_0001','binning_method':'uniform','n_bins':5,
     'output_dir':tempfile.mkdtemp()})
print(r.get('validation_status'), r.get('reaction_network_file'))
"
```

Expected: `success` and a path to a `reaction_network.txt`.

Finally, the whole backend:

```bash
python3 workflow.py --interactive --model lambda-pflotran
```

---

## 8. Two behaviours to know before relying on it

Both measured on this install, neither documented upstream.

**`binning_method` has no effect.** `uniform`, `cumulative`, `class_based` and
`bulk` return byte-identical networks at every bin count tested. The server is
routed to a reconstructed binning package that implements neither those four
methods nor their semantics. Selecting one is not a choice.

**Use `n_bins=5`.** Donor species are named for each bin's mean carbon number,
*rounded*, so bins whose means round to the same integer collapse into one
species. For sample `SPS_0001`: `n=3` gives `C21, C24, C21`; `n=4` gives two
collisions; `n=5` is distinct. The framework validates this and falls back to a
verified fixture, but requesting 5 avoids the fallback. This is a property of
one sample's carbon distribution, not a general rule — check any new sample.

---

## Troubleshooting

| symptom | cause |
|---|---|
| server "starts" but no tools appear | the `fastmcp` shim is not first on `sys.path`; upstream's placeholder `SimpleMCP` won |
| `server dir not found` | `REACTION_MCP_DIR` unset or wrong |
| PETSc `configure` fails downloading HDF5 | Compy's TLS is too old — supply the tarball locally (step 2) |
| pip installs fail to build | `export PIP_ONLY_BINARY=":all:"` |
| tools time out | raise `timeout` in `mcp_config.json`; simulations are not HTTP-fast |
| MPI errors when a tool runs a simulation | `MPI_COMMAND` should be `mpirun` on Compy, `srun` on Slurm-native sites |
