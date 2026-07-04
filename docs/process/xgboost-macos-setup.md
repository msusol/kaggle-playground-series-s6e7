# XGBoost setup on macOS (MacPorts, shared venv)

## Prerequisites

- The shared Kaggle venv at `/Users/marksusol/LosusAI/Projects/Kaggle/.venv`
  (never create per-project venvs — see project conventions).
- MacPorts (`port`), not Homebrew — this project never uses `brew`.
- `sudo` access for the one-time `libomp` install.

## The problem

XGBoost's official macOS wheel (`pip install xgboost`) dynamically links against
`libomp.dylib` (the LLVM OpenMP runtime) via a hardcoded `LC_RPATH` entry pointing
at Homebrew's install location (`/opt/homebrew/opt/libomp/lib`). On a machine
without Homebrew, `import xgboost` fails:

```
XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.
...
Reason: tried: '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file)...
```

MacPorts installs the equivalent library at a different path
(`/opt/local/lib/libomp/libomp.dylib`), which XGBoost's wheel doesn't know to look
for.

## Steps

1. Install the OpenMP runtime via MacPorts:
   ```
   sudo port install libomp
   ```

2. Patch the installed XGBoost binary's rpath to point at the MacPorts location
   instead of the Homebrew path it expects:
   ```
   install_name_tool -rpath /opt/homebrew/opt/libomp/lib /opt/local/lib/libomp \
     /Users/marksusol/LosusAI/Projects/Kaggle/.venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
   ```

3. Re-sign the binary (`install_name_tool` invalidates the code signature on
   Apple Silicon; without this step the OS refuses to load the modified binary):
   ```
   codesign --force --sign - \
     /Users/marksusol/LosusAI/Projects/Kaggle/.venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
   ```

4. Verify:
   ```
   /Users/marksusol/LosusAI/Projects/Kaggle/.venv/bin/python -c "import xgboost; print(xgboost.__version__)"
   ```

## Expected output

`import xgboost` succeeds with no environment variables needed — the fix is
permanent (baked into the binary's load path) and works in any process spawned
from this venv, including JupyterLab kernels, with no special launch environment
required.

## Troubleshooting

- **This patch does not survive `pip install --upgrade xgboost` or
  `pip install --force-reinstall xgboost`** — reinstalling fetches the original
  wheel again with the Homebrew rpath restored. Re-apply steps 2-3 after any
  xgboost reinstall/upgrade.
- **`DYLD_LIBRARY_PATH=/opt/local/lib/libomp` as a workaround does *not* work for
  JupyterLab.** It only helps a single foreground process it's set on directly.
  Setting it via `os.environ[...]` inside a notebook cell before `import xgboost`
  does not work either — dyld resolves `DYLD_*` variables at process launch, not
  per-`dlopen`-call. Background/detached processes launched via Claude Code's Bash
  tool specifically strip `DYLD_*` variables (a library-injection hardening
  measure), so even relaunching JupyterLab as a background process with the env
  var set does not propagate it to spawned kernels. The `install_name_tool` patch
  above is the only fix that's actually permanent and background-process-safe.
- **Do not** create a `/opt/homebrew/opt/libomp/lib/libomp.dylib` symlink as an
  alternative fix — it works, but creates a Homebrew-shaped path on a machine that
  deliberately doesn't use Homebrew, which is confusing later. Prefer the
  `install_name_tool` patch, which uses the real MacPorts path directly.
