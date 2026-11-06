# Security and privacy

This alpha is a **single-user, local research application**. The launcher
refuses non-loopback host bindings. Host/Origin checks and input limits reduce
some accidental cross-origin/local attacks; they are not user authentication,
multi-user isolation, or a production web-service security boundary.

Do not expose it with a public reverse proxy, tunnel, `0.0.0.0`, or a shared
untrusted machine. Anyone with access to the running process or its user data
may be able to inspect saved text and experiments. Exports, SQLite archives,
debug logs and model provenance paths may contain sensitive research inputs.
Review them before sharing.

Models load from local safetensors. Remote model code and pickle checkpoints
are not enabled. Nonetheless, inference libraries and native parsers are
complex software: only use trusted checkpoints, keep dependencies updated, and
run as an ordinary user. The registry is local configuration, not an untrusted
upload format. Source/revision metadata alone does not authenticate local files.

Model management is a **privileged local interface**: it can inspect supplied
local model paths, edit its selected registry and explicitly download/convert
prepared checkpoints. Anyone who can use this local session can perform those
actions. Host/Origin checks still apply; there is no additional administrator
password. Registry edits have backups and revision-conflict protection, but
backup files and installation logs can themselves contain local paths.
Disconnecting/hiding a model never deletes its weights. Cancellation only
removes a fresh staging directory owned by that installation; completed outputs
remain. The GUI never installs Python packages or accepts arbitrary remote-code
instructions. Only curated, pinned public sources can be downloaded through it;
manual registry edits and separately acquired checkpoints are the user's trust
decision. The Electron picker exposes directory selection only on trusted
model-management routes, with context isolation and Node integration disabled.

Experimental-addon connections are also privileged local settings. They may
read artifact manifests and frozen provenance references outside the selected
directory. Use only packages you trust. The GUI cannot accept a new Python
implementation or executable filename from a package: the legacy detector's
external feature extractors must match a small reviewed SHA-256 allowlist in
the application itself. A self-declared package checksum cannot add code to
that allowlist. Calibration files remain outside Git; settings/receipts/backups
can contain private absolute paths and must not be shared without review.
Receipt invalidation uses observed size/mtime and model metadata between full
verification runs, not a new cryptographic proof on every request. Native
readers retain their full-content/transfer validation before use.

Normal inference does not call a cloud model API. Explicit setup downloads
contact Hugging Face; pip/npm installation contacts their package registries.
Dependency behaviour and infrastructure are outside this project's complete
control. Use a restricted/offline environment for sensitive research after
dependencies and models have been acquired and checked.

For a security report, contact the maintainer privately at
`josef.slerka@ff.cuni.cz`. Avoid public reports containing private prompts,
credentials, archive files, or exploit payloads with real user data. There is
no guaranteed response time or hosted-service security SLA.
