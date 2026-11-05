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

Normal inference does not call a cloud model API. Explicit setup downloads
contact Hugging Face; pip/npm installation contacts their package registries.
Dependency behaviour and infrastructure are outside this project's complete
control. Use a restricted/offline environment for sensitive research after
dependencies and models have been acquired and checked.

For a security report, contact the maintainer privately at
`josef.slerka@ff.cuni.cz`. Avoid public reports containing private prompts,
credentials, archive files, or exploit payloads with real user data. There is
no guaranteed response time or hosted-service security SLA.
