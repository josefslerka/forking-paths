# Forking Paths — optional Electron window

The browser is sufficient for all research tools. The Electron shell provides
a separate window, copy/paste menus and local server lifecycle management; it
does not bundle Python or model weights.

The native application is named **Forking Paths**; its Czech interface is
**Rozcestí**. The original bundle identifier and Electron profile directory are
intentionally preserved, so settings, language preferences and source-directory
connections survive the name change. Python data/cache directories and
`APERTUS_*` configuration variables are unchanged as well.

## Development launch

First install Python into `.venv` as in the root README (and the numerical
extra for your platform). Model weights are not required to open the shell.
Then, with Node.js 22.12 or newer:

```sh
cd apertus/garden/desktop
npm ci
npm test
npm start
```

The shell finds the repository above its source directory, prefers the
repository's `.venv`, validates the registry through the same Python CLI,
and starts a loopback server using the same registry and user-data location.
Override with `APERTUS_ROOT`, `APERTUS_PYTHON`, `APERTUS_MODEL_REGISTRY` or
`APERTUS_DATA_DIR`. No Anaconda path or author-specific model layout is needed.
It may attach to an already running compatible local server on port 8765;
check the displayed model rather than assuming a new server was started.
Without ready weights the server opens **Model management** instead of refusing
to start. There you can connect a local directory with **Choose folder**, review
a prepared installation, test loading, select a default or disconnect an entry.
Disconnecting never deletes weights. The native picker is limited to the
current workbench origin and model-management routes, not a general filesystem
API. All model/registry actions use the same guarded local server as the browser.

Changes made from bundled presets create a user-owned registry under the data
directory. On subsequent launches the shell prefers an explicit registry, then
the checkout's `models.toml`, then the user registry, then bundled presets.

To connect to a server you explicitly manage, set `APERTUS_WORKBENCH_URL`
(see the shell's source for the accepted URL contract). Only use trusted,
appropriately protected servers. A desktop window does not add authentication
to the Python application.

## Local macOS build

```sh
CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack
```

This produces `dist/mac-arm64/Forking Paths.app`, a local Apple-silicon app directory. Open its settings/chooser
to select the source repository if it cannot find it. Models and `.venv` stay
external. Public signed/notarized installers and automatic updates are not
provided by this alpha repository. Before distributing a desktop binary,
review Electron/Chromium security updates, licenses, signing and platform QA.

The lockfile pins an updated Electron and contains narrow `minimatch` overrides
for build-tool dependencies. Recheck them when upgrading electron-builder;
`npm audit` is a dependency advisory check, not a full application security audit.
Packaging commands explicitly disable publishing.
