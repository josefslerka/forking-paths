# Release checklist

Source repository: [josefslerka/forking-paths](https://github.com/josefslerka/forking-paths).
The source is published under Apache-2.0. Public source availability does not
mean that the local application server is safe to expose, or that signed
installers, model weights or experimental packages are distributed here.

Before each source or binary release:

- Run the Python, synthetic browser, desktop-helper and wheel-install gates
  from CONTRIBUTING. Record versions and skipped asset-dependent checks.
- Check first launch without weights, model-management consent/cancellation,
  registry backups and the native desktop directory chooser. Never run release
  tests against a normal user's registry or delete their weight directories.
- Numerically smoke-test each advertised adapter on real local weights and
  keep the test registry private. Include the checkpoint/precision in results.
- Review `git ls-files`, sizes, source strings and the complete Git history for
  personal paths, secrets, private stimuli, results, corpora and model weights.
  `.gitignore` cannot remove already tracked history.
  `python tools/audit_release.py` supplies a read-only tracked-file hygiene gate;
  it is a heuristic aid, not proof that no sensitive content exists.
- Inspect the built wheel/sdist: static assets, presets, license and notices
  must be present; private artifacts and developer environments must not be.
- Recheck model-source licenses/revisions and dependency notices. Source
  notices do not cover a redistributed desktop binary's complete dependency set.
- Update dependency/security advisories, especially the optional Electron shell.
  No public installer is signed/notarized or auto-updated by this repository.
- Fill in repository/homepage URLs and preferred version/DOI in citation
  metadata once a public destination actually exists. Do not publish fabricated
  badges, download links, benchmarks or numerical guarantees.
- Obtain the owner's explicit approval before creating a GitHub repository,
  pushing commits, uploading releases, or publishing packages.
