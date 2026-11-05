# What a measurement says — and what it does not

The workbench measures one specified language model under one specified input.
These are inspectable predictions to compare with linguistic hypotheses, not
direct measurements of human comprehension.

- **Probabilities:** next-token softmax over the model's complete vocabulary.
  The visible top-k list is only a display cut; the remaining mass is retained.
- **Surprisal:** `−log₂ P(observed token | earlier tokens)`, in bits. The text
  supplies the observed continuation; the model is not generating that text.
  Scoring uses temperature 1. A generation temperature is a different control.
- **Word view:** sums token surprisals only when the orthographic span aligns
  cleanly with token boundaries. This is the probability of the exact token
  sequence, not a marginal over every possible encoding of a linguistic word.
  Punctuation is a separate observed unit; contaminated boundaries stay
  unscored. The token trace remains available.
- **Pre-word entropy:** uncertainty over the next-token distribution just before
  a word begins, not entropy over all possible whole words. `Δ surprisal` in
  word view is the difference from the preceding word, not an experimental
  condition contrast.
- **Context contrast:** the same exact continuation is scored after two prefixes
  by teacher forcing. A positive information advantage A−B means that exact
  path was more probable under A. This is not a probability that A is true.
  Boundary tokenization and leading spaces matter and are checked explicitly.
- **JS divergence:** compares the two full next-token distributions, not only
  the supplied continuation. It describes a distributional change. It is not a
  direct estimate of processing difficulty or a causal intervention.
- **Embeddings:** layer 0 is a token's fixed input row. A multi-token word does
  not have a single row; contextual states are read at the selected final
  fragment's position. Two contexts can share the input row and diverge later.
  PCA is only a projection; original-space distances remain the metric.
- **RAW/readout:** applying a final readout to intermediate states shows what
  is readable by that readout, not where a concept is uniquely stored.
- **Attribution:** an accounting decomposition of one realized forward into a
  logit contrast. Reconstruction does not establish necessity or causation.
- **Patching/steering:** an actual intervention, with scope and identity controls.
  A result depends on the selected layer, positions, direction and baseline.
  Do not interpret it as unrestricted control of an abstract meaning.

## Comparing models and languages

Token IDs, word fragmentation, vocabulary sizes and training distributions
differ across models. A token's surprisal is not directly comparable to a
different model's differently segmented token. Fix orthographic spans, context
and scoring conventions, keep token traces, and report token/word coverage.
Do not rank languages by raw bits/token. Quantized and unquantized execution
are distinct numerical conditions; small observed differences can be small,
but no equality is assumed without measuring it on the relevant items.

Use original/base checkpoints for next-token text studies unless your research
question specifically concerns instruction/chat models. The workbench scores
raw supplied text, not a hidden chat template. Record BOS, spacing, Unicode
normalization (or its absence), model revision, backend/device, dtype,
quantization, source revision and stimulus text.

## Optional research artifacts

Trained TUNED/J-Lens readers and predictive-stylometry detector calibrations
are not in this repository. Their presence must be validated for a particular
checkpoint and domain. A detector score cannot establish authorship or justify
an academic misconduct accusation. The public default disables these tools
for real-model registrations; synthetic interface fixtures do not substitute
for independent validation.

## Reproducibility

Exports retain model and computation metadata. Registry source metadata is a
declaration; it is labelled as such. Download/conversion receipts contain file
hashes, but the runtime does not rehash many gigabytes on every request. Keep
your receipts, registry snapshot, checkpoint and exported results together.
For publishable experiments, add your own stimulus set, preregistered contrasts
where appropriate, numerical checks and human-behaviour validation.
