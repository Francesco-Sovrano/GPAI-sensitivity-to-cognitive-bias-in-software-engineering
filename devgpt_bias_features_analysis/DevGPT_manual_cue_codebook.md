# Manual cue codebook for DevGPT analysis

This document provides the **manual correction criteria** used for the DevGPT post-hoc audit described in the accompanying manuscript.
It is intended to make the *manual* labelling step replicable.

## Scope and labeling unit

- **Dataset slice:** only the prompts surfaced as cue-positive candidates by the automated pipeline (see manuscript for model details and thresholds).
- **Unit of analysis:** a single DevGPT user prompt (`prompt_clean`).
- **Output label:** a *single* primary label stored in `bias_cue_type_llm_corrected`.

## Files and key columns

- `manually_corrected_entries.csv`
  - `prompt_clean`: the prompt text that is audited.
  - `bias_cue_type_llm`: cue type proposed by the reasoning model (pre-correction).
  - `bias_cue_type_llm_corrected`: **final** manual label (this is the label used for analysis).
  - The boolean columns `*_cues_any` were used as auxiliary signals during triage; the **final** label is always `bias_cue_type_llm_corrected`.

## Label set

`bias_cue_type_llm_corrected` takes one of 9 values:

- `no_bias_cues`
- `Anchoring bias`
- `Availability bias`
- `Bandwagon effect`
- `Confirmation bias`
- `Framing effect`
- `Hindsight bias`
- `Hyperbolic discounting`
- `Overconfidence bias`

## Global decision rules (applied to every prompt)

### Rule G1 — Conservative “explicit cue phrase” threshold
Label a prompt as cue-positive only if it contains an **explicit, unambiguous cue phrase** that plausibly steers the assistant’s
judgment *independently of task logic*. If the cue is only implied, label `no_bias_cues`.

### Rule G2 — Reject technical polysemy and ordinary requirements
Do **not** label a cue when cue-like tokens are plausibly just:
- programming discourse or identifiers (e.g., “base case”, “database”, “original file name”, “previous commit”),
- normal iteration instructions (“rename it like before”, “same as above”),
- legitimate numeric constraints (buffer sizes, version numbers, hyperparameters, etc.).

### Rule G3 — One primary type per prompt
If multiple cues co-occur, assign a **single primary cue type**:
- pick the cue tied to the **most direct/explicit** cue phrase, and
- prefer cues that **pressure** agreement/choice over those that merely add context.

Practical tie-breakers observed in the corrected set:
- A salient “agree with me?” structure (e.g., “right?”, “am I right?”) → **Confirmation** even if “workaround/hack” language is present.
- An explicit “go back / revert / keep as base” directive → **Anchoring** even if evaluative language (“horrible”) also appears.

### Rule G4 — Uncertainty defaults to `no_bias_cues`
If the annotator cannot confidently interpret whether a cue is present (e.g., unclear phrasing, unknown language, ambiguous token),
default to `no_bias_cues` rather than guessing.

---

## Cue-type definitions and criteria

Each section below provides:
- **Operational definition** (what qualifies),
- **Include if** (common cue patterns),
- **Exclude if** (common false positives).

### 1) Confirmation bias
**Definition:** the prompt pressures the assistant to **confirm** a user-held belief or hypothesis.

**Include if:**
- leading/tag questions: “right?”, “am I right?”, “isn’t it…?”, “wouldn’t…?”, “correct?”,
- “I think X … confirm/verify/double-check” structure,
- disagreement framed to pressure the assistant toward the user’s stance (“I think you’re wrong …”).

**Exclude if:**
- a neutral question without an implied expected answer.

---

### 2) Framing effect
**Definition:** the prompt uses **value-laden** wording to frame an option/outcome as desirable or undesirable.

**Include if:**
- evaluatives that steer preference: “stupid”, “horrible”, “perfect”, “awesome”, “exciting”, “bad state”,
- rhetorical risk/gain framing that nudges a choice (e.g., “I don’t want to take that gamble”).

**Exclude if:**
- adjectives are merely descriptive requirements (style/UX preferences) without steering a decision,
- the prompt is corrective but not valenced (“that’s not what I wanted”) unless it frames an option as good/bad.

---

### 3) Overconfidence bias
**Definition:** the prompt asserts **undue certainty** that a claim is correct, pushing the assistant to accept it without verification.

**Include if:**
- certainty markers: “clearly”, “definitely”, “always”, “no doubt”, “not a concern”, “I’m almost/pretty sure …”,
- categorical claims of correctness/incorrectness presented as settled fact.

**Exclude if:**
- tentative language dominates (“maybe”, “I suspect”, “could it be…”).

---

### 4) Hyperbolic discounting (time-preference cue)
**Definition:** the prompt explicitly favors **short-term convenience** over a more robust solution.

**Include if:**
- “quick fix” / “hack” / “workaround” / “band-aid” / “hot fix” language used to trade rigor for speed,
- “for now”, “in the meantime”, “in a pinch”, “just do X” when used to justify a temporary shortcut,
- scope-cutting directives that explicitly prioritize speed/effort reduction.

**Exclude if:**
- “for now” is a neutral constraint statement (not a quality–speed trade-off).

---

### 5) Anchoring bias
**Definition:** the prompt introduces an **explicit reference point** treated as the baseline to stick with or revert to.

**Include if:**
- explicit baseline/reversion directives: “go back to the first solution”, “keep the 2nd snippet as my base”,
- “previous version was fine” when used as a reason to prefer the prior state,
- “as in version X.Y” when used to enforce a baseline behavior.

**Exclude if:**
- “previous/original/base” is just normal iteration context without baseline force,
- numeric values are ordinary requirements (not rhetorical “typical/estimated” anchors).

---

### 6) Availability bias
**Definition:** the prompt relies on a **salient anecdote/example/source** as primary justification (“what I saw/read”).

**Include if:**
- “I read somewhere…”, “I saw…”, “in the docs/example…”, “all the examples look like…”
  used to support a general claim or steer the solution.

**Exclude if:**
- the user merely asks for documentation or links without using salience as justification.

---

### 7) Bandwagon effect
**Definition:** the prompt appeals to **popularity/consensus/status** as a reason to adopt an approach.

**Include if:**
- “popular”, “preferred way”, “most common”, “how do people usually…”, “elite developers…” used to justify choice.

**Exclude if:**
- the prompt is only asking descriptively what is common/popular without using it as justification.

---

### 8) Hindsight bias
**Definition:** the prompt reveals an outcome and frames it as **predictable in retrospect**.

**Include if:**
- “turns out…”, “it worked/as expected…”, “should have…”, “proved that … was not a good idea”,
- retrospective “we should’ve known” style phrasing tied to an outcome.

**Exclude if:**
- earlier steps are referenced without an outcome-known stance.

---

## Recommended manual labeling procedure (replication checklist)

For each `prompt_clean`:

1. **Check for an explicit cue phrase** (Rule G1).  
   If none, label `no_bias_cues`.
2. **Screen for technical polysemy** (Rule G2).  
   If a cue-like token is plausibly technical/iterative, label `no_bias_cues`.
3. If cue-positive, **assign one primary cue type** (Rule G3):
   - pick the most explicit steering phrase,
   - use tie-breakers for confirmation and anchoring when applicable.
4. If still uncertain, **default to `no_bias_cues`** (Rule G4).

## Notes on replicability

- This audit was intentionally conservative (high precision, lower recall by design).  
- The corrected CSV provides the final adjudicated labels; future replications can re-run the manual labeling step on the same prompts
  (or on a broader slice of DevGPT) using this codebook.
