# AGENTS.md

Notes for anyone, human or otherwise, changing this code.

## What this is

A binary image classifier trained on labels the user made, plus the browser
automation to gather the labels and apply the predictions.

Two halves, deliberately separable. The ML half needs no browser; the automation
half needs no model to be useful (`collect` just saves images). Only numpy is a
hard dependency. TensorFlow and Selenium are optional extras.

## Start here

```bash
pip install -e ".[all,dev]"
pytest
swipeml backbones
swipeml selectors
```

`swipeml backbones` and `swipeml selectors` need nothing installed beyond the
package and are the quickest way to see what the thing is.

## The rules that matter

**Preprocessing belongs to the backbone, and the manifest carries it.** This is
the whole point of `backbones.py` and `manifest.py`. Every ImageNet backbone
wants its pixels scaled a particular way, getting it wrong produces no error at
all, and the previous version got it wrong for all five. Never hardcode
`rescale=1./255` or an input size; take both from the backbone, and record them
in the manifest so serving cannot diverge from training.

**A model without a manifest is not served.** `predict.Classifier` raises rather
than guessing. A guessed preprocessing is invisible when it is wrong: the model
returns confident nonsense and looks merely mediocre.

**Never report accuracy without the baseline.** `dataset.majority_baseline` is
what a model has to beat to have learned anything, and on an unbalanced set that
can be 80%. `evaluate.Report.beats_baseline` exists so the report can say so out
loud. The previous README reported 85% with no baseline and no split.

**The threshold is chosen from validation scores, not left at 0.5.** 0.5 is
right only when the classes are balanced and both mistakes cost the same.
`evaluate.choose_threshold` takes an objective and an optional minimum
precision.

**Validate before importing an optional dependency.** This has now bitten three
times in this codebase and once in a sibling: `train()` checked the dataset
*after* importing TensorFlow, so a missing data folder reported "needs
TensorFlow"; `_find_one` imported Selenium before validating the strategy type,
so a typo reported a missing dependency. Argument and state checks go first,
heavy imports second.

**Keep the extras optional.** TensorFlow, Selenium and requests are imported
inside the functions that use them, never at module scope in `src/`, and never
before the code has established that it needs them. CI has a `core` job that
installs none of them and asserts they are absent from `sys.modules` after
importing every module.

Check it locally before pushing:

```bash
python scripts/check_optional_extras.py
```

That runs the suite with all of them blocked by a meta-path finder, changing
nothing about your environment. It has already caught this: `fetch_image` did
`import requests` unconditionally even when a session was supplied, which made
the `HttpGetter` protocol pointless and passed locally only because requests
was installed.

**No globals.** Everything goes through `TrainingConfig`, `Settings` or an
argument. The previous version read an `ARGS` module global from inside a method,
so the class could only be used by running the script as `__main__`.

**Return and report, do not raise, for things that are normal.** One unreadable
download in fifty, one driver missing from a save, one card whose image cannot
be found. These are expected; the loop records them and continues. Reserve
exceptions for genuine faults.

**`--live` is required to click.** `swipe` defaults to a dry run. Do not change
that default.

**Output is ASCII.** A pound sign or an em-dash crashes a legacy Windows console
code page.

**3.10 is the floor.** No `datetime.UTC` (3.11), `StrEnum` (3.11) or `tomllib`.
Use `timezone.utc`. mypy targets 3.12 here, because numpy 2.x stubs use 3.12
type syntax that mypy will not parse under an older target, so the 3.10 CI job
is the only guard against a version-specific API.

## Secrets and other people's data

**A browser profile path is personal.** It contains a username and a machine
identifier. `Settings.describe()` reports whether one is set and valid and never
prints it, and `test_describe_does_not_print_the_browser_profile_path` enforces
that. The previous version put it in a tracked `config.py`; the same pattern in
a sibling repository published a Windows username and a Firefox profile ID.

**Training data never goes in the repository.** `.gitignore` covers
`training_data/`, `unsorted/`, `models/`, `*.keras`, `*.h5` and `split.json`.
The images are photographs of real people; they are not ours to host. A trained
model encodes the dataset it came from, so that stays local too.

**Nothing here handles credentials.** The automation uses a browser profile that
is already signed in. Do not add a login flow.

## Testing

```bash
pytest
ruff check src tests
mypy
```

Everything runs offline. No TensorFlow, no Selenium, no browser, no network, no
photographs.

**The image fixtures are byte strings.** `conftest.JPEG_BYTES` is a JPEG magic
number and padding, which is enough for `dataset.looks_like_an_image` and for
file handling. `conftest.HTML_ERROR_PAGE` is what a rate-limited image URL
actually returns, and several tests exist only to prove it never reaches the
training set.

**Selenium is stubbed at exactly one function.** `locators._find_one` is the only
place that touches it, which is why the strategy ordering, the fallback
behaviour and the error messages can all be tested with a dict.

**requests is stubbed at one method.** `images.fetch_image` takes a `session`
satisfying the `HttpGetter` protocol.

**If you add a backbone**, add it to `BACKBONES` with its real native input size
and its real preprocessing style, and check `test_preprocessing_style_differs_between_families`
still makes sense. Getting either wrong is the bug this package exists to avoid.

**If you add a locator strategy**, give the locator at least two. There is a test
asserting it, because one selector is how the previous version ended up
permanently broken.

## Things that look like bugs and are not

- `ConfusionMatrix.precision` is 0.0, not 1.0, when nothing was predicted
  positive. A model that never says yes has no precision, and defining it as 1.0
  would flatter exactly the degenerate case worth catching.
- `roc_auc` returns NaN when only one class is present. It is undefined there,
  and NaN will not be mistaken for a score.
- `roc_auc` uses the rank-sum identity rather than integrating the curve. Tied
  scores get their average rank, which is the case that matters: a
  barely-trained network outputs the same probability for everything.
- `dataset.class_weights` is not rounded. The weights multiply a loss, and
  rounding to a few decimals stops the classes contributing exactly equally,
  which is the one thing they exist to do.
- `IMBALANCE_WARNING` is compared with `>=`. 1.5:1 is already the point where
  accuracy stops meaning much.
- `collect` takes one staging folder, not a liked and a disliked one. Nothing
  here can tell which way you swiped, and pretending otherwise would make the
  labels, which are the entire point, untrustworthy.
- `choose_threshold`'s candidates are computed in plain Python rather than with
  `np.linspace`. Same arithmetic, no array, and the type is obvious.
