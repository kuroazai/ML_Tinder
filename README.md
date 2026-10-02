# swipeml

Train a binary image classifier on labels you made yourself, then apply it
through the browser.

The model learns *your* labels. It is not a judgement about anyone; it is a few
hundred of your own decisions, fitted. Treated as anything more than that it
will be wrong, and the evaluation here is built to tell you how wrong.

```bash
pip install -e ".[all,dev]"
swipeml backbones                   # what you can train on
swipeml data                        # inspect the dataset, before training on it
swipeml train --backbone xception   # train, pick a threshold, write a manifest
swipeml predict photo.jpg           # classify
swipeml swipe --dry-run             # decide without clicking
```

> Driving a site through a browser is usually against its terms of service.
> This automates an account you are signed into, from a profile you already
> own, and whether that is appropriate is between you and whoever runs the
> site. Read their terms.

---

## What it does

| | |
|---|---|
| `swipeml.dataset` | What is in the labelled folders, and what is wrong with it |
| `swipeml.backbones` | Five pretrained backbones, each with the preprocessing it was trained with |
| `swipeml.training` | Augmentation, class weights, two-phase fine-tuning, honest checkpointing |
| `swipeml.evaluate` | Precision, recall, ROC-AUC, and choosing the threshold |
| `swipeml.manifest` | What a saved model remembers about how it was trained |
| `swipeml.predict` | Serving, using the manifest so it cannot diverge from training |
| `swipeml.automation` | Selenium session, resilient locators, the swipe loop |

Only numpy is required. TensorFlow and Selenium are optional extras, and the
dataset, metric, locator and configuration code all work without either. CI has
a job that installs neither and asserts nothing imports them at module scope.

## The bug that matters most

Every ImageNet backbone expects its pixels scaled a particular way:

| backbone | input | preprocessing |
|---|---|---|
| vgg16 | 224x224 | caffe: BGR, per-channel mean subtracted |
| resnet50 | 224x224 | caffe |
| inceptionv3 | 299x299 | tf: scaled to -1..1 |
| xception | 299x299 | tf |
| inceptionresnetv2 | 299x299 | tf |

Feed a tf-style network caffe-style inputs and it still trains, still reports a
plausible accuracy, and is quietly worse than it should be. **Nothing errors.**

The previous version of this project chose the right `preprocess_input` for each
architecture, returned it from `build_model`, and then the caller ignored it:

```python
model, preprocess_input = build_model(model_name, input_shape=(224, 224, 3))
# preprocess_input is never used again
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

`rescale=1./255` is correct for none of the five. Every backbone was also forced
to 224x224, throwing away a fifth of the resolution the inception family's
weights expect.

So preprocessing is part of a backbone's identity here, and a trained model is
saved with a **manifest** recording which backbone, which input size, the class
order and the chosen threshold. `swipeml predict` reads the manifest rather than
taking anyone's word for it, and refuses to serve a model that has none:

```
$ swipeml model
models/model.keras
backbone     xception
input size   299x299
preprocess   tf
classes      disliked, liked  (positive: liked)
threshold    0.620
```

That closes the gap the old code fell straight into. Training resized to 224 and
divided by 255; prediction resized to 320 and then called
`.reshape(1, 224, 224, 3)`, which raises

```
ValueError: cannot reshape array of size 307200 into shape (1,224,224,3)
```

on every single call. The predictor could never have run, which is what the old
README meant by "currently absolutely broken".

## Accuracy is not the number you want

The old README reported 85% accuracy and no baseline beside it. On a two-class
set that is 80/20, answering the same way every time scores 80%, so 85% could
mean the model learned almost nothing. Whether it was good depended entirely on
the split, and the split was never reported.

`swipeml data` reports the baseline before you train:

```
200 image(s) in training_data
  disliked            80
  liked              120
  imbalance        1.5:1
  majority baseline 60.0%  <- a model must beat this to have learned anything
```

And training reports against it:

```
                  said yes     said no
       was yes          76          24
        was no          92         308

accuracy   0.768
precision  0.452
recall     0.760
f1         0.567

roc auc    0.822
baseline   0.800  (always predict the majority)

This model does not beat always predicting the majority class. Its accuracy is
not evidence that it learned anything.
```

Which is the interesting case: ROC-AUC of 0.822 says the model ranks images
genuinely well, while its accuracy at the default threshold is *below* the
baseline. Both numbers are true and only one of them is the one people quote.

### The threshold is chosen, not assumed

0.5 is correct when the classes are balanced and both mistakes cost the same.
Neither is true here. So the threshold comes from the validation scores:

```bash
swipeml train --threshold-objective f1          # the default
swipeml train --minimum-precision 0.8           # maximise recall, keep precision >= 0.8
```

The second is the useful one for an automated loop: "only act when you are
fairly sure". The chosen threshold goes in the manifest, so prediction and
training agree on it.

### The rest of the training fixes

- **The best weights were being thrown away.** `EarlyStopping(patience=3)`
  without `restore_best_weights=True` leaves the model three epochs past its
  best. The run then did `model.save('tinder_model_main.h5')`, overwriting the
  `ModelCheckpoint` file that *did* hold the best weights, and the predictor
  loaded the overwritten one. The reported accuracy belonged to a model nobody
  ever used.
- **No augmentation**, on a few hundred photographs with a 14-million parameter
  feature extractor.
- **No class weights**, on a dataset that is whatever you happened to swipe.
- **The split moved between runs.** `ImageDataGenerator(validation_split=...)`
  divides each class by directory order, so adding one file reshuffles it and
  two runs' validation numbers are not comparable. The split is now made per
  class, seeded, and written to `split.json`.
- **Fine-tuning** is a second phase, after the head has settled, which is the
  only point at which unfreezing the backbone helps.

## The automation

The hard part is not driving the browser, it is finding anything on a page that
does not want to be automated. The old code did it the way that cannot work:

```python
divs = soup.find_all("div", class_="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox")
btn = '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div/div[4]/div/div[4]/button'
```

Those class names are Atomic CSS. They encode styling, not meaning, so they
change whenever the design does. The XPath is a literal path through the DOM, so
one extra wrapper element anywhere above the button breaks it. Both had already
broken.

Neither can be made reliable. What *can* be made reliable is failing usefully.
Each element has several strategies, ordered by how stable they are, and the one
that worked is reported:

```
$ swipeml selectors
like button
  [ 10] ok      css=button[aria-label*="Like" i]
  [ 15] ok      css=[data-testid="gamepadLike"]
  [ 30] ok      text=like
  [ 60] ok      css=main button:has(svg[viewBox])
card image
  [ 10] ok      css=[data-testid="card"] div[style*="background-image"]
  [ 20] ok      css=div[aria-label][style*="background-image"]
  [ 40] ok      css=div[style*="background-image"]
  [ 90] fragile css=div.Bdrs\(8px\).Bgz\(cv\)

1 fragile selector(s):
  card image: built on generated styling class(es) Bdrs(8px), Bgz(cv):
  these change whenever the design does
```

It knows which of its own selectors are rubbish, and says so. When one is used,
the run notes it, so drift shows up in the log before it becomes a failure. When
all of them fail:

```
could not find 'like button'. Tried 4 strategy(ies):
  css='button[aria-label*="Like" i]' (no match)
  css='[data-testid="gamepadLike"]' (no match)
  text='like' (no match)
  css='main button:has(svg[viewBox])' (no match)

The page has almost certainly changed. Update the strategies for 'like button'
in swipeml/automation/locators.py.
```

That is fixable in minutes. `NoSuchElementException` with an absolute XPath is
an afternoon.

### The loop

```bash
swipeml collect --limit 200          # save cards, you label them
swipeml swipe --dry-run              # classify and file, click nothing
swipeml swipe --live --limit 30      # act
```

`--live` is required to click anything. A flag you have to type is the right
amount of friction for a loop driving a real account. The old default was worse
than unsafe, it was broken: `--validation` defaulted to `1`, and that branch
`continue`d without decrementing the counter, so the default invocation
downloaded images forever and never swiped.

`collect` saves to one folder and you sort it. Nothing here can read which way
you swiped, and saying otherwise would be a lie: the old code polled
`keyboard.is_pressed` in a `while True` with no sleep, which burns a CPU core,
needs root on Linux, and still only knows a key went down somewhere.

### Downloads

```python
response = requests.get(url, stream=True)
with open(temp_name, 'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)
```

No timeout, so a slow server hangs the run. No status check, so a 404 or a
rate-limit page gets written to a `.jpg`, added to the training set, and
classified. Now: a timeout, the status checked, the content type checked, a size
cap, and the first bytes verified against the format's magic number before the
file is kept. `swipeml data` also finds any that got through previously:

```
'liked' has 1 file(s) that are not images despite the extension: rate_limited.jpg
```

## Selenium 4

The old browser code cannot run on any currently installed Selenium:

- `webdriver.Firefox(profile)` — a positional profile was removed in Selenium 4
- `find_element_by_xpath` — removed in Selenium 4
- `self.browser.driver.find_element(...)` — `self.browser` *is* the driver, so
  every method using `.driver` raised `AttributeError`, and `auto_tinder`
  referred to `src.driver`, which was never set at all
- `driver.close()` shuts one window and leaves the process running, which over a
  few interrupted runs leaves orphaned geckodriver processes holding the profile
  open. It is `quit()` now.

## Setting up

```bash
pip install -e ".[all,dev]"
cp .env.example .env
```

Point `SWIPEML_BROWSER_PROFILE` at a browser profile that is already signed in.
That is how this never touches your credentials: nothing here ever sees a
password. `SWIPEML_URL` is blank on purpose, so a fresh clone cannot point
anywhere by accident.

The old version shipped a tracked `config.py` with the profile path in it, which
conflicts on every pull. `swipeml config` reports whether a profile is set and
valid without printing the path, because a profile path identifies a person and
a machine, and there is a test asserting it stays out of the output.

## Layout

```
src/swipeml/
├── cli.py            the commands
├── config.py         settings from the environment
├── dataset.py        what is in the folders, and the majority baseline
├── backbones.py      the five backbones and their preprocessing
├── manifest.py       what a saved model remembers; kills train/serve skew
├── training.py       augmentation, class weights, two-phase fine-tuning
├── evaluate.py       metrics and threshold selection, on numpy alone
├── predict.py        serving, through the manifest
└── automation/
    ├── session.py    Selenium 4, profile via Options
    ├── locators.py   layered strategies, and knowing which will rot
    ├── images.py     downloads that cannot write an error page as a jpg
    └── swiper.py     the loop, with a budget and a dry run
```

## Development

```bash
pip install -e ".[all,dev]"
pytest              # 138 tests
ruff check src tests
mypy
```

No test needs TensorFlow, Selenium, a browser, a network, or a photograph. The
image fixtures are the shortest byte strings that count as their format, which
is enough for everything except running a model, and a model is not what most of
these tests are about. Selenium is stubbed at one function; `requests` at one
method.

The metrics are implemented on numpy rather than pulling in scikit-learn for
twenty lines of arithmetic, which also means the definitions are visible. The
ROC-AUC uses the rank-sum form with averaged tied ranks, because a
barely-trained network outputs the same probability for everything and
integrating a curve built from unique score values gets that case wrong.

The old `requirements.txt` listed twelve packages unconditionally, including
tensorflow, keras, keras-applications, scikit-learn, cssutils and keyboard.
Installing it to look at the dataset code pulled in all of them.

## Licence

MIT. See [LICENSE](LICENSE).
