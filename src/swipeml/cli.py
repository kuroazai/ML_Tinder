"""The command line.

    swipeml config                     what is configured
    swipeml backbones                  the available backbones and their sizes
    swipeml data                       inspect the dataset, before training on it
    swipeml train                      train, pick a threshold, save a manifest
    swipeml model                       what a saved model was trained with
    swipeml predict <images>           classify files
    swipeml selectors                  which page selectors are expected to rot
    swipeml collect                    gather cards to label
    swipeml swipe --dry-run            decide without clicking
    swipeml swipe --live               act on the predictions

`swipe` requires `--live` to click anything. A flag you have to type is the
right amount of friction for a loop that drives a real account; the original
defaulted to a mode that downloaded images forever and never swiped.

Output is ASCII, because a pound sign or an em-dash crashes a legacy Windows
console code page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import backbones
from .config import ConfigError, Settings


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(args.env_file)
    if getattr(args, "data_root", None):
        settings.data_root = Path(args.data_root)
    if getattr(args, "model", None):
        settings.model_path = Path(args.model)
    return settings


# -- commands --------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> int:
    print(_settings(args).describe())
    return 0


def cmd_backbones(args: argparse.Namespace) -> int:
    print(backbones.describe())
    print()
    print(backbones.SIZE_NOTE)
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    from . import dataset

    settings = _settings(args)
    try:
        report = dataset.inspect(settings.data_root)
    except dataset.DatasetError as exc:
        print(exc)
        return 1

    print(report.describe())
    if report.total:
        print()
        print("class weights (by keras class index):", report.class_weights())
    # Non-zero when something would make a training run misleading, so this is
    # usable as a gate in a script.
    return 2 if report.problems() else 0


def cmd_train(args: argparse.Namespace) -> int:
    from .dataset import DatasetError
    from .training import TrainingConfig, train

    settings = _settings(args)
    try:
        config = TrainingConfig(
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            fine_tune_epochs=args.fine_tune_epochs,
            augment=not args.no_augment,
            use_class_weights=not args.no_class_weights,
            threshold_objective=args.threshold_objective,
            minimum_precision=args.minimum_precision,
        )
    except (ValueError, KeyError) as exc:
        print(exc)
        return 1

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        train(settings.data_root, settings.model_path, config)
    except DatasetError as exc:
        print(exc)
        return 1
    except ImportError as exc:
        print(f"training needs TensorFlow: {exc}")
        print("try: pip install 'swipeml[train]'")
        return 1

    print(f"\nsaved {settings.model_path} and its manifest")
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    from .manifest import Manifest, ManifestError

    settings = _settings(args)
    if not settings.model_path.exists():
        print(f"no model at {settings.model_path}")
        return 1
    try:
        manifest = Manifest.load(settings.model_path)
    except ManifestError as exc:
        print(exc)
        return 1

    print(f"{settings.model_path}")
    print(manifest.describe())
    if manifest.notes:
        print("\nfrom the training run:")
        for key, value in manifest.notes.items():
            print(f"  {key:<22}{value}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import Classifier, PredictError

    settings = _settings(args)
    try:
        classifier = Classifier(settings.model_path)
    except Exception as exc:  # noqa: BLE001 - PredictError or ManifestError
        print(exc)
        return 1

    print(classifier.describe())
    print()
    try:
        predictions = classifier.predict_many(args.images)
    except ImportError as exc:
        print(f"prediction needs TensorFlow: {exc}")
        return 1
    except PredictError as exc:
        print(exc)
        return 1

    for prediction in predictions:
        print(f"  {prediction}")
    return 0


def cmd_selectors(args: argparse.Namespace) -> int:
    """Which page selectors are expected to rot, and why."""
    from .automation import locators

    for locator in locators.ALL_LOCATORS:
        print(f"{locator.name}")
        for strategy in locator.ordered():
            mark = "fragile" if strategy.fragile else "ok"
            print(f"  [{strategy.stability:>3}] {mark:<8}{strategy.how}={strategy.what}")
    print()
    print(locators.fragility_report())
    return 0


def _session(settings: Settings):
    from .automation import BrowserSession

    return BrowserSession(
        url=settings.require_url(),
        profile_dir=settings.browser_profile,
        browser=settings.browser,
        headless=settings.headless,
    )


def cmd_collect(args: argparse.Namespace) -> int:
    from .automation import SessionError, SwipeLimits, collect

    settings = _settings(args)
    try:
        limits = SwipeLimits(max_actions=args.limit, max_minutes=args.max_minutes)
        session = _session(settings)
    except (ConfigError, ValueError) as exc:
        print(exc)
        return 1

    try:
        with session:
            result = collect(
                session,
                staging_dir=settings.staging_dir,
                limits=limits,
            )
    except SessionError as exc:
        print(exc)
        return 1
    except ImportError as exc:
        print(f"the automation commands need Selenium: {exc}")
        print("try: pip install 'swipeml[automation]'")
        return 1

    print(result.summary())
    print(f"\nnow sort {settings.staging_dir} into {settings.liked_dir.name}/ "
          f"and {settings.disliked_dir.name}/, then run: swipeml data")
    return 0


def cmd_swipe(args: argparse.Namespace) -> int:
    from .automation import SessionError, SwipeLimits, run
    from .predict import Classifier

    settings = _settings(args)
    try:
        limits = SwipeLimits(
            max_actions=args.limit,
            delay_range=(args.min_delay, args.max_delay),
            max_minutes=args.max_minutes,
        )
        session = _session(settings)
        classifier = Classifier(settings.model_path)
    except (ConfigError, ValueError) as exc:
        print(exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - PredictError or ManifestError
        print(exc)
        return 1

    if not args.live:
        print("dry run: cards will be classified and filed, nothing will be "
              "clicked. Pass --live to act.")

    try:
        with session:
            result = run(
                session,
                classifier,
                liked_dir=settings.liked_dir,
                disliked_dir=settings.disliked_dir,
                limits=limits,
                dry_run=not args.live,
            )
    except SessionError as exc:
        print(exc)
        return 1
    except ImportError as exc:
        print(f"the automation commands need Selenium: {exc}")
        print("try: pip install 'swipeml[automation]'")
        return 1

    for outcome in result.outcomes:
        print(f"  {outcome}")
    print()
    print(result.summary())
    return 0


# -- wiring ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swipeml",
        description="Learn what you like from your own labels, then apply it.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--data-root", help="the labelled image folders")
    parser.add_argument("--model", help="the model file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config", help="show what is configured").set_defaults(
        handler=cmd_config)
    subparsers.add_parser("backbones", help="available backbones").set_defaults(
        handler=cmd_backbones)
    subparsers.add_parser(
        "data", help="inspect the dataset before training on it"
    ).set_defaults(handler=cmd_data)
    subparsers.add_parser("model", help="what a saved model was trained with").set_defaults(
        handler=cmd_model)
    subparsers.add_parser(
        "selectors", help="which page selectors are expected to rot"
    ).set_defaults(handler=cmd_selectors)

    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.add_argument("--backbone", default=backbones.DEFAULT_BACKBONE,
                              choices=sorted(backbones.BACKBONES))
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--fine-tune-epochs", type=int, default=0,
                              help="unfreeze the top of the backbone for this many "
                                   "more epochs, after the head has settled")
    train_parser.add_argument("--no-augment", action="store_true")
    train_parser.add_argument("--no-class-weights", action="store_true")
    train_parser.add_argument("--threshold-objective", default="f1",
                              choices=["f1", "accuracy", "precision", "recall"])
    train_parser.add_argument("--minimum-precision", type=float, default=0.0,
                              help="pick the threshold that maximises recall while "
                                   "keeping precision at or above this")
    train_parser.set_defaults(handler=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="classify images")
    predict_parser.add_argument("images", nargs="+")
    predict_parser.set_defaults(handler=cmd_predict)

    collect_parser = subparsers.add_parser("collect", help="gather cards to label")
    collect_parser.add_argument("--limit", type=int, default=50)
    collect_parser.add_argument("--max-minutes", type=float, default=30.0)
    collect_parser.set_defaults(handler=cmd_collect)

    swipe_parser = subparsers.add_parser("swipe", help="classify and optionally act")
    swipe_parser.add_argument("--live", action="store_true",
                              help="actually click. Without this nothing is clicked.")
    swipe_parser.add_argument("--limit", type=int, default=50)
    swipe_parser.add_argument("--min-delay", type=float, default=2.0)
    swipe_parser.add_argument("--max-delay", type=float, default=5.0)
    swipe_parser.add_argument("--max-minutes", type=float, default=30.0)
    swipe_parser.set_defaults(handler=cmd_swipe)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
