"""Robustness patches for long Windows training runs (transient file locks on results.csv)."""

import time

from ultralytics.engine.trainer import BaseTrainer

_orig_save_metrics = BaseTrainer.save_metrics


def _save_metrics_retry(self, metrics):
    for attempt in range(8):
        try:
            return _orig_save_metrics(self, metrics)
        except PermissionError:
            time.sleep(1.0 + attempt)
    return _orig_save_metrics(self, metrics)


def install():
    BaseTrainer.save_metrics = _save_metrics_retry