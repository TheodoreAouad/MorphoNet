"""Init layers module."""

import mlflow

PAD_MODE = "reflect"

if mlflow.active_run() is not None:
    mlflow.log_param("pad_mode", PAD_MODE)
