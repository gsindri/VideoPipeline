# Suppress TensorFlow warnings (only affects TF if it's imported later)
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF info/warning logs
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Disable oneDNN message

import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")

__all__ = ["__version__"]
__version__ = "0.0.2"
