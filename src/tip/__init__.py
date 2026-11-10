"""tip: inspection policies with ML advice against adaptive electricity thieves."""
import os as _os

# Beberapa sandbox/kontainer melarang pthread_setaffinity_np; runtime OpenMP (PyTorch/MKL) lalu abort
# saat diimpor. Menonaktifkan afinitas aman di mesin biasa dan bisa ditimpa lewat variabel lingkungan.
_os.environ.setdefault("KMP_AFFINITY", "disabled")

__version__ = "0.1.0"
