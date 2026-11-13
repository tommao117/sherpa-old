"""Execute a Git subprocess with a lifetime-bound supervisor and bounded resources."""
import ctypes
import os
import resource
import signal
import subprocess
import sys


def terminate(signum, frame):
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    owner = int(sys.argv[1])
    signal.signal(signal.SIGTERM, terminate)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "parent-death protection unavailable")
    if os.getppid() != owner:
        terminate(None, None)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    cpu_seconds = int(os.environ.get("SHERPA_GIT_CPU_SECONDS", "15"))
    file_bytes = int(os.environ.get("SHERPA_GIT_FILE_LIMIT_MIB", "32")) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    child = subprocess.Popen(sys.argv[2:])
    return child.wait()


if __name__ == "__main__":
    sys.exit(main())
