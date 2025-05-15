import os
import threading
import hashlib
import tempfile
import uuid
import logging

SHM_DIR = "/dev/shm" if os.path.exists("/dev/shm") else tempfile.gettempdir()

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

class SHMCache:
    _instance_lock = threading.Lock()
    _instance = None

    _shared_registry = None

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(SHMCache, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        """Initialize the cache."""
        self.process_id = str(uuid.uuid4())

    def _get_shm_path(self, path):
        """Get shared memory path for a file."""
        file_hash = hashlib.md5(path.encode()).hexdigest()
        return os.path.join(SHM_DIR, f"shm_mmap_{file_hash}")

    def open(self, path):
        """Get a memory-mapped file from shared memory."""
        path = os.path.abspath(path)

        shm_path = self._get_shm_path(path)
        logger.debug(f'shared memory path: {shm_path}')
        if os.path.exists(shm_path):
            logger.debug(f'reading from cache: {path}')
            return open(shm_path, "rb")

        logger.debug(f'creating cache for: {path}')
        with open(path, 'rb') as f:
            content = f.read()
            with open(shm_path, "wb") as f_shm:
                f_shm.write(content)

            return open(shm_path, "rb")


def get_cache_instance():
    return SHMCache()
