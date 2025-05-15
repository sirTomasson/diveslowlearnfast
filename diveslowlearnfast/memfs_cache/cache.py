import os
import logging

from fs.memoryfs import MemoryFS


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'ERROR'))

class MemFSCache():

    def __init__(self, mem_fs: MemoryFS):
        self.mem_fs = mem_fs


    def open(self, path):
        path = os.path.abspath(path)
        if self.mem_fs.exists(path):
            logger.debug(f'reading from cache: {path}')
            file = self.mem_fs.open(path, 'rb')
        else:
            logger.debug(f'reading from disk: {path}')
            dirname = os.path.dirname(path)
            if not self.mem_fs.exists(dirname):
                self.mem_fs.makedirs(dirname)

            with open(path, 'rb') as source_file:
                content = source_file.read()
                with self.mem_fs.open(path, 'wb') as f:
                    f.write(content)

            file = self.mem_fs.open(path, 'rb')

        return file



