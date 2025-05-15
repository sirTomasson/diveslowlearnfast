
import av

from datetime import datetime
from fs.memoryfs import MemoryFS
from diveslowlearnfast.memfs_cache import MemFSCache
from glob import glob

def main():
    memfs = MemoryFS()
    fs_cache = MemFSCache(memfs)
    files = glob('/Users/youritomassen/Projects/xai/data/Diving48/rgb/*.mp4')[:1000]

    print(f'Starting benchmark, cache = False')
    start = datetime.now()
    for file in files:
        with fs_cache.open(file) as f:
            container = av.open(f)
            next(container.decode(video=0))

    end = datetime.now()
    print(f'Benchmark complete, took: {(end - start)}')

    print(f'Starting benchmark, cache = True')
    start = datetime.now()
    for file in files:
        with fs_cache.open(file) as f:
            container = av.open(f)
            next(container.decode(video=0))

    end = datetime.now()
    print(f'Benchmark complete, took: {(end - start)}')


if __name__ == '__main__':
    main()