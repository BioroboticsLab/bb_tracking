"""
Legacy types for loading old data.
"""

from collections import namedtuple

Detection = namedtuple('Detection', ['id', 'timestamp', 'x', 'y', 'orientation', 'beeId', 'meta'])
Track = namedtuple('Track', ['id', 'ids', 'timestamps', 'meta'])
