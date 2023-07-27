

from redis import Redis as _Redis

class Redis(_Redis):
    def __init__(self, *args, **kwargs):
        kwargs['socket_timeout'] = 3600  # 1 hour
        kwargs['socket_connect_timeout'] = 3600  # 1 hour
        super().__init__(*args, **kwargs)

