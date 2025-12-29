import redis
from app.core.config import settings

# Connection Pool for performance
pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=False # Keep raw bytes for efficiency, decode manually
)

def get_redis_client():
    return redis.Redis(connection_pool=pool)