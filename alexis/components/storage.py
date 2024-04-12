"""Different storage implementations for storing json data."""
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from pymongo import MongoClient
from redis import Redis

from alexis.config import settings
from alexis.utils import load_entry_point

redis_client = Redis.from_url(settings.REDIS_URL)
mongo_client: MongoClient = MongoClient(settings.MONGO_URI)
mongo_db = mongo_client.get_database(settings.MONGO_DATABASE)

Projection = list[str] | None


def _(value) -> str:
    """Try to decode bytes."""
    return value.decode() if isinstance(value, bytes) else value


class Storage(ABC):
    """Abstract base class for storage."""

    @abstractmethod
    def get(
        self,
        collection: str,
        id: Any | None = None,
        only: Projection = None,
        **query,
    ) -> dict | None:
        """Get an object from the storage."""

    @abstractmethod
    def set(self, obj: dict, collection: str, id: Any) -> None:
        """Set an object in the storage."""

    @abstractmethod
    def delete(self, collection: str, id: Any) -> None:
        """Delete an object from the storage."""

    @abstractmethod
    def object_ids(
        self, collection: str, limit: int | None = None, skip: int = 0, **query
    ) -> Iterable[Any]:
        """Get the ids of all objects in the storage."""

    @abstractmethod
    def all(
        self,
        collection: str,
        limit: int | None = None,
        skip: int = 0,
        only: Projection = None,
        **query,
    ) -> Iterable[dict]:
        """Get all objects from the storage."""


class InMemoryStorage(Storage):
    """In-memory storage implementation."""

    _storage: dict[str, dict]

    def __init__(self):
        """Initialize the storage."""
        self._storage = {}

    def _apply_projection(self, obj: dict, only: Projection) -> dict:
        """Apply projection on the object."""
        if only is None:
            return obj
        return {key: obj[key] for key in only if key in obj}

    def get(
        self,
        collection: str,
        id: Any | None = None,
        only: Projection = None,
        **query,
    ) -> dict | None:
        """Get an object from the storage."""
        if id is not None:
            return self._storage.get(f"{collection}:{id}")
        for obj in self.all(collection, **query):
            return self._apply_projection(obj, only)
        return None

    def set(self, obj: dict, collection: str, id: Any) -> None:
        """Set an object in the storage."""
        old_obj = self._storage.get(f"{collection}:{id}", {})
        old_obj.update(obj)
        self._storage[f"{collection}:{id}"] = old_obj

    def delete(self, collection: str, id: Any) -> None:
        """Delete an object from the storage."""
        self._storage.pop(f"{collection}:{id}", None)

    def object_ids(
        self, collection: str, limit: int | None = None, skip: int = 0, **query
    ) -> Iterable[Any]:
        """Get all keys from the storage."""
        count = 0
        for key in self._storage.keys():
            if key.startswith(f"{collection}:"):
                if limit is not None and count > limit:
                    break
                count += 1
                if count <= skip:
                    continue
                yield _(key).split(":")[-1]

    def all(
        self,
        collection: str,
        limit: int | None = None,
        skip: int = 0,
        only: Projection = None,
        **query,
    ) -> Iterable[dict]:
        """Get all objects from the storage."""
        for key in self.object_ids(collection, limit, skip):
            obj = self._storage.get(key)
            if not obj:
                continue
            if all(obj.get(key) == value for key, value in query.items()):
                yield self._apply_projection(obj, only)


class RedisHashStorage(Storage):
    """Redis storage implementation using redis hash."""

    def _get_key(self, collection: str, id: Any) -> str:
        """Get the key for the object."""
        return f"{collection}:{id}"

    def _get_object(self, key: str) -> dict:
        """Get the object from the storage."""
        obj = redis_client.hgetall(key) or {}
        return {
            _(key): _(value)
            for key, value in obj.items()  # type: ignore
        }

    def _apply_projection(self, obj: dict, only: Projection) -> dict:
        """Apply projection on the object."""
        if only is None:
            return obj
        return {key: obj[key] for key in only if key in obj}

    def _set_object(self, key: str, obj: dict) -> None:
        """Set the object in the storage."""
        old_obj = self._get_object(key)
        old_obj.update(obj)
        redis_client.hset(key, mapping=old_obj)

    def get(
        self,
        collection: str,
        id: Any | None = None,
        only: Projection = None,
        **query,
    ) -> dict | None:
        """Get an object from the storage."""
        if id is not None:
            key = self._get_key(collection, id)
            return self._get_object(key) or None

        for obj in self.all(collection, **query):
            return self._apply_projection(obj, only)
        return None

    def set(
        self,
        obj: dict,
        collection: str,
        id: Any,
    ) -> None:
        """Set an object in the storage."""
        key = self._get_key(collection, id)
        self._set_object(key, obj)

    def delete(self, collection: str, id: Any) -> None:
        """Delete an object from the storage."""
        key = self._get_key(collection, id)
        redis_client.delete(key)

    def object_ids(
        self, collection: str, limit: int | None = None, skip: int = 0, **query
    ) -> Iterable[Any]:
        """Get all keys from the storage."""
        keys: list[bytes | str] = redis_client.keys(f"{collection}:*")  # type: ignore
        count = 0
        for key in keys:
            if limit is not None and count > limit:
                break
            count += 1
            if count <= skip:
                continue
            yield _(key).split(":")[-1]

    def all(
        self,
        collection: str,
        limit: int | None = None,
        skip: int = 0,
        only: Projection = None,
        **query,
    ) -> Iterable[dict]:
        """Get all objects from the storage."""
        for key in self.object_ids(collection, limit, skip):
            obj = self._get_object(key)
            if not obj:
                continue
            if all(obj.get(key) == value for key, value in query.items()):
                yield self._apply_projection(obj, only)


class RedisJsonStorage(RedisHashStorage):
    """Redis storage implementation using redis json."""

    def _get_object(self, key: str) -> dict:
        """Get the object from the storage."""
        obj: str = redis_client.get(key)  # type: ignore
        return json.loads(obj) if obj else {}

    def _set_object(self, key: str, obj: dict) -> None:
        """Set the object in the storage."""
        # perform partial updates if the object already exists
        old_obj = self._get_object(key)
        old_obj.update(obj)
        redis_client.set(key, json.dumps(old_obj))


class MongoStorage(Storage):
    """Mongodb storage implementation."""

    def get(
        self,
        collection: str,
        id: Any | None = None,
        only: Projection = None,
        **query,
    ) -> dict | None:
        """Get an object from the storage."""
        if id is not None:
            query["_id"] = id
        obj = mongo_db[collection].find_one(query, projection=only)
        if obj:
            obj.pop("_id", None)
        return obj

    def set(self, obj: dict, collection: str, id: Any) -> None:
        """Set an object in the storage."""
        if mongo_db[collection].count_documents({"_id": id}):
            mongo_db[collection].update_one({"_id": id}, {"$set": obj})
        else:
            obj["_id"] = id
            mongo_db[collection].insert_one(obj)

    def delete(self, collection: str, id: Any) -> None:
        """Delete an object from the storage."""
        mongo_db[collection].delete_one({"_id": id})

    def object_ids(
        self,
        collection: str,
        limit: int | None = None,
        skip: int = 0,
        **query,
    ) -> Iterable[Any]:
        """Get all keys from the storage."""
        db_query = mongo_db[collection].find(query, projection={"_id": True})
        if skip:
            db_query = db_query.skip(skip)
        if limit:
            db_query = db_query.limit(limit)
        for obj in db_query:
            yield obj["_id"]

    def all(
        self,
        collection: str,
        limit: int | None = None,
        skip: int = 0,
        only: Projection = None,
        **query,
    ) -> Iterable[dict]:
        """Get all objects from the storage."""
        db_query = mongo_db[collection].find(query, projection=only)
        if skip:
            db_query = db_query.skip(skip)
        if limit:
            db_query = db_query.limit(limit)
        for obj in db_query:
            obj.pop("_id", None)
            yield obj


DefaultStorageClass: type[Storage] = load_entry_point(settings.DEFAULT_STORAGE)
default_storage = DefaultStorageClass()


__all__ = [
    "Storage",
    "InMemoryStorage",
    "RedisHashStorage",
    "RedisJsonStorage",
    "MongoStorage",
    "DefaultStorageClass",
    "default_storage",
]
