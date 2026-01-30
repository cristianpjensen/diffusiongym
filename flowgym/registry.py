"""Registry system for base models and rewards."""

from typing import Any, Callable, Generic, List, TypeVar, overload

T = TypeVar("T")


class RegistryEntry(Generic[T]):
    """Entry in a registry containing the class and metadata.

    Parameters
    ----------
    cls : type[T]
        The class to register.

    kwargs : dict[str, Any] | None
        Default kwargs for the class constructor.
    """

    def __init__(self, cls: type[T], kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.default_kwargs = kwargs or {}

    def instantiate(self, **override_kwargs: Any) -> T:
        """Instantiate the registered class with given kwargs.

        Parameters
        ----------
        **override_kwargs : dict
            Keyword arguments to pass to the class constructor. These override any default kwargs.

        Returns
        -------
        instance : T
            An instance of the registered class.
        """
        kwargs = {**self.default_kwargs, **override_kwargs}
        return self.cls(**kwargs)


class Registry(Generic[T]):
    """Generic registry for mapping string identifiers to classes.

    Parameters
    ----------
    name : str
        Name of the registry (for error messages).
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, RegistryEntry[T]] = {}

    @overload
    def register(
        self,
        id: str,
        entry_point: type[T],
        **default_kwargs: Any,
    ) -> type[T]: ...

    @overload
    def register(
        self,
        id: str,
        entry_point: None = None,
        **default_kwargs: Any,
    ) -> Callable[[type[T]], type[T]]: ...

    def register(
        self,
        id: str,
        entry_point: type[T] | None = None,
        **default_kwargs: Any,
    ) -> type[T] | Callable[[type[T]], type[T]]:
        """Register a class in the registry.

        Can be used as a decorator or called directly.

        Parameters
        ----------
        id : str
            Unique identifier for the class.

        entry_point : type[T], optional
            The class to register. If None, returns a decorator.

        **default_kwargs : Any
            Default keyword arguments for the class constructor.

        Returns
        -------
        decorator : Callable[[type[T]], type[T]]
            Decorator function (if entry_point is None) or the class itself.

        Examples
        --------
        As a decorator:
        >>> @base_model_registry.register("my_base_model")
        ... class MyBaseModel(BaseModel):
        ...     pass

        Direct call:
        >>> base_model_registry.register("my_base_model", MyBaseModel)
        """
        if entry_point is not None:
            if id in self._registry:
                raise ValueError(f"{id} is already registered in {self.name} with a different class")

            self._registry[id] = RegistryEntry(entry_point, default_kwargs)
            return entry_point

        # Return decorator
        def _decorator(cls: type[T]) -> type[T]:
            if id in self._registry:
                raise ValueError(f"{id} is already registered in {self.name} with a different class")

            self._registry[id] = RegistryEntry(cls, default_kwargs)
            return cls

        return _decorator

    def get(self, id: str) -> RegistryEntry[T]:
        """Get a registry entry by ID.

        Parameters
        ----------
        id : str
            The unique identifier.

        Returns
        -------
        entry : RegistryEntry[T]
            The registry entry.

        Raises
        ------
        KeyError
            If the ID is not registered.
        """
        if id not in self._registry:
            raise KeyError(f"{id} is not registered in {self.name}. Available: {', '.join(sorted(self._registry.keys()))}")

        return self._registry[id]

    def list(self) -> List[str]:
        """List all registered IDs.

        Returns
        -------
        ids : List[str]
            Sorted list of registered IDs.
        """
        return sorted(self._registry.keys())

    def __contains__(self, id: str) -> bool:
        """Check if an ID is registered.

        Parameters
        ----------
        id : str
            The unique identifier.

        Returns
        -------
        is_registered : bool
            True if the ID is registered.
        """
        return id in self._registry


# Global registries
base_model_registry = Registry("BASE_MODEL_REGISTRY")  # type: ignore
reward_registry = Registry("REWARD_REGISTRY")  # type: ignore
