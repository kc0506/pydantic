"""Decorator for validating function calls."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast, overload

from ._internal import _typing_extra, _validate_call

__all__ = ('validate_call',)

if TYPE_CHECKING:
    from .config import ConfigDict

    AnyCallableT = TypeVar('AnyCallableT', bound=Callable[..., Any])


@overload
def validate_call(
    *,
    config: ConfigDict | None = None,
    validate_return: bool = False,
    use_overloads: bool = False,
) -> Callable[[AnyCallableT], AnyCallableT]: ...


@overload
def validate_call(func: AnyCallableT, /) -> AnyCallableT: ...


def validate_call(
    func: AnyCallableT | None = None,
    /,
    *,
    config: ConfigDict | None = None,
    validate_return: bool = False,
    use_overloads: bool = False,
) -> AnyCallableT | Callable[[AnyCallableT], AnyCallableT]:
    """Usage docs: https://docs.pydantic.dev/2.10/concepts/validation_decorator/

    Returns a decorated wrapper around the function that validates the arguments and, optionally, the return value.

    Usage may be either as a plain decorator `@validate_call` or with arguments `@validate_call(...)`.

    Args:
        func: The function to be decorated.
        config: The configuration dictionary.
        validate_return: Whether to validate the return value.
        use_overloads: Whether to validate against the annotations of overloads instead of that of the decorated function.

    Returns:
        The decorated function.
    """
    local_ns = _typing_extra.parent_frame_namespace()

    def validate(function: AnyCallableT) -> AnyCallableT:
        validate_call_wrapper = _validate_call.ValidateCallWrapper(
            cast(_validate_call.ValidateCallSupportedTypes, function), config, validate_return, local_ns
        )
        return _validate_call.update_wrapper(function, validate_call_wrapper.__call__)  # type: ignore

    if func is not None:
        return validate(func)
    else:
        return validate
