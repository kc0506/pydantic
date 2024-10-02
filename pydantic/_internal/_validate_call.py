from __future__ import annotations as _annotations

import functools
import inspect
from functools import partial
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, LambdaType, MethodType
from typing import Any, Callable, Union, cast, get_args, get_overloads

import pydantic_core

from .. import ValidationError
from ..config import ConfigDict
from ..errors import PydanticErrorCodes, PydanticUserError
from ..plugin._schema_validator import PluggableSchemaValidator, create_schema_validator
from . import _generate_schema, _typing_extra
from ._config import ConfigWrapper

# This should be aligned with `GenerateSchema.match_types`
ValidateCallSupportedTypes = Union[
    LambdaType,
    FunctionType,
    MethodType,
    BuiltinFunctionType,
    BuiltinMethodType,
    functools.partial,
]


def get_name(func: ValidateCallSupportedTypes) -> str:
    return f'partial({func.func.__name__})' if isinstance(func, functools.partial) else func.__name__


def get_qualname(func: ValidateCallSupportedTypes) -> str:
    return f'partial({func.func.__qualname__})' if isinstance(func, functools.partial) else func.__qualname__


def _check_function_type(function: object):
    ERROR_CODE: PydanticErrorCodes = 'validate-call-type'

    supported_types = get_args(ValidateCallSupportedTypes)
    if isinstance(function, supported_types):
        try:
            inspect.signature(cast(ValidateCallSupportedTypes, function))
        except ValueError:
            raise PydanticUserError(f"Input function `{function}` doesn't have a valid signature", code=ERROR_CODE)

        if isinstance(function, partial):
            try:
                assert not isinstance(partial.func, partial), 'Partial of partial'
                _check_function_type(function.func)
            except PydanticUserError as e:
                raise PydanticUserError(
                    f'Partial of `{function.func}` is invalid because the type of `{function.func}` is not supported by `validate_call`',
                    code=ERROR_CODE,
                ) from e

        return

    if isinstance(function, (classmethod, staticmethod, property)):
        name = type(function).__name__
        raise PydanticUserError(
            f'The `@{name}` decorator should be applied after `@validate_call` (put `@{name}` on top)', code=ERROR_CODE
        )

    if inspect.isclass(function):
        raise PydanticUserError(
            '`validate_call` should be applied to functions, not classes (put `@validate_call` on top of `__init__` or `__new__` instead)',
            code=ERROR_CODE,
        )
    if callable(function):
        raise PydanticUserError(
            '`validate_call` should be applied to functions, not instances or other callables. Use `validate_call` explicitly on `__call__` instead.',
            code=ERROR_CODE,
        )

    raise PydanticUserError(
        '`validate_call` should be applied to one of the following: function, method, partial, or lambda',
        code=ERROR_CODE,
    )


def _update_wrapper(wrapped: ValidateCallSupportedTypes, wrapper: Callable[..., Any]):
    if inspect.iscoroutinefunction(wrapped):

        @functools.wraps(wrapped)
        async def wrapper_function(*args, **kwargs):  # type: ignore
            return await wrapper(*args, **kwargs)
    else:

        @functools.wraps(wrapped)
        def wrapper_function(*args, **kwargs):
            return wrapper(*args, **kwargs)

    # We need to manually update this because `partial` object has no `__name__` and `__qualname__`.
    wrapper_function.__name__ = get_name(wrapped)
    wrapper_function.__qualname__ = get_qualname(wrapped)

    return wrapper_function


def _validate_result(
    res: Any, validator: pydantic_core.SchemaValidator | PluggableSchemaValidator, is_awaitable: bool = False
):
    async def async_validate():
        assert inspect.isawaitable(res), f'Expected an awaitable, got {res}'
        return validator.validate_python(await res)

    if is_awaitable:
        return async_validate()
    return validator.validate_python(res)


def _create_wrapper(
    function: ValidateCallSupportedTypes,
    function_validator: pydantic_core.SchemaValidator | PluggableSchemaValidator,
    return_validator: pydantic_core.SchemaValidator | PluggableSchemaValidator | None,
    overload_args_validators: list[pydantic_core.SchemaValidator | PluggableSchemaValidator] | None,
    overload_return_validators: list[pydantic_core.SchemaValidator | PluggableSchemaValidator] | None,
):
    is_awaitable = inspect.iscoroutinefunction(function)

    def wrapper(*args, **kwargs):
        matched_return_validator = None

        if overload_args_validators is not None:
            args_kwargs = None
            assert overload_return_validators is not None
            for validator, overload_return_validator in zip(overload_args_validators, overload_return_validators):
                try:
                    args_kwargs = pydantic_core.ArgsKwargs(
                        *validator.validate_python(pydantic_core.ArgsKwargs(args, kwargs))
                    )
                    matched_return_validator = overload_return_validator
                    break
                except ValidationError:
                    pass

            if args_kwargs is None:
                raise ValidationError(f'No overload matched for {function}')
        else:
            assert overload_return_validators is None
            args_kwargs = pydantic_core.ArgsKwargs(args, kwargs)

        res = function_validator.validate_python(args_kwargs)

        if return_validator is not None:
            res = _validate_result(res, return_validator, is_awaitable)

        if matched_return_validator is not None:
            res = _validate_result(res, matched_return_validator, is_awaitable)

        return res

    wrapper_function = _update_wrapper(function, wrapper)
    wrapper_function.raw_function = function  # type: ignore
    return wrapper_function


def _get_return_annotation(func: Callable) -> Any:
    signature = inspect.signature(func)
    return signature.return_annotation if signature.return_annotation is not signature.empty else Any


def wrap_validate_call(
    function: ValidateCallSupportedTypes,
    config: ConfigDict | None,
    validate_return: bool,
    namespace: dict[str, Any] | None,
    use_overloads: bool = False,
):
    _check_function_type(function)

    if isinstance(function, partial):
        schema_type = function.func
        module = function.func.__module__
    else:
        schema_type = function
        module = function.__module__
    qualname = core_config_title = get_qualname(function)

    global_ns = _typing_extra.get_module_ns_of(function)
    # TODO: this is a bit of a hack, we should probably have a better way to handle this
    # specifically, we shouldn't be pumping the namespace full of type_params
    # when we take namespace and type_params arguments in eval_type_backport
    type_params = (namespace or {}).get('__type_params__', ()) + getattr(schema_type, '__type_params__', ())
    namespace = {
        **{param.__name__: param for param in type_params},
        **(global_ns or {}),
        **(namespace or {}),
    }
    config_wrapper = ConfigWrapper(config)
    core_config = config_wrapper.core_config(core_config_title)
    create_validator = partial(
        create_schema_validator,
        schema_type=schema_type,
        schema_type_module=module,
        schema_type_name=qualname,
        schema_kind='validate_call',
        config=core_config,
        plugin_settings=config_wrapper.plugin_settings,
    )

    def get_schema(obj: Any, _config_wrapper: ConfigWrapper | None = None):
        gen_schema = _generate_schema.GenerateSchema(_config_wrapper or config_wrapper, namespace)
        schema = gen_schema.clean_schema(gen_schema.generate_schema(obj))
        return schema

    function_validator = create_validator(schema=get_schema(function))

    if use_overloads:
        overloads = get_overloads(function)
        if not overloads:
            raise ValueError(f'No overloads found for {function}')

        overload_config = ConfigWrapper({**(config or {}), 'check_args_only': True})

        overload_args_validators = [create_validator(get_schema(func, overload_config)) for func in overloads]
        overload_return_validators = [
            create_validator(get_schema(_get_return_annotation(func), overload_config)) for func in overloads
        ]
    else:
        overload_args_validators = None
        overload_return_validators = None

    if validate_return:
        signature = inspect.signature(function)
        return_type = signature.return_annotation if signature.return_annotation is not signature.empty else Any
        gen_schema = _generate_schema.GenerateSchema(config_wrapper, namespace)
        schema = gen_schema.clean_schema(gen_schema.generate_schema(return_type))
        return_validator = create_validator(schema=schema)

    else:
        return_validator = None

    return _create_wrapper(
        function, function_validator, return_validator, overload_args_validators, overload_return_validators
    )
