from enum import Enum
from typing import Union, Iterable


def validate_enum_class(enum_class: Enum) -> None:
    if not isinstance(enum_class, type) or not issubclass(enum_class, Enum):
        raise TypeError(f"{enum_class} is not valid Enum class. Please provide a valid Enum class.")


def contains_name(
    enum_class: Enum,
    name: Union[str, int],
) -> bool:
    validate_enum_class(enum_class)

    if any(name in member.name for member in enum_class):
        return True
    else:
        raise AttributeError(
            f"Attribute error: {enum_class.__name__} is not in {name}."
        )


def contains_names(
    enum_class: Enum,
    names: Iterable[Union[str, int]],
) -> bool:
    validate_enum_class(enum_class)

    for name in names:
        if not any(name in member.name for member in enum_class):
            raise AttributeError(
                f"Attribute error: {enum_class.__name__} is not in {name}."
            )
    return True


def contains_one_of(
    enum_class: Enum,
    names: Iterable[Union[str, int]],
) -> bool:
    validate_enum_class(enum_class)

    for name in names:
        if any(name in member.name for member in enum_class):
            return True

    raise AttributeError(
        f"Attribute error: {enum_class.__name__} has anything."
    )
