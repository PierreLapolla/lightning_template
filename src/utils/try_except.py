from utils.logger import log
from typing import Callable, Any, Optional, TypeVar


def try_except(
    _func=None,
    *,
    finally_callable: Optional[Callable[[], Any]] = None,
    error_callable: Optional[Callable[[], Any]] = None,
) -> Callable:
    """
    Decorator to catch exceptions in a function.
    Optionally execute a callable in the finally block or when an error is caught.

    :param _func: The function to decorate.
    :param finally_callable: Optional parameterless callable to execute in finally.
    :param error_callable: Optional parameterless callable to execute when an error is caught.
    :return: The wrapper function.
    """
    R = TypeVar("R")

    def decorator_func(func: Callable[..., R]) -> Callable[..., Optional[R]]:
        def wrapper(*args, **kwargs) -> Optional[R]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = f"{func.__name__}: {e.__class__.__name__}: {e}"
                log.error(message)
                if error_callable:
                    error_callable()
                    log.debug(
                        f"{error_callable.__name__} executed from {func.__name__}"
                    )
                return None
            finally:
                if finally_callable:
                    finally_callable()
                    log.debug(
                        f"{finally_callable.__name__} executed from {func.__name__}"
                    )

        return wrapper

    if _func is None:
        return decorator_func
    else:
        return decorator_func(_func)


if __name__ == "__main__":
    """
    Example usage of the try_except decorator with both error and finally callables.
    """

    def cleanup():
        print("Cleanup executed.")

    def handle_error():
        print("Error handled.")

    @try_except(error_callable=handle_error, finally_callable=cleanup)
    def divide(a, b):
        return a / b

    divide(1, 0)
