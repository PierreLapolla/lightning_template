from utils.logger import log


def try_except(func: callable) -> callable:
    """
    Decorator to catch exceptions in a function.
    :param func: function to catch exceptions in
    :return: wrapper function that catches exceptions in the function
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            message = f"An exception occurred in {func.__name__}: {e}"
            log.error(message)

    return wrapper


if __name__ == "__main__":
    """
    Example usage of the try_except decorator.
    """

    @try_except
    def divide(a, b):
        return a / b

    divide(1, 0)
