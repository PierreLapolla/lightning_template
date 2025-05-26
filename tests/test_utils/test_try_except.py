from utils.try_except import try_except


# Helper to create callables and flag container
def create_callables():
    flags = {"cleanup_called": False, "error_called": False}

    def cleanup():
        flags["cleanup_called"] = True

    def error_handler():
        flags["error_called"] = True

    return cleanup, error_handler, flags


def reset_flags(flags):
    flags["cleanup_called"] = False
    flags["error_called"] = False


def test_divide_success_cleanup():
    cleanup, _, flags = create_callables()

    @try_except(finally_callable=cleanup)
    def divide(a, b):
        return a / b

    reset_flags(flags)
    result = divide(10, 2)
    assert result == 5
    assert flags["cleanup_called"]


def test_divide_exception_cleanup():
    cleanup, _, flags = create_callables()

    @try_except(finally_callable=cleanup)
    def divide(a, b):
        return a / b

    reset_flags(flags)
    result = divide(1, 0)
    assert result is None
    assert flags["cleanup_called"]


def test_divide_exception_error_callable():
    cleanup, error_handler, flags = create_callables()

    @try_except(finally_callable=cleanup, error_callable=error_handler)
    def divide(a, b):
        return a / b

    reset_flags(flags)
    result = divide(1, 0)
    assert result is None
    assert flags["error_called"]
    assert flags["cleanup_called"]


def test_multiply_success_no_error():
    cleanup, error_handler, flags = create_callables()

    @try_except(finally_callable=cleanup, error_callable=error_handler)
    def multiply(a, b):
        return a * b

    reset_flags(flags)
    result = multiply(3, 4)
    assert result == 12
    assert flags["cleanup_called"]
    assert not flags["error_called"]


def test_function_with_kwargs():
    cleanup, error_handler, flags = create_callables()

    @try_except(finally_callable=cleanup, error_callable=error_handler)
    def add(a=0, b=0):
        return a + b

    # Valid keyword arguments call
    reset_flags(flags)
    result = add(a=7, b=5)
    assert result == 12
    assert flags["cleanup_called"]
    assert not flags["error_called"]

    # Invalid call triggering exception
    reset_flags(flags)
    result = add(a="a", b=5)
    assert result is None
    assert flags["cleanup_called"]
    assert flags["error_called"]


def test_no_args_function():
    cleanup, error_handler, flags = create_callables()

    @try_except(finally_callable=cleanup, error_callable=error_handler)
    def greet():
        return "hello"

    reset_flags(flags)
    result = greet()
    assert result == "hello"
    assert flags["cleanup_called"]
    assert not flags["error_called"]
