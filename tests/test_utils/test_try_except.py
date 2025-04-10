from utils.try_except import try_except


def test_try_except_decorator(caplog):
    flag = {"cleanup_executed": False}

    def cleanup():
        flag["cleanup_executed"] = True

    @try_except(finally_callable=cleanup)
    def divide(a, b):
        return a / b

    flag["cleanup_executed"] = False
    result = divide(4, 2)
    assert result == 2, "Expected 4 / 2 to equal 2"
    assert flag["cleanup_executed"] is True, "Expected finally_callable to be executed"

    flag["cleanup_executed"] = False
    result = divide(1, 0)
    assert result is None, "Expected 1 / 0 to return None due to exception"
    assert flag["cleanup_executed"] is True, (
        "Expected finally_callable to be executed even on exception"
    )
