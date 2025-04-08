from utils.try_except import try_except


def test_try_except_decorator(caplog):
    @try_except
    def divide(a, b):
        return a / b

    assert divide(4, 2) == 2, "Expected 4 / 2 to equal 2"
    assert divide(1, 0) is None, "Expected 1 / 0 to equal None"
