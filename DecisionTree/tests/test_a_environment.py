def test_imports():
    """
    Please don't import sklearn or scipy to solve any of the problems in this
    assignment.  If you fail this test, we will give you a zero for this
    assignment, regardless of how sklearn or scipy was used in your code.

    the 'a' in the file name is so this test is run first.
    """
    import sys
    disallowed = ["scipy", "sklearn"]
    for key in list(sys.modules.keys()):
        for bad in disallowed:
            if bad in key:
                del sys.modules[key]

    import src
    for key in list(sys.modules.keys()):
        for bad in disallowed:
            assert bad not in key, f"Illegal import of {key}"
