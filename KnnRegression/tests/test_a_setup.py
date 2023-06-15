import filetype
import json
import os
import random
import re

n_free_response = 6


def test_setup():
    '''
    For HW2, we have changed how the autograder gives feedback (by pushing its
    pytest output to your github repo, rather than posting it to Canvas). To help
    you get used to this, we have added another piece to the `test_setup` test that
    will require you to read the feedback that's added to your github repository.
    After you add your NetID to the `netid` file and create your PDFs, you should
    get an error such as:

    ```
        assert inf.readline().strip() == secret, msg
        AssertionError: See tests/test_a_setup.py for details on this error.
        assert 'password' != '', 
    ```

    Go ahead and commit and push your code anyways, and wait for the autograder
    to run. When it does, it will create a new `feedback/` folder in your
    repository with a file named something like `Oct_14_12_00__abcd1234.txt`.
    You can see this file on GitHub, or download it by calling `git pull origin
    main`. This file will show the autograder pytest output, and will contain a
    similar error message to the one you saw before, except it will contain
    your password:

    ```
        assert inf.readline().strip() == secret, msg
        AssertionError: See tests/test_a_setup.py for details on this error.
        assert 'password' != '2da3e727', 
    ```

    In this example, `2da3e727` is your password. You need to add that to your
    `password` file, replacing the `autograder_password_goes_here` with a single
    line containing this password.  Note that when you try to commit and push this
    change, you may get a scary-looking git error such as:

    ```
       ! [rejected]        main -> main (fetch first)
      error: failed to push some refs to 'git@github.com:nucs349f22/hw2-knn-regression-username.git'
      hint: Updates were rejected because the remote contains work that you do
      hint: not have locally. This is usually caused by another repository pushing
      hint: to the same ref. You may want to first integrate the remote changes
      hint: (e.g., 'git pull ...') before pushing again.
      hint: See the 'Note about fast-forwards' in 'git push --help' for details.
    ```

    All this means is that the autograder has successfully given you feedback,
    and you need to run `git pull origin main` to download it from GitHub to
    your local machine. When you do, it will likely open the [git editor](
    https://stackoverflow.com/questions/2596805/how-do-i-make-git-use-the-editor-of-my-choice-for-editing-commit-messages),
    which may be [vim by
    default](https://stackoverflow.com/questions/11828270/how-do-i-exit-vim).
    All you need to do is type `:wq<Enter>` or `ZZ` to exit vim.

    You should then get a message saying `Merge made by the 'recursive'
    strategy`, and the autograder feedback will now be available in the
    `feedback/` folder on your local machine.  You can call `git push origin
    main` to push your updated `password` to your repo, so that you can pass
    the autograder's `test_setup`. If you want to pass `test_setup` locally,
    you should also put your NetID and this password in `tests/secrets.txt`; if
    your NetID is `xyz0123` and the password given to you is `2da3e727`, then
    put `xyz0123:2da3e727` in `tests/secrets.txt`.

    This additional hurdle is designed to help you understand what the autograder
    is doing. It is running `python -m pytest` (or just `python -m pytest -k
    test_setup` for the setup extra credit) and giving you the output in your
    `feedback/` folder. Take advantage of this feedback! If you are passing tests
    locally but not on the autograder, it's important to understand why so that you
    can fix those issues and get credit for your work. Note that your grade (and a
    summary of tests passed) will still be uploaded to Canvas.
    '''

    with open('netid', 'r') as inf:
        lines = inf.readlines()

    assert len(lines) == 1, "Just a single line with your NetID"

    netid = str(lines[0].strip())
    assert netid != "NETID_GOES_HERE", "Add your NetID"
    assert netid.lower() == netid, "Lowercase NetID, please"
    assert re.search(r"^[a-z]{3}[0-9]{3,4}$", netid) is not None, "Your NetID looks like xyz0123"

    files = os.listdir(".")
    for i in range(1, 1 + n_free_response):
        fn = f"{netid}_q{i}.pdf"
        assert fn in files, f"Please create {fn}"
        guess = filetype.guess(fn)
        msg = f"Is {fn} actually a pdf?"
        assert guess is not None, msg
        assert guess.mime == 'application/pdf', msg

    with open("password", "r") as inf:
        msg = "See tests/test_a_setup.py for details on this error."
        secret = get_feedback_secret(netid)
        assert inf.readline().strip() == secret, msg


def get_feedback_secret(netid):
    '''
    On the autograder server, this will grab a 'password' for you and compare
    it against what you have in the `password` file. Once you see the feedback
    the autograder pushes to your repository, you can find the password and add
    it to your `password` file. To pass this test locally, put `netid:password`
    in `tests/secrets.txt`; e.g., if your NetID is xyz0123 and your password
    were abcd1234, you would add a line with `xyz0123:abcd1234`.
    '''
    fn = "tests/secrets.txt"
    if os.path.exists(fn):
        with open(fn) as inf:
            for line in inf:
                if line.strip().startswith(netid):
                    return line.strip().split(":")[1]

    return 'ed02c6c7'
