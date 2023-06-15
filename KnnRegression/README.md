# HW 2: K Nearest Neighbors and Polynomial Regression

This assignment is due October 26 at 11:59pm CDT. There are two points of extra
credit for passing the `test_setup` test case, due *early* on October 19 at
11:59pm CDT. Late work will not be accepted except in extreme circumstances.

## Academic integrity

Your work must be your own. Do not submit other people's work as your own, and
do not allow others to submit your work as theirs. You may *talk* with other
students about the homework, but you may not share code with them in any way.
If you have a question about an error message or about why a numpy function
returns what it does, post it on Piazza. If you need help debugging your code,
make a *private* post on Piazza or come to office hours. If you're unsure of
the academic integrity policies, ask for help; we can help you avoid breaking
the rules, but we can't un-report a suspected violation.

By pushing your code to GitHub, you agree to these rules, and understand that
there may be severe consequences for violating them. 

## What's changed since HW1?

- The autograder now posts the detailed output of pytest into your GitHub repo
  rather than into the Canvas assignment comments.
- We've added one more piece to the `test_setup` case which requires you to
  understand how this feedback system works. See `problems.md` and
  `tests/test_a_setup.py` for details.
- We ask that you specify `python=3.9` in your conda (or other) environment, to
  avoid any small differences between it and python 3.10.
- The coding portion has decreased to be worth 10 points; the free-response
  has increased to 10 points. However, some FRQs rely on your code working
  before you can answer them.

## Important instructions

Your work will be graded and aggregated using an autograder that will download
the code and free response questions from each student's repository. If you
don't follow the instructions, you run the risk of getting *zero points*. The
`test_setup` test case gives you extra credit for following these instructions 
and will make it possible to grade your work easily.

The essential instructions:
- Your code and written answers must be *pushed* to GitHub for us to grade them!
  We will only grade the latest version of your code that was pushed to GitHub
  before the deadline.
- Your NetID must be in the `netid` file; replace `NETID_GOES_HERE` with your
  netid.
- Your answer to each free response question should be in *its own PDF* with
  the filename `XXX_qYYY.pdf`, where `XXX` is your NetID and `YYY` is the question
  number. So if your NetID is `xyz0123`, your answer to free response question 2
  should be in a PDF file with the filename `xyz0123_q2.pdf`.
- Please do not put your name in your free response PDFs -- we will grade these
  anonymously. 

## Clone this repository and environment setup

For more detailed versions of these instructions, refer to the HW1 README.
Once you have cloned this repository, you should create a virtual environment,
by running:

``conda create -n cs349hw2 python=3.9``

Note that we encourage you to use python 3.9, because that is what the autograder
uses.  You can *activate* the environment with:

``conda activate cs349hw2``

You will have to activate the environment every time you want to use it.
Navigate to the root directory of this repository and run:

``pip install -r requirements.txt``

## What to do for this assignment

The detailed instructions for the work you need to do are in `problems.md`.

For the coding portion of the assignment, you will:
- Implement mean squared error loss
- Implement three distance measures
- Generate polynomial data on which to fit your models 
- Implement a polynomial regression model
- Implement a k-nearest neighbor model

You will also write up answers to six free response questions, which
may require writing a bit more code.

In every function where you need to write code, there is a `raise
NotImplementeError` in the code. You will replace that line with code that
completes what the function docstring asks you to do.  The test cases will
guide you through the work you need to do and tell you how many points you've
earned. The test cases can be run from the root directory of this repository
with:

``python -m pytest -s``

To run a single test, you can specify it with `-k`, e.g., `python -m pytest -s
-k test_setup`.  To run a group of tests, you can use `-k` with a prefix, e.g.,
`python -m pytest -s -k test_knn` will run all tests that begin with
`test_knn`.  The `-s` means that any print statements you include will in
fact be printed; the default behavior (`python -m pytest`) will suppress
everything but the pytest output.

We will use these test cases to grade your work! Even if you change the test
cases such that you pass the tests on your computer, we're still going to use
the original test cases to grade your assignment.

## Questions? Problems? Issues?

Ask a question on Piazza, and we'll help you there.
