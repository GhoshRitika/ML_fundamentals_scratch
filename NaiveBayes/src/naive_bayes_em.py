import warnings
import numpy as np

from src.utils import softmax, stable_log_sum
from src.naive_bayes import NaiveBayes


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data, that uses both unlabeled and
        labeled data in the Expectation-Maximization algorithm

    Note that the class definition above indicates that this class
        inherits from the NaiveBayes class. This means it has the same
        functions as the NaiveBayes class unless they are re-defined in this
        function. In particular you should be able to call `self.predict_proba`
        using your implementation from `src/naive_bayes.py`.
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def initialize_params(self, vocab_size, n_labels):
        """
        Initialize self.alpha such that
            `log p(y_i = k) = -log(n_labels)`
            for all k
        and initialize self.beta such that
            `log p(w_j | y_i = k) = -log(vocab_size)`
            for all j, k.

        """
        self.alpha= np.empty((n_labels))
        self.alpha[:] = -np.log(n_labels)
        self.beta = np.empty((vocab_size, n_labels))
        self.beta[:] = -np.log(vocab_size)
        # raise NotImplementedError

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the NaiveBayes superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.alpha and self.beta, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            the handout. In particular, Figure 2 shows the algorithm for
            semi-supervised Naive Bayes.

        self.alpha should contain the marginal probability of each class label.

        self.beta should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Note: if self.max_iter is 0, your function should call
            `self.initialize_params` and then break. In each
            iteration, you should complete both an E-step and
            an M-step.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size
        self.initialize_params(vocab_size, n_labels)
        if self.max_iter== 0:
            return
        self.y_copy = np.copy(y)
        max = 0
        while max <= self.max_iter:
            for i in range(n_labels):
                sum = 0
                for k in range(len(y)):
                    if np.isnan(y[k]):
                        self.y_copy[k] = self.predict_proba(X)[k,i]
                        sum += self.y_copy[k]
                    if self.y_copy[k] == i:
                        sum += 1
                    self.alpha[i] = np.log(sum/n_docs)
            self.beta= np.zeros([vocab_size, n_labels])
            for j in range(vocab_size):
                for m in range(n_labels):
                    num = 0
                    den =0
                    for i in range(n_docs):
                        if self.y_copy[i] == m:
                            num +=X[i, j] + self.smoothing
                        for k in range(vocab_size):
                            if self.y_copy[i] == m:
                                den +=X[i, k] + self.smoothing
                    self.beta[j, m] = np.log(num/den)
            max += 1
        # raise NotImplementedError

    def likelihood(self, X, y):
        r"""
        Using fit self.alpha and self.beta, compute the likelihood of the data.
            This function *should* use unlabeled data.
            This likelihood is defined in equation (14) of `naive_bayes.pdf`.

        For unlabeled data, we predict `p(y_i = y' | X_i)` using the
            previously-learned p(x|y, beta) and p(y | alpha).
            For labeled data, we define `p(y_i = y' | X_i)` as
            1 if `y_i = y'` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.

        Following equation (14) in the `naive_bayes.pdf` writeup, the log
            likelihood of the data after t iterations can be written as:

            \sum_{i=1}^N \log \sum_{y'=1}^2 \exp(
                \log p(y_i = y' | X_i, \alpha, \beta) + \alpha_{y'}
                + \sum_{j=1}^V X_{i, j} \beta_{j, y'})

            You can visualize this formula in http://latex2png.com

            The tricky aspect of this likelihood is that we are simultaneously
            computing $p(y_i = y' | X_i, \alpha^t, \beta^t)$ to predict a
            distribution over our latent variables (the unobserved $y_i$) while
            at the same time computing the probability of seeing such $y_i$
            using $p(y_i =y' | \alpha^t)$.

            Note: In implementing this equation, it will help to use your
                implementation of `stable_log_sum` to avoid underflow. See the
                documentation of that function for more details.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data.
        """

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2
        sum_a = 0
        sum_b = 0
        sum_d = 0
        sum_e = 0
        for i in range(n_docs):
            sum_b = 0
            for k in range(n_labels):
                if np.isnan(y[i]):
                    sum_d = np.log(self.y_copy[i])
                elif y[i] == k:
                    sum_d = np.log(1)
                sum_e = self.alpha[k]
                sum_c =0
                for j in range(vocab_size):
                    if X[i,j] == 0:
                        sum_c += 0
                    elif self.beta[j, int(self.y_copy[k])] < -745:
                        sum_c += -np.inf
                    else:
                        sum_c += X[i,j] * self.beta[j, k]
                sum_b += np.exp(sum_d + sum_e + sum_c)
            sum_a += np.log(sum_b)
        return sum_a
