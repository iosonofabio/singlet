# vim: fdm=indent
# author:     Fabio Zanini
# date:       17/06/20
# content:    Custom ks_2samp that also returns the position of the max
import warnings
from collections import namedtuple
import math
from math import gcd
import numpy as np
from scipy.stats import distributions
import scipy.special as special


KstestResult = namedtuple(
        'KstestResult',
        ('statistic', 'value', 'pvalue'))


def _compute_prob_inside_method(m, n, g, h):
    """
    Count the proportion of paths that stay strictly inside two diagonal lines.
    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)
    Returns
    -------
    p : float
        The proportion of paths that stay inside the two lines.
    Count the integer lattice paths from (0, 0) to (m, n) which satisfy
    |x/m - y/n| < h / lcm(m, n).
    The paths make steps of size +1 in either positive x or positive y directions.
    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    """
    # Probability is symmetrical in m, n.  Computation below uses m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Count the integer lattice paths from (0, 0) to (m, n) which satisfy
    # |nx/g - my/g| < h.
    # Compute matrix A such that:
    #  A(x, 0) = A(0, y) = 1
    #  A(x, y) = A(x, y-1) + A(x-1, y), for x,y>=1, except that
    #  A(x, y) = 0 if |x/m - y/n|>= h
    # Probability is A(m, n)/binom(m+n, n)
    # Optimizations exist for m==n, m==n*p.
    # Only need to preserve a single column of A, and only a sliding window of it.
    # minj keeps track of the slide.
    minj, maxj = 0, min(int(np.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    # Make a vector long enough to hold maximum window needed.
    lenA = min(2 * maxj + 2, n + 1)
    # This is an integer calculation, but the entries are essentially
    # binomial coefficients, hence grow quickly.
    # Scaling after each column is computed avoids dividing by a
    # large binomial coefficent at the end, but is not sufficient to avoid
    # the large dyanamic range which appears during the calculation.
    # Instead we rescale based on the magnitude of the right most term in
    # the column and keep track of an exponent separately and apply
    # it at the end of the calculation.  Similarly when multiplying by
    # the binomial coefficint
    dtype = np.float64
    A = np.zeros(lenA, dtype=dtype)
    # Initialize the first column
    A[minj:maxj] = 1
    expnt = 0
    for i in range(1, m + 1):
        # Generate the next column.
        # First calculate the sliding window
        lastminj, lastlen = minj, curlen
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 0
        # Now fill in the values
        A[0:maxj - minj] = np.cumsum(A[minj - lastminj:maxj - lastminj])
        curlen = maxj - minj
        if lastlen > curlen:
            # Set some carried-over elements to 0
            A[maxj - minj:maxj - minj + (lastlen - curlen)] = 0
        # Rescale if the right most value is over 2**900
        val = A[maxj - minj - 1]
        _, valexpt = math.frexp(val)
        if valexpt > 900:
            # Scaling to bring down to about 2**800 appears
            # sufficient for sizes under 10000.
            valexpt -= 800
            A = np.ldexp(A, -valexpt)
            expnt += valexpt

    val = A[maxj - minj - 1]
    # Now divide by the binomial (m+n)!/m!/n!
    for i in range(1, n + 1):
        val = (val * i) / (m + i)
        _, valexpt = math.frexp(val)
        if valexpt < -128:
            val = np.ldexp(val, -valexpt)
            expnt += valexpt
    # Finally scale if needed.
    return np.ldexp(val, expnt)


def _compute_prob_outside_square(n, h):
    """
    Compute the proportion of paths that pass outside the two diagonal lines.
    Parameters
    ----------
    n : integer
        n > 0
    h : integer
        0 <= h <= n
    Returns
    -------
    p : float
        The proportion of paths that pass outside the lines x-y = +/-h.
    """
    # Compute Pr(D_{n,n} >= h/n)
    # Prob = 2 * ( binom(2n, n-h) - binom(2n, n-2a) + binom(2n, n-3a) - ... )  / binom(2n, n)
    # This formulation exhibits subtractive cancellation.
    # Instead divide each term by binom(2n, n), then factor common terms
    # and use a Horner-like algorithm
    # P = 2 * A0 * (1 - A1*(1 - A2*(1 - A3*(1 - A4*(...)))))

    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


def _count_paths_outside_method(m, n, g, h):
    """
    Count the number of paths that pass outside the specified diagonal.
    Parameters
    ----------
    m : integer
        m > 0
    n : integer
        n > 0
    g : integer
        g is greatest common divisor of m and n
    h : integer
        0 <= h <= lcm(m,n)
    Returns
    -------
    p : float
        The number of paths that go low.
        The calculation may overflow - check for a finite answer.
    Exceptions
    ----------
    FloatingPointError: Raised if the intermediate computation goes outside
    the range of a float.
    Notes
    -----
    Count the integer lattice paths from (0, 0) to (m, n), which at some
    point (x, y) along the path, satisfy:
      m*y <= n*x - h*g
    The paths make steps of size +1 in either positive x or positive y directions.
    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk.
    Hodges, J.L. Jr.,
    "The Significance Probability of the Smirnov Two-Sample Test,"
    Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    """
    # Compute #paths which stay lower than x/m-y/n = h/lcm(m,n)
    # B(x, y) = #{paths from (0,0) to (x,y) without previously crossing the boundary}
    #         = binom(x, y) - #{paths which already reached the boundary}
    # Multiply by the number of path extensions going from (x, y) to (m, n)
    # Sum.

    # Probability is symmetrical in m, n.  Computation below assumes m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Not every x needs to be considered.
    # xj holds the list of x values to be checked.
    # Wherever n*x/m + ng*h crosses an integer
    lxj = n + (mg-h)//mg
    xj = [(h + mg * j + ng-1)//ng for j in range(lxj)]
    # B is an array just holding a few values of B(x,y), the ones needed.
    # B[j] == B(x_j, j)
    if lxj == 0:
        return np.round(special.binom(m + n, n))
    B = np.zeros(lxj)
    B[0] = 1
    # Compute the B(x, y) terms
    # The binomial coefficient is an integer, but special.binom() may return a float.
    # Round it to the nearest integer.
    for j in range(1, lxj):
        Bj = np.round(special.binom(xj[j] + j, j))
        if not np.isfinite(Bj):
            raise FloatingPointError()
        for i in range(j):
            bin = np.round(special.binom(xj[j] - xj[i] + j - i, j-i))
            Bj -= bin * B[i]
        B[j] = Bj
        if not np.isfinite(Bj):
            raise FloatingPointError()
    # Compute the number of path extensions...
    num_paths = 0
    for j in range(lxj):
        bin = np.round(special.binom((m-xj[j]) + (n - j), n-j))
        term = B[j] * bin
        if not np.isfinite(term):
            raise FloatingPointError()
        num_paths += term
    return np.round(num_paths)


def _attempt_exact_2kssamp(n1, n2, g, d, alternative):
    """Attempts to compute the exact 2sample probability.
    n1, n2 are the sample sizes
    g is the gcd(n1, n2)
    d is the computed max difference in ECDFs
    Returns (success, d, probability)
    """
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    saw_fp_error, prob = False, np.nan
    try:
        if alternative == 'two-sided':
            if n1 == n2:
                prob = _compute_prob_outside_square(n1, h)
            else:
                prob = 1 - _compute_prob_inside_method(n1, n2, g, h)
        else:
            if n1 == n2:
                # prob = binom(2n, n-h) / binom(2n, n)
                # Evaluating in that form incurs roundoff errors
                # from special.binom. Instead calculate directly
                jrange = np.arange(h)
                prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
            else:
                num_paths = _count_paths_outside_method(n1, n2, g, h)
                bin = special.binom(n1 + n2, n1)
                if not np.isfinite(bin) or not np.isfinite(num_paths) or num_paths > bin:
                    saw_fp_error = True
                else:
                    prob = num_paths / bin

    except FloatingPointError:
        saw_fp_error = True

    if saw_fp_error:
        return False, d, np.nan
    if not (0 <= prob <= 1):
        return False, d, prob
    return True, d, prob


def ks_2samp(data1, data2, alternative='two-sided', mode='auto'):
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.  The alternative hypothesis
    can be either 'two-sided' (default), 'less' or 'greater'.
    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
          * 'two-sided'
          * 'less': one-sided, see explanation in Notes
          * 'greater': one-sided, see explanation in Notes
    mode : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):
          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic
    Returns
    -------
    statistic : float
        KS statistic, i.e. max distance in the cumulative distributions of
        data1 and data2. Sign is + if the cumulative of data1 < the one of
        data2 at that location, else -.
    x : float
        Value of x where maximal distance is reached.
    pvalue : float
        Two-tailed p-value.
    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample KS test, the distribution is
    assumed to be continuous.
    In the one-sided test, the alternative is that the empirical
    cumulative distribution function F(x) of the data1 variable is "less"
    or "greater" than the empirical cumulative distribution function G(x)
    of the data2 variable, ``F(x)<=G(x)``, resp. ``F(x)>=G(x)``.
    If the KS statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.
    If the mode is 'auto', the computation is exact if the sample sizes are
    less than 10000.  For larger sizes, the computation uses the
    Kolmogorov-Smirnov distributions to compute an approximate value.
    The 'two-sided' 'exact' computation computes the complementary probability
    and then subtracts from 1.  As such, the minimum probability it can return
    is about 1e-16.  While the algorithm itself is exact, numerical
    errors may accumulate for large sample sizes.   It is most suited to
    situations in which one of the sample sizes is only a few thousand.
    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk [1]_.
    References
    ----------
    .. [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
           Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    Examples
    --------
    >>> from scipy import stats
    >>> import anndataks
    >>> np.random.seed(12345678)  #fix random seed to get the same result
    >>> n1 = 200  # size of first sample
    >>> n2 = 300  # size of second sample
    For a different distribution, we can reject the null hypothesis since the
    pvalue is below 1%:
    >>> rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
    >>> rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
    >>> anndataks.ks_2samp(rvs1, rvs2)
    (0.20833333333333334, 0.25, 5.129279597781977e-05)
    For a slightly different distribution, we cannot reject the null hypothesis
    at a 10% or lower alpha since the p-value at 0.144 is higher than 10%
    >>> rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
    >>> anndataks.ks_2samp(rvs1, rvs3)
    (0.10333333333333333, 0.25, 0.14691437867433876)
    For an identical distribution, we cannot reject the null hypothesis since
    the p-value is high, 41%:
    >>> rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
    >>> anndataks.ks_2samp(rvs1, rvs4)
    (0.07999999999999996, 0, 0.41126949729859719)
    """
    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
    if np.ma.is_masked(data1):
        data1 = data1.compressed()
    if np.ma.is_masked(data2):
        data2 = data2.compressed()

    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    idx_cdf1 = np.searchsorted(data1, data_all, side='right')
    idx_cdf2 = np.searchsorted(data2, data_all, side='right')
    # These are the fractions, used in the computations. We keep the array
    # above as reference to get back the value of the distributions at those
    # indices
    cdf1 = idx_cdf1 / n1
    cdf2 = idx_cdf2 / n2
    cddiffs = cdf1 - cdf2
    # idx_ keep track of indices of cdf1, cdf2, cddiffs
    idx_minS = np.argmin(cddiffs)
    idx_maxS = np.argmax(cddiffs)
    minS = -cddiffs[idx_minS]
    maxS = cddiffs[idx_maxS]
    alt2Dvalue = {
        'less': (idx_minS, minS),
        'greater': (idx_maxS, maxS),
        'two-sided': (idx_minS, minS) if minS >= maxS else (idx_maxS, maxS),
        }
    idx_d, d = alt2Dvalue[alternative]

    # Figure out which of the two is higher at that index
    sign = 1 if cddiffs[idx_d] < 0 else -1

    # Track down the two values at that index
    # The -1 is because we used 'right' in sortedsearch
    idx1 = idx_cdf1[idx_d] - 1
    idx2 = idx_cdf2[idx_d] - 1
    # Get the values of the distributions at those indices and the midpoint,
    # as a way to deal with short arrays
    val1 = data1[idx1]
    val2 = data2[idx2]
    val_midpoint = 0.5 * (val1 + val2)

    g = gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -np.inf
    original_mode = mode
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
    elif mode == 'exact':
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int_).max / n2g:
            mode = 'asymp'
            warnings.warn(
                "Exact ks_2samp calculation not possible with samples sizes "
                "%d and %d. Switching to 'asymp' " % (n1, n2), RuntimeWarning)

    if mode == 'exact':
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'
            if original_mode == 'exact':
                warnings.warn(f"ks_2samp: Exact calculation unsuccessful. "
                              f"Switching to mode={mode}.", RuntimeWarning)

    if mode == 'asymp':
        # The product n1*n2 is large.  Use Smirnov's asymptoptic formula.
        if alternative == 'two-sided':
            en = n1 * n2 / (n1 + n2)
            #prob = distributions.kstwo.sf(d, np.round(en))
            prob = distributions.kstwobign.sf(d * np.sqrt(en))
        else:
            m, n = max(n1, n2), min(n1, n2)
            z = np.sqrt(m*n/(m+n)) * d
            # Use Hodges' suggested approximation Eqn 5.3
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

    prob = np.clip(prob, 0, 1)
    return KstestResult(sign * d, prob, val_midpoint)
