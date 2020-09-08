def simpson(f, a, n, h, progress_bar=None):
    """
    Simpson rule for integral approximation of function `f` for `n` steps/strips of width `h`,
    starting from point `a`.
    :param f: the function to evaluate the integral for
    :param a: left bound for area evaluation
    :param n: number of strips
    :param h: strip size
    :param progress_bar: a method that wraps the `range` generator and produces a progress bar.
    e.g. use the package `tqdm`, and the progress bar will be called as: `tqdm(range(n))`
    :return: the integral value
    """
    res = 0
    progress_range = range(n)
    if progress_bar is not None:
        progress_range = progress_bar(progress_range)
    for i in progress_range:
        xi = a + i*h
        xip1 = a + (i+1)*h
        res += 1./6*(xip1-xi)*(f(xi)+4*f(0.5*(xi+xip1))+f(xip1))
    return res
