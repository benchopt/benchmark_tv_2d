2-dimensional Total Variation (TV) Benchmark
============================================
|Build Status| |Python 3.6+|

This benchmark is dedicated to solver of TV-2D regularised regression problem:

$$\\boldsymbol{u} \\in \\underset{\\boldsymbol{u} \\in \\mathbb{R}^{n \\times m}}{\\mathrm{argmin}} f(\\boldsymbol{y}, A \\boldsymbol{u}) + g(\\boldsymbol{u})$$


- $\\boldsymbol{y} \\in \\mathbb{R}^{n \\times m}$ is a vector of observations or targets.
- $A \\in \\mathbb{R}^{n \\times n}$ is a design matrix or forward operator.
- $\\lambda > 0$ is a regularization hyperparameter.
- $f(\\boldsymbol{y}, A\\boldsymbol{u}) = \\sum\\limits\_{k=1}^{n} \\sum\\limits\_{l=1}^{m} l(y\_{k,l}, (A\\boldsymbol{u})_{k,l})$ is a loss function, where $l$ can be quadratic loss as $l(y, x) = \\frac{1}{2} \\vert y - x \\vert_2^2$, or Huber loss $l(y, x) = h\_{\\delta} (y - x)$ defined by


$$
h\_{\\delta}(t) = \\begin{cases} \\frac{1}{2} t^2 & \\mathrm{ if } \\vert t \\vert \\le \\delta \\\\ \\delta \\vert t \\vert - \\frac{1}{2} \\delta^2 & \\mathrm{ otherwise} \\end{cases}
$$


- $D_1 \\in \\mathbb{R}^{(n-1) \\times n}$ and $D_2 \\in \\mathbb{R}^{(m-1) \\times m}$ are finite difference operators, such that the regularised TV-2D term $g(\\boldsymbol{u}) = \\lambda \\| \\boldsymbol{u} \\|\_{TV}$ expressed as follows.


In isotropic cases:


$$
g(\\boldsymbol{u}) = \\lambda \\| \\sqrt{ (D\_1 \\boldsymbol{u})^2 + (\\boldsymbol{u} D\_2^{\\top})^2 } \\|\_{1} = \\lambda \\sum\\limits_{k = 1}^{n-1} \\sum\\limits\_{l = 1}^{m-1} \\sqrt{\\vert u_{k+1,l} - u_{k,l} \\vert^2 + \\vert u\_{k,l+1} - u\_{k,l} \\vert^2}
$$


In anisotropic cases:


$$
g(\\boldsymbol{u}) = \\lambda \\| D_1 \\boldsymbol{u} \\|_{1} + \\| \\boldsymbol{u} D_2^{\\top} \\|_{1} = \\lambda \\sum\\limits_{k = 1}^{n-1} \\sum\\limits_{l = 1}^{m-1} (\\vert u_{k+1,l} - u_{k,l} \\vert + \\vert u_{k,l+1} - u_{k,l} \\vert)
$$


where n (or `height`) and m (or `width`) stand for the dimension of targeted vector.


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_tv_2d
   $ benchopt run benchmark_tv_2d

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_tv_2d --config benchmark_tv_2d/example_config.yml


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_tv_2d/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_tv_2d/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
