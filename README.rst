Benchmark repository for Total Variation 2D
===========================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
2D Total Variation regularization consists in solving the following program:

.. math::

    \min_{w} \frac{1}{2} \|y - A w\|^2_2 + J(D w)

where D stands for a finite difference operator, J is either a l1-norm or
l1/l2-norm and A is a linear operator.


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_tv_2d
   $ benchopt run benchmark_tv_2d

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_tv_2d -d deblurring --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Template| image:: https://github.com/benchopt/template_benchmark/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/template_benchmark/actions
.. |Build Status| image:: https://github.com/benchopt/benchmark_tv_2d/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_tv_2d/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
