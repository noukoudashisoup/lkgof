# KSD Model Comparison Test

This repository contains a Python 3.6 implementation of the KSD model model test
described in [this paper](https://arxiv.org/abs/1907.00586):

    A Kernel Stein Test for Comparing Latent Variable Models
    Heishiro Kanagawa, Wittawat Jitkrittum, Lester Mackey, Kenji Fukumizu, Arthur Gretton
    https://arxiv.org/abs/1907.00586

## Installation

Use the `pip` command to install the package.
It will be necessary to edit out codes for replicating the experiment results; we therefore recommend to install the package as follows:

1. Clone the repository with  `git clone git@github.com:noukoudashisoup/lkgof.git`
2. In the cloned directory (after `cd` ), run `pip install -e .`

## Requirements

The package requires `numpy`, `scipy`, `autograd`, and `numpyro==0.2.4` (these will be installed when you install the package).
NumPyro is only used for the experiment with PPCA models.

We also require the following packages:

* The `freqopttest` (containing the MMD two-sample test) package
  from  [its git repository](https://github.com/wittawatj/interpretable-test).

* The `kgof` package. This can be obtained from [its git
  repository](https://github.com/wittawatj/kernel-gof).

* The `kmod` package. This can be obtained from [its git
  repository](https://github.com/wittawatj/kernel-gof). This and `freqopttest` are required for the MMD test, a benchmark method in our experiments.

## Reproduce experimental results

All experiments can be found in `lkgof/ex/`; e.g. `ex1_vary_n.py`.
Each python script in the directory is runnable with a command line
argument. For example in
`ex1_vary_n.py`, we aim to check the size or the test power of each testing algorithm
as a function of the sample size `n`. The script `ex1_vary_n.py` takes a
problem name as its argument. See `run_ex1.sh` which is a standalone Bash
script on how to execute  `ex1_power_vs_n.py`.

We used [independent-jobs](https://github.com/wittawatj/independent-jobs)
package to parallelize our experiments over a
[Slurm](http://slurm.schedmd.com/) cluster (the package is not needed if you
just need to use our developed tests). For example, for
`ex1_vary_n.py`, a job is created for each combination of

    (problem label, test algorithm, n, trial, significance level)

If you do not use Slurm, you can change the line

    engine = SlurmComputationEngine(batch_parameters)

to

    engine = SerialComputationEngine()

which will instruct the computation engine to just use a normal for-loop on a
single machine (will take a lot of time). Other computation engines that you
use might be supported. Running simulation will
create a lot of result files (one for each tuple above) saved as Pickle. Also, the `independent-jobs`
package requires a scratch folder to save temporary files for communication
among computing nodes. Path to the folder containing the saved results can be specified in
`lkgof/config.py` by changing the value of `expr_results_path`:

    # Full path to the directory to store experimental results.
    'expr_results_path': '/full/path/to/where/you/want/to/save/results/',

The scratch folder needed by the `independent-jobs` package can be specified in the same file.
To plot the results, see the experiment's corresponding Jupyter notebook in the
`ipynb/` folder. For example, for `ex1_vary_n.py` see
`ipynb/ex1_results.ipynb` to plot the results.