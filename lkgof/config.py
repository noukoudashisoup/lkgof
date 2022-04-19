
"""
This file defines global configuration of the project.
Casual usage of the package should not need to change this.
"""

import lkgof.glo as glo
import os

# This dictionary of keys is used only when scripts under lkgof/ex/ are
# executed.
expr_configs = {
    # Full path to the directory to store temporary files when running
    # experiments.
    # 'scratch_path': '/nfs/gatsbystor/heishiro/tmp/lkmod',
    'scratch_path': '/ceph/scratch/heishiro/tmp/lkmod',

    # Slurm partitions.
    # When using SlurmComputationEngine for running the experiments, the paritions (groups of computing nodes)
    # can be specified here. Set to None to not set to any value (i.e., use the default partition).
    # The value is a string. For more than one partition, set to, for instance, "wrkstn,compute".
    'slurm_partitions': "cpu,gpu,debug",
    # 'slurm_partitions': None,

    # Full path to the directory to store experimental results.
    'expr_results_path': '/nfs/gatsbystor/heishiro/results/lkmod',


    # Full path to the problems directory
    # A "problems" directory contains subdirectories, each containing all files
    # related to that particular problem e.g., arXiv, etc.
    'problems_path': '/nfs/gatsbystor/heishiro/lkmod/problems',

    # Full path to the data directory
    'data_path': os.path.join(os.path.dirname(glo.get_root()), 'data'),
}
