"""Module containing convenient functions for plotting"""

from builtins import range
from builtins import object
__author__ = 'wittawat'

import lkgof.glo as glo
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import autograd.numpy as np


def get_func_tuples():
    """
    Return a list of tuples where each tuple is of the form
        (func_name used in the experiments, label name, plot line style)
    """
    func_tuples = [
            ('met_gmmd_med', 'MMD (Gauss-med)', 'mo-.',),
            ('met_imqmmd_cov', 'MMD (IMQ-cov)', 'mo--',),
            ('met_imqmmd_med', 'MMD (IMQ-med)', 'mo:',),
            ('met_imqmmd_medtrunc', 'MMD (IMQ-med)', 'mo:',),
            ('met_gksd_med', 'KSD (Gauss-med)', 'r*-.',),
            ('met_imqksd_cov', 'KSD (IMQ-cov)', 'r*--',),
            ('met_imqksd_med', 'KSD (IMQ-med)', 'r*:',),
            ('met_dis_hlksd', 'LKSD', 'gx--',),
            ('met_glksd_med', 'LKSD (Gauss-med)', 'gv-.',),
            ('met_imqlksd_cov', 'LKSD (IMQ-cov)', 'gv--',),
            ('met_imqlksd_med', 'LKSD (IMQ-med)', 'gv:',),
            ('met_imqlksd_medtrunc', 'LKSD (IMQ-med)', 'gv:',),
            ('met_imqlksd_med_balltrunc', 'LKSD (IMQ-med)', 'gv:',),

            ('met_dis_gbowmmd', 'MMD (GBoW)', 'mo-',),
            ('met_dis_imqbowmmd', 'MMD', 'mo:',),
            ('met_dis_imqbowmmd_moremc', 'MMD (IMQBoW) More MC', 'mo--',),
            ('met_dis_imqbowmmd_cheap', 'MMD (IMQBoW) cheap', 'mo-',),
            ('met_dis_gbowlksd', 'KSD (Gauss BoW)', 'bv-.',),
            ('met_dis_ebowlksd', 'KSD (Exp BoW)', 'C1v--',),
            ('met_dis_imqbowlksd', 'LKSD', 'gv:',),
            ('met_dis_imqbow_mflksd', 'LKSD (Alt.)', 'C3P:',),
            ('met_dis_imqbowlksd_moremc', 'LKSD (IMQ BoW) More MC', 'gp--',),
           ]

    func_tuples_mcsize = [
            ('met_imqlksd_med_mc1', 'MC1', 'gv-'),
            ('met_imqlksd_med_mc10', 'MC10', 'g<--'),
            ('met_imqlksd_med_mc100', 'MC100', 'g^-.'),
            ('met_imqlksd_med_mc1000', 'MC1000', 'g>:'),
            ('met_imqlksd_med_mala_mc1', 'MALA MC1', 'C0v-'),
            ('met_imqlksd_med_mala_mc10', 'MALA MC10', 'C0<-.'),
            ('met_imqlksd_med_mala_mc100', 'MALA MC100', 'C0^--'),
            ('met_imqlksd_med_mala_mc1000', 'MALA MC1000', 'C0>:'),
            ('met_dis_imqbowlksd_mc1', 'MC1', 'gv-'),

            ('met_dis_imqbowlksd_mc1', 'MC1', 'gv-'),
            ('met_dis_imqbowlksd_mc10', 'MC10', 'gv--'),
            ('met_dis_imqbowlksd_mc100', 'MC100', 'gv-.'),
            ('met_dis_imqbowlksd_mc1000', 'MC1000', 'gv:'),
            ('met_dis_imqbowlksd_mc10000', 'MC10000', 'g3-'),
    ]

    func_tuples_varests = [
        ('met_imqlksd_med', 'LKSD (Jackknife)', 'gv--',),
        ('met_imqlksd_med_ustatvar', 'LKSD (U-stat)', 'g8-.',),
        ('met_imqlksd_med_vstatvar', 'LKSD (V-stat)', 'gs:',),
        ('met_dis_imqbowlksd', 'LKSD (Jackknife)', 'gv--',),
        ('met_dis_imqbowlksd_ustatvar', 'LKSD (U-stat)', 'g8-.',),
        ('met_dis_imqbowlksd_vstatvar', 'LKSD (V-stat)', 'gs:',),
    ]

    func_tuples_kernelparams = [
        ('met_imqlksd', 'LKSD(IMQ)', 'gv'),
        ('met_glksd', 'LKSD(Gauss)', 'bs'),
        ('met_imqmmd', 'MMD (IMQ)', 'mx'),
        ('met_gmmd', 'MMD(Gauss)', 'ko'),

        ('met_imqbowmmd', 'MMD (IMQ BoW)', 'mx'),
        ('met_gbowmmd', 'MMD (GBoW)', 'ko'),
        ('met_dis_gbowlksd', 'LKSD(Gauss BoW)', 'bs'),
        ('met_dis_imqbowlksd', 'LKSD(IMQ BoW)', 'gv'),
    ]

    # func_tuples += func_tuples_varests
    # func_tuples += func_tuples_mcsize
    # func_tuples += func_tuples_kernelparams
    return func_tuples

def set_default_matplotlib_options():
    # font options
    font = {
    #     'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 30
    }
    matplotlib.rc('font', **{'family': 'serif', })


    # matplotlib.use('cairo')
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('font', **font)
    plt.rc('lines', linewidth=3, markersize=10)
    # matplotlib.rcParams['ps.useafm'] = True
    # matplotlib.rcParams['pdf.use14corefonts'] = True

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

def get_func2label_map():
    # map: job_func_name |-> plot label
    func_tuples = get_func_tuples()
    #M = {k:v for (k,v) in zip(func_names, labels)}
    M = {k:v for (k,v,_) in func_tuples}
    return M


def func_plot_fmt_map():
    """
    Return a map from job function names to matplotlib plot styles 
    """
    # line_styles = ['o-', 'x-',  '*-', '-_', 'D-', 'h-', '+-', 's-', 'v-', 
    #               ',-', '1-']
    func_tuples = get_func_tuples()
    M = {k:v for (k, _, v) in func_tuples}
    return M


class PlotValues(object):
    """
    An object encapsulating values of a plot where there are many curves, 
    each corresponding to one method, with common x-axis values.
    """
    def __init__(self, xvalues, methods, plot_matrix):
        """
        xvalues: 1d numpy array of x-axis values
        methods: a list of method names
        plot_matrix: len(methods) x len(xvalues) 2d numpy array containing 
            values that can be used to plot
        """
        self.xvalues = xvalues
        self.methods = methods
        self.plot_matrix = plot_matrix

    def ascii_table(self, tablefmt="pipe"):
        """
        Return an ASCII string representation of the table.

        tablefmt: "plain", "fancy_grid", "grid", "simple" might be useful.
        """
        methods = self.methods
        xvalues = self.xvalues
        plot_matrix = self.plot_matrix

        import tabulate
        # https://pypi.python.org/pypi/tabulate
        aug_table = np.hstack((np.array(methods)[:, np.newaxis], plot_matrix))
        return tabulate.tabulate(aug_table, xvalues, tablefmt=tablefmt)

# end of class PlotValues

def plot_prob_reject(ex, fname, func_xvalues, xlabel, func_title=None, 
        return_plot_values=False):
    """
    plot the empirical probability that the statistic is above the threshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - xlabel: label of the x-axis. 
    - func_title: a function: results dictionary -> title of the plot
    - return_plot_values: if true, also return a PlotValues as the second
      output value.

    Return loaded results
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    results = glo.ex_load_result(ex, fname)

    def rej_accessor(jr):
        rej = jr['test_result']['h0_rejected']
        # When used with vectorize(), making the value float will make the resulting 
        # numpy array to be of float. nan values can be stored.
        return float(rej)

    def pval_accessor(jr):
        pval = jr['test_result']['pvalue']
        return float(pval)
    
    def stat_accessor(jr):
        stat = jr['test_result']['test_stat']
        return stat

    #value_accessor = lambda job_results: job_results['test_result']['h0_rejected']
    vf_pval = np.vectorize(rej_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    jr = results['job_results']
    if ex == 3 and len(jr.shape) > 3:
        mean = np.mean(vf_pval(jr), axis=1)
        std_rejs = np.std(mean, axis=0)
        lowerpec_rejs = np.percentile(mean, 5, axis=0)
        upperpec_rejs = np.percentile(mean, 95, axis=0)
        jr = jr.reshape((-1,)+jr.shape[2:])

    rejs = vf_pval(jr)
    # print(np.mean(rejs, axis=1)[:, :, 1])
    repeats, _, n_methods = jr.shape

    vf_pval = np.vectorize(pval_accessor)
    pvals = vf_pval(jr)
    # std_pvals = np.std(vf_pval(results['job_results']), axis=0)

    stats = np.vectorize(stat_accessor)(jr)

    # yvalues (corresponding to xvalues) x #methods
    mean_rejs = np.mean(rejs, axis=0)
    #print mean_rejs
    #std_pvals = np.std(rejs, axis=0)
    #std_pvals = np.sqrt(mean_rejs*(1.0-mean_rejs))

    xvalues = func_xvalues(results)

    # ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_funcs'] ]
    plotted_methods = []
    for i in range(n_methods):    
        # te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plotted_methods.append(method_label)
        print(method_label, mean_rejs[:, i], np.mean(stats[:, :, i], axis=0), np.std(stats[:, :, i], axis=0))
        # print('pval', method_label, (pvals[:, :, i]))
        print('pval', method_label, np.mean(pvals[:, :, i], axis=0))
        plt.plot(xvalues, mean_rejs[:, i], fmt, label=method_label, fillstyle='none')
        if ex == 3:
            y = mean_rejs[:, i]
            yerr = std_rejs[:, i]
            # yerr = np.vstack([y-lowerpec_rejs[:, i], upperpec_rejs[:, i]-y])
            # print('stds', yerr, method_label)
            plt.errorbar(xvalues, y, yerr=yerr, fmt=fmt, label=method_label, fillstyle='none', alpha=.5)
        else:
            plt.plot(xvalues, mean_rejs[:, i], fmt, label=method_label, fillstyle='none')
    '''
    else:
        # h0 is true 
        z = stats.norm.isf( (1-confidence)/2.0)
        for i in range(n_methods):
            phat = mean_rejs[:, i]
            conf_iv = z*(phat*(1-phat)/repeats)**0.5
            #plt.errorbar(test_sizes, phat, conf_iv, fmt=line_styles[i], label=method_labels[i])
            plt.plot(test_sizes, mean_rejs[:, i], line_styles[i], label=method_labels[i])
    '''
            
    ylabel = 'Rejection rate'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(np.hstack((xvalues) ))
    
    alpha = results['alpha']
    plt.legend(loc='best')
    title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
            repeats, alpha) if func_title is None else func_title(results)
    plt.title(title)
    plt.grid()
    if return_plot_values:
        return results, PlotValues(xvalues=xvalues, methods=plotted_methods,
                plot_matrix=mean_rejs.T)
    else:
        return results
        
def plot_prob_reject_with_params(ex, fname, func_xvalues, func_parvalues, xlabel, 
                                 paramname, 
                                 func_title=None,
                                 return_plot_values=False):
    """
    Plot the empirical probability that the statistic is above the threshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for variations of a method
    - xlabel: label of the x-axis. 
    - func_title: a function: results dictionary -> title of the plot
    - return_plot_values: if true, also return a PlotValues as the second
      output value.

    Return loaded results
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    results = glo.ex_load_result(ex, fname)

    def rej_accessor(jr):
        rej = jr['test_result']['h0_rejected']
        # When used with vectorize(), making the value float will make the resulting 
        # numpy array to be of float. nan values can be stored.
        return float(rej)

    vf_pval = np.vectorize(rej_accessor)
    
    jr = results['job_results']
    rejs = vf_pval(jr)
    repeats, _, _ , n_methods = jr.shape

    mean_rejs = np.mean(rejs, axis=0)
    xvalues = func_xvalues(results)
    parvalues = func_parvalues(results)
    n_params = len(parvalues)

    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_funcs'] ]
    plotted_methods = []
    for i in range(n_methods):    
        for j in range(n_params):
            linestyle = (0, (2+j, 1+j, i*2, i*2))
            # te_proportion = 1.0 - results['tr_proportion']
            fmt = line_styles[func_names[i]]
            #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
            param = parvalues[j]
            method_label = method_labels[func_names[i]]
            method_label = '{label} {name}={param}'.format(label=method_label,
                                                    name=paramname, param=param)
            plotted_methods.append(method_label)
            print(method_label, mean_rejs[:, j, i])
            #print(method_label, mean_rejs[:, i], np.mean(stats[:, :, i], axis=0), np.std(stats[:, :, i], axis=0))
            # print('pval', method_label, (pvals[:, :, i]))
            # print('pval', method_label, np.mean(pvals[:, :, i], axis=0))
            plt.plot(xvalues, mean_rejs[:, j, i], fmt, label=method_label,
                     linestyle=linestyle, fillstyle='none')
    '''
    else:
        # h0 is true 
        z = stats.norm.isf( (1-confidence)/2.0)
        for i in range(n_methods):
            phat = mean_rejs[:, i]
            conf_iv = z*(phat*(1-phat)/repeats)**0.5
            #plt.errorbar(test_sizes, phat, conf_iv, fmt=line_styles[i], label=method_labels[i])
            plt.plot(test_sizes, mean_rejs[:, i], line_styles[i], label=method_labels[i])
    '''
            
    ylabel = 'Rejection rate'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(np.hstack((xvalues) ))
    
    alpha = results['alpha']
    plt.legend(loc='best')
    title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
            repeats, alpha) if func_title is None else func_title(results)
    plt.title(title)
    plt.grid()
    if return_plot_values:
        return results, PlotValues(xvalues=xvalues, methods=plotted_methods,
                plot_matrix=mean_rejs.T)
    else:
        return results


def plot_runtime(ex, fname, func_xvalues, xlabel, func_title=None):
    results = glo.ex_load_result(ex, fname)
    value_accessor = lambda job_results: job_results['time_secs']
    vf_pval = np.vectorize(value_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    jr = results['job_results']
    if ex == 3 and len(jr.shape) > 3:
        jr = jr[0]

    times = vf_pval(jr)
    repeats, _, n_methods = jr.shape
    time_avg = np.mean(times, axis=0)
    time_std = np.std(times, axis=0)

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_funcs'] ]
    for i in range(n_methods):    
        # te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        print(time_avg[:, i])
        plt.errorbar(xvalues, time_avg[:, i], yerr=time_std[:,i], fmt=fmt,
                label=method_label)
            
    ylabel = 'Time (s)'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim([np.min(xvalues), np.max(xvalues)])
    plt.xticks( xvalues, xvalues )
    plt.legend(loc='best')
    plt.gca().set_yscale('log')
    title = '%s. %d trials. '%( results['prob_label'],
            repeats ) if func_title is None else func_title(results)
    plt.title(title)
    #plt.grid()
    return results


def box_meshgrid(func, xbound, ybound, nx=50, ny=50):
    """
    Form a meshed grid (to be used with a contour plot) on a box
    specified by xbound, ybound. Evaluate the grid with [func]: (n x 2) -> n.
    
    - xbound: a tuple (xmin, xmax)
    - ybound: a tuple (ymin, ymax)
    - nx: number of points to evluate in the x direction
    
    return XX, YY, ZZ where XX is a 2D nd-array of size nx x ny
    """
    
    # form a test location grid to try 
    minx, maxx = xbound
    miny, maxy = ybound
    loc0_cands = np.linspace(minx, maxx, nx)
    loc1_cands = np.linspace(miny, maxy, ny)
    lloc0, lloc1 = np.meshgrid(loc0_cands, loc1_cands)
    # nd1 x nd0 x 2
    loc3d = np.dstack((lloc0, lloc1))
    # #candidates x 2
    all_loc2s = np.reshape(loc3d, (-1, 2) )
    # evaluate the function
    func_grid = func(all_loc2s)
    func_grid = np.reshape(func_grid, (ny, nx))
    
    assert lloc0.shape[0] == ny
    assert lloc0.shape[1] == nx
    assert np.all(lloc0.shape == lloc1.shape)
    
    return lloc0, lloc1, func_grid

def get_density_cmap():
    """
    Return a colormap for plotting the model density p.
    Red = high density 
    white = very low density.
    Varying from white (low) to red (high).
    """
    # Add completely white color to Reds colormap in Matplotlib
    list_colors = plt.cm.datad['Reds']
    list_colors = list(list_colors)
    list_colors.insert(0, (1, 1, 1))
    list_colors.insert(0, (1, 1, 1))
    lscm = matplotlib.colors.LinearSegmentedColormap.from_list("my_Reds", list_colors)
    return lscm


def plot_prob_reject_heatmap(ex, fname, func_xvalues, func_yvalues, xlabel, ylabel, 
                             xscale='linear', yscale='linear',
                             func_title=None, return_plot_values=False):
    """
    plot the empirical probability that the statistic is above the threshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - func_yvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - xlabel: label of the x-axis. 
    - ylabel: label of the y-axis. 
    - func_title: a function: results dictionary -> title of the plot
    - return_plot_values: if true, also return a PlotValues as the second
      output value.

    Return loaded results
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    results = glo.ex_load_result(ex, fname)

    def rej_accessor(jr):
        rej = jr['test_result']['h0_rejected']
        # When used with vectorize(), making the value float will make the resulting 
        # numpy array to be of float. nan values can be stored.
        return float(rej)

    def pval_accessor(jr):
        pval = jr['test_result']['pvalue']
        return float(pval)
    
    def stat_accessor(jr):
        stat = jr['test_result']['test_stat']
        return stat

    #value_accessor = lambda job_results: job_results['test_result']['h0_rejected']
    vf_pval = np.vectorize(rej_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    jr = results['job_results']
    rejs = vf_pval(jr)
    repeats, _, _, n_methods = jr.shape

    vf_pval = np.vectorize(pval_accessor)
    pvals = vf_pval(jr)
    # std_pvals = np.std(vf_pval(results['job_results']), axis=0)

    stats = np.vectorize(stat_accessor)(jr)

    # yvalues (corresponding to xvalues) x #methods
    mean_rejs = np.mean(rejs, axis=0)
    #print mean_rejs
    #std_pvals = np.std(rejs, axis=0)
    #std_pvals = np.sqrt(mean_rejs*(1.0-mean_rejs))

    xvalues = func_xvalues(results)
    yvalues = func_yvalues(results)

    # ns = np.array(results[xkey])
    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_funcs'] ]
    plotted_methods = []

    fig = plt.figure(figsize=(10, 10))

    grid = AxesGrid(fig, (1, 1, 1),
                nrows_ncols=(1, n_methods),
                axes_pad=1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.2
                )

    for i, ax in enumerate(grid):
        # fmt = line_styles[func_names[i]]
        method_label = method_labels[func_names[i]]
        plotted_methods.append(method_label)
        extent = (min(xvalues), max(xvalues), min(yvalues), max(yvalues))
        im = ax.imshow(mean_rejs[:, :, i], vmin=0, vmax=1., origin='lower', extent=extent, cmap='hot_r')
        ax.set_title(method_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    cbar = ax.cax.colorbar(im)
    # fig.subplots_adjust(right=0.8)
    # fig.colorbar(pos, ax=ax)
    # alpha = results['alpha']
    # plt.legend(loc='best')
    # title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
    #         repeats, alpha) if func_title is None else func_title(results)
    # plt.title(title)
    # plt.grid()
    # if return_plot_values:
    #     return results, PlotValues(xvalues=xvalues, methods=plotted_methods,
    #             plot_matrix=mean_rejs.T)
    # else:
    #     return results
