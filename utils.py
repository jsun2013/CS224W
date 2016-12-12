from __future__ import print_function
import pandas as pd
import snap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import mpmath
import powerlaw
from collections import defaultdict
import os

def create_net_file(df, fname, min_year, max_year):
    players = set()
    edges = defaultdict(int)
    for _, row in df[(df['year'] <= max_year) & (df['year'] >= min_year)].iterrows():
        if len(row['loser_name']) > 0 and len(row['winner_name']) > 0:
            players.add(row['loser_name'])
            players.add(row['winner_name'])
            edges[(row['loser_name'], row['winner_name'])] += 1
    players = list(players)
    id2names = {}
    names2id = {}
    for i, name in enumerate(players):
        id2names[i + 1] = name
        names2id[name] = i + 1
    if not os.path.isfile(fname+".net"):
        with open(fname+".net", "w+") as f:
            f.write("*Vertices {}\n".format(len(players)))
            for name in players:
                f.write("{} \"{}\" 1.0\n".format(names2id[name], name))
            f.write("*Arcs {}\n".format(len(edges.keys())))
            for e in edges:
                name1, name2 = e
                id1 = names2id[name1]
                id2 = names2id[name2]
                f.write("{} {} {}\n".format(id1, id2, edges[e]))
        return True
    return False




def read_alt_files(fglob):
    temp_list = []
    for f in fglob:
        temp_df = pd.read_csv(f, index_col=False, header=0)
        temp_list.append(temp_df)
    df = pd.concat(temp_list, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d")
    df['year'] = df['tourney_date'].dt.year
    return df


def get_graph(df):
    G = snap.TUNGraph.New()
    ids = set()
    for _, row in df.iterrows():
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        ids.add(winner_id)
        ids.add(loser_id)
        if not(G.IsNode(winner_id)):
            G.AddNode(winner_id)
        if not G.IsNode(loser_id):
            G.AddNode(loser_id)
        if not G.IsEdge(loser_id, winner_id):
            G.AddEdge(loser_id, winner_id)
    return G


def get_dists(G):
    deg_counts = []
    degs = []
    deg_vect = snap.TIntPrV()
    snap.GetDegCnt(G, deg_vect)
    for item in deg_vect:
        deg = item.GetVal1()
        cnt = item.GetVal2()
        deg_counts.append(cnt)
        degs.append(deg)

    out_deg = []
    out_counts = []
    cur_deg = min(degs)
    for deg, cnt in zip(degs, deg_counts):
        # while cur_deg < deg:
        #     out_deg.append(cur_deg)
        #     out_counts.append(0)
        #     cur_deg += 1
        out_deg.append(deg)
        out_counts.append(cnt)
        cur_deg += 1

    deg_counts = np.asarray(out_counts)
    degs = np.asarray(out_deg)
    pdf = deg_counts.astype(float) / sum(deg_counts)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)
    ccdf = 1 - cdf
    return deg_counts, degs, cdf, ccdf, pdf


def get_deg_data(G):
    result_degree = snap.TIntV()
    snap.GetDegSeqV(G, result_degree)
    deg_data = []
    for i in range(result_degree.Len()):
        deg_data.append(result_degree[i])
    return deg_data

def plot_both(beta, alpha, xmin, Lambda_a, Lambda_b, pdf, ccdf, degs, title=None):
    C = beta * Lambda_b * np.exp(Lambda_b * xmin ** beta)
    pdf_hat = degs[degs >= xmin] ** (beta - 1) * np.exp(-Lambda_b * degs[degs >= xmin] ** beta) * C
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.scatter(degs, pdf, marker='o', label='True', color=sns.xkcd_rgb["deep blue"])
    ax.plot(degs[degs >= xmin], pdf_hat, marker='*', label='Stretched', color=sns.xkcd_rgb["pale red"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Degree')

    C = Lambda_a ** (1 - alpha) / (mpmath.gammainc(1 - alpha, Lambda_a * xmin))
    pdf_hat = degs[degs >= xmin] ** -alpha * np.exp(-Lambda_a * degs[degs >= xmin]) * C
    ax.plot(degs[degs >= xmin], pdf_hat, marker='D', label='Truncated', color=sns.xkcd_rgb["medium green"])
    if title:
        ax.set_title(title)
    ax.legend()
    plt.show()

def plot_stretched(beta, Lambda, xmin, pdf, ccdf, degs, title=None, ax=None):
    C = beta*Lambda *np.exp(Lambda*xmin**beta)
    pdf_hat = degs[degs>=xmin] **(beta-1) * np.exp(-Lambda * degs[degs>=xmin]**beta) * C
    ccdf_hat = 1 - np.cumsum(pdf_hat)

    fig = plt.figure(figsize=(12, 9))
    if not ax:
        ax = fig.add_subplot(111)
    else:
        fig.axes.append(ax)
    ax.scatter(degs, pdf, marker='o', label='True', color=sns.xkcd_rgb["deep blue"])
    ax.plot(degs[degs>=xmin], pdf_hat, marker='*', label='Fit', color=sns.xkcd_rgb["pale red"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Degree')
    if title:
        ax.set_title(title)
    ax.legend()

    plot_text_x = 10**(np.mean(np.log10(ax.get_xlim())))
    plot_text_y = plot_text_x ** (beta-1) * np.exp(-Lambda * plot_text_x) * C
    ax.annotate(r'$\beta=%.2f$, $\lambda=%.2g$, ' % (beta, Lambda), xy=(plot_text_x, plot_text_y), color=sns.xkcd_rgb["pale red"]
                )

    plt.show()
    return ax

def plot_fit(alpha, Lambda, xmin, pdf, ccdf, degs, title=None, ax=None):
    C = Lambda ** (1 - alpha) / (mpmath.gammainc(1 - alpha, Lambda * xmin))
    pdf_hat = degs[degs>=xmin] ** -alpha * np.exp(-Lambda * degs[degs>=xmin]) * C
    ccdf_hat = 1 - np.cumsum(pdf_hat)
    if not ax:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
    ax.scatter(degs, pdf, marker='o', label='True', color=sns.xkcd_rgb["deep blue"])
    ax.plot(degs[degs>=xmin], pdf_hat, marker='*', label='Fit', color=sns.xkcd_rgb["pale red"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Degree')
    if title:
        ax.set_title(title)
    ax.legend()

    plot_text_x = 10**(np.mean(np.log10(ax.get_xlim())))
    plot_text_y = plot_text_x ** -alpha * np.exp(-Lambda * plot_text_x) * C
    ax.annotate(r'$\alpha=%.2f$, $\lambda=%.2g$, ' % (alpha, Lambda), xy=(plot_text_x, plot_text_y), color=sns.xkcd_rgb["pale red"]
                )

    plt.show()


def print_fit_results(data, discrete=False):
    results = powerlaw.Fit(data,xmin=2, discrete=discrete)
    t = PrettyTable(['', 'power', 'lognormal', 'exponential', 'power with cutoff', 'stretched exponential'])
    print("alpha: {}".format(results.power_law.alpha))
    print("x_min: {}".format(results.power_law.xmin))
    print("Number of Data Points: {}".format(len(data)))

    dists = ['power_law', 'lognormal_positive', 'exponential', 'truncated_power_law', 'stretched_exponential']

    for i in range(len(dists)):
        dist = dists[i]
        row = [dist]
        for j in range(len(dists)):
            R, p = results.distribution_compare(dist, dists[j])
            row.append('{:.2f},{:.3f}'.format(R, p))
        t.add_row(row)
    print(t)

def print_fit_results_tex(data, discrete=True, xmin=2):
    results = powerlaw.Fit(data, discrete=discrete,xmin=None)
    print("alpha: {}".format(results.power_law.alpha))
    print("x_min: {}".format(results.power_law.xmin))
    print("Number of Data Points: {}".format(len(data)))

    dists = ['power_law', 'lognormal_positive', 'exponential', 'truncated_power_law', 'stretched_exponential']
    names = ['Power', 'Log-Normal', 'Exp', 'Cutoff', 'Stretched']

    print()
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|c||c|c||c|c||c|c||c|c||c|c|}")
    print("\\hline")
    title = ""
    for name in names:
        title += " & \\multicolumn{2}{c||}{" + name + "}"
    title += "\\\\"
    print(title)
    print("\\cline{2-11}")
    print(" & LR & p-val"*5 + "\\\\\\hline" )

    for i, name in enumerate(names):
        dist = dists[i]
        row = name
        for j in range(len(dists)):
            R, p = results.distribution_compare(dist, dists[j])
            row += "&"
            if i == j:
                row += "-&-"
            else:
                row += '{:.2f}&{:.2f}'.format(R, p)
        row += "\\\\\\hline"
        print(row)
    print("\\end{tabular}")

def fit_stretched(data, discrete=True, calc_p=False, manual_xmin=None):
    results = powerlaw.Fit(data, discrete=discrete)
    if manual_xmin:
        xmin = manual_xmin
    else:
        xmin = results.xmin

    dist = powerlaw.Stretched_Exponential(xmin=xmin, discrete=discrete)
    dist.fit(data)
    empirical_ks = dist.KS(data)
    beta = dist.beta
    Lambda = dist.Lambda
    p_val = None

    if calc_p:
        num_synthetic = 1000
        n = len(data)
        n_tail = sum(data >= xmin)
        p_tail = float(n_tail) / n
        p_count = 0.0
        for _ in range(num_synthetic):
            test_set = []
            for jjj in range(n):
                if np.random.rand() > p_tail:
                    x = np.random.choice(np.arange(0, xmin))
                    test_set.append(x)
                else:
                    while True:
                        r = np.random.rand()
                        x = np.floor((xmin - 0.5) * (1 - r) ** (-1 / (alpha - 1)) + 0.5)
                        p = (x / xmin) ** (-alpha)
                        if np.random.rand() <= p:
                            test_set.append(x)
                            break
            test_set = np.asarray(test_set)
            test_dist = powerlaw.Truncated_Power_Law(xmin=xmin, discrete=discrete)
            test_dist.fit(test_set)
            if test_dist.KS(test_set) > empirical_ks:
                p_count += 1
        p_val = p_count / 1000

    return dist.beta, dist.Lambda, results.xmin, p_val

def fit_truncated(data, discrete=True, calc_p=False, manual_xmin=None):
    results = powerlaw.Fit(data, discrete=discrete)
    if manual_xmin:
        xmin = manual_xmin
    else:
        xmin = results.xmin

    dist = powerlaw.Truncated_Power_Law(xmin=xmin, discrete=discrete)
    dist.fit(data)
    empirical_ks = dist.KS(data)
    alpha = dist.alpha
    Lambda = dist.Lambda
    p_val = None

    if calc_p:
        num_synthetic = 1000
        n = len(data)
        n_tail = sum(data >= xmin)
        p_tail = float(n_tail) / n
        p_count = 0.0
        for _ in range(num_synthetic):
            test_set = []
            for jjj in range(n):
                if np.random.rand() > p_tail:
                    x = np.random.choice(np.arange(0, xmin))
                    test_set.append(x)
                else:
                    while True:
                        r = np.random.rand()
                        x = np.floor((xmin - 0.5) * (1 - r) ** (-1 / (alpha - 1)) + 0.5)
                        p = (x / xmin) ** (-alpha)
                        if np.random.rand() <= p:
                            test_set.append(x)
                            break
            test_set = np.asarray(test_set)
            test_dist = powerlaw.Truncated_Power_Law(xmin=xmin, discrete=discrete)
            test_dist.fit(test_set)
            if test_dist.KS(test_set) > empirical_ks:
                p_count += 1
        p_val = p_count / 1000

    return dist.alpha, dist.Lambda, results.xmin, p_val


def add_df_to_G(df, G, directed=False):
    for _, row in df.iterrows():
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        if not(G.IsNode(winner_id)):
            G.AddNode(winner_id)
        if not G.IsNode(loser_id):
            G.AddNode(loser_id)
        if not G.IsEdge(loser_id, winner_id) or directed:
            G.AddEdge(loser_id, winner_id)


def get_fit_params_by_year(df):
    years = sorted(df['year'].unique())
    out_alphas = []
    out_lambdas = []
    out_xmins = []
    out_years = []
    for year in years:
        G = get_graph(df[df['year']==year])
        deg_data = get_deg_data(G)
        try:
            alpha, Lambda, xmin, _ = fit_truncated(deg_data)
            out_alphas.append(alpha)
            out_lambdas.append(Lambda)
            out_xmins.append(xmin)
            out_years.append(year)
        except Exception as err:
            print("Year {} fitting failed with {} exception".format(year, type(err).__name__))
    return out_alphas, out_lambdas, out_xmins, out_years


def cumulative_get_fit_params_by_year(df):
    years = sorted(df['year'].unique())
    out_alphas = []
    out_lambdas = []
    out_xmins = []
    out_years = []
    G = snap.TUNGraph.New()
    for year in years:
        add_df_to_G(df[df['year']==year], G)
        deg_data = get_deg_data(G)
        try:
            alpha, Lambda, xmin, _ = fit_truncated(deg_data)
            out_alphas.append(alpha)
            out_lambdas.append(Lambda)
            out_xmins.append(xmin)
            out_years.append(year)
        except Exception as err:
            print("Year {} fitting failed with {} exception".format(year, type(err).__name__))
    return out_alphas, out_lambdas, out_xmins, out_years


def get_densification(df):
    years = sorted(df['year'].unique())
    out_num_nodes = []
    out_num_edges = []
#     out_bfs_diameters = []
    out_anf_diameters = []
    for year in years:
        G = get_graph(df[df['year']==year])
        out_num_nodes.append(G.GetNodes())
        out_num_edges.append(G.GetEdges())
        scc = snap.GetMxScc(G)
        out_anf_diameters.append(snap.GetAnfEffDiam(scc))
    return out_num_nodes, out_num_edges, out_anf_diameters, years


def cumulative_get_densification(df):
    years = sorted(df['year'].unique())
    out_num_nodes = []
    out_num_edges = []
    out_anf_diameters = []
    G = snap.TUNGraph.New()
    for year in years:
        add_df_to_G(df[df['year']==year], G)
        out_num_nodes.append(G.GetNodes())
        out_num_edges.append(G.GetEdges())
        out_anf_diameters.append(snap.GetAnfEffDiam(G))
    return out_num_nodes, out_num_edges, out_anf_diameters, years

def directed_get_densification(df):
    years = sorted(df['year'].unique())
    out_num_nodes = []
    out_num_edges = []
    out_anf_diameters = []
    G = snap.TNEANet.New()
    for year in years:
        add_df_to_G(df[df['year']==year], G, directed=True)
        out_num_nodes.append(G.GetNodes())
        out_num_edges.append(G.GetEdges())
        out_anf_diameters.append(snap.GetAnfEffDiam(G))
    return out_num_nodes, out_num_edges, out_anf_diameters, years


def get_deg_info(G, lut_df):
    df = pd.DataFrame()
    ids = []
    degs = []
    for node in G.Nodes():
        nid = node.GetId()
        deg = node.GetDeg()
        ids.append(nid)
        degs.append(deg)
    df['id'] = ids
    df['deg'] = degs
    return df.join(lut_df.set_index('id'), on="id")

def get_directed_graph(df):
    G = snap.TNEANet.New()
    ids = set()
    for _, row in df.iterrows():
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        ids.add(winner_id)
        ids.add(loser_id)
        if not(G.IsNode(winner_id)):
            G.AddNode(winner_id)
        if not G.IsNode(loser_id):
            G.AddNode(loser_id)
        G.AddEdge(loser_id, winner_id)
    return G

def get_in_dists(G):
    deg_counts = []
    degs = []
    deg_vect = snap.TIntPrV()
    snap.GetInDegCnt(G, deg_vect)
    for item in deg_vect:
        deg = item.GetVal1()
        cnt = item.GetVal2()
        deg_counts.append(cnt)
        degs.append(deg)

    out_deg = []
    out_counts = []
    cur_deg = min(degs)
    for deg, cnt in zip(degs, deg_counts):
        # while cur_deg < deg:
        #     out_deg.append(cur_deg)
        #     out_counts.append(0)
        #     cur_deg += 1
        out_deg.append(deg)
        out_counts.append(cnt)
        cur_deg += 1

    deg_counts = np.asarray(out_counts)
    degs = np.asarray(out_deg)
    pdf = deg_counts.astype(float) / sum(deg_counts)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)
    ccdf = 1 - cdf
    return deg_counts, degs, cdf, ccdf, pdf


def get_in_deg_data(G):
    deg_data = []
    for node in G.Nodes():
        deg_data.append(node.GetInDeg())
    return deg_data

def get_out_dists(G):
    deg_counts = []
    degs = []
    deg_vect = snap.TIntPrV()
    snap.GetOutDegCnt(G, deg_vect)
    for item in deg_vect:
        deg = item.GetVal1()
        cnt = item.GetVal2()
        deg_counts.append(cnt)
        degs.append(deg)

    out_deg = []
    out_counts = []
    cur_deg = min(degs)
    for deg, cnt in zip(degs, deg_counts):
        # while cur_deg < deg:
        #     out_deg.append(cur_deg)
        #     out_counts.append(0)
        #     cur_deg += 1
        out_deg.append(deg)
        out_counts.append(cnt)
        cur_deg += 1

    deg_counts = np.asarray(out_counts)
    degs = np.asarray(out_deg)
    pdf = deg_counts.astype(float) / sum(deg_counts)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)
    ccdf = 1 - cdf
    return deg_counts, degs, cdf, ccdf, pdf


def get_out_deg_data(G):
    deg_data = []
    for node in G.Nodes():
        deg_data.append(node.GetOutDeg())
    return deg_data

def get_page_ranks(df, G, names_df):
    PRankH = snap.TIntFltH()
    snap.GetPageRank(G, PRankH)
    ranks = []
    for item in PRankH:
        name = names_df[names_df["id"] == item]["name"].values[0]
        ranks.append( (name, PRankH[item]) )
    return sorted(ranks, key=lambda x : x[1], reverse=True)

def cumulative_year_page_rank(df, id2names):
    years = sorted(df['year'].unique())
    page_ranks = {}
    G = snap.TNEANet.New()
    for year in years:
        cur_ranks = {}
        add_df_to_G(df[df['year'] == year], G, directed=True)
        PRankH = snap.TIntFltH()
        snap.GetPageRank(G, PRankH)
        for id in PRankH:
            cur_ranks[id2names[id]] = PRankH[id]
        page_ranks[year] = cur_ranks

    return page_ranks

def single_year_page_rank(df, id2names):
    years = sorted(df['year'].unique())
    page_ranks = {}
    for year in years:
        G = snap.TNEANet.New()
        cur_ranks = {}
        add_df_to_G(df[df['year'] == year], G, directed=True)
        PRankH = snap.TIntFltH()
        snap.GetPageRank(G, PRankH)
        for id in PRankH:
            cur_ranks[id2names[id]] = PRankH[id]
        page_ranks[year] = cur_ranks
    return page_ranks


def get_player_ranks(name, names2id, ranks_by_year):
    year = min(ranks_by_year.keys())
    ranks = []
    years = []
    while year in ranks_by_year:
        if name in ranks_by_year[year]:
            ranks.append(ranks_by_year[year][name])
            years.append(year)
        year += 1
    return ranks, years






