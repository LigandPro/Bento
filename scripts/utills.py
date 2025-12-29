from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import networkx as nx

import pandas as pd
import numpy as np

import plotly.graph_objects as go

def getMetrics(df, indices):
    result = {}
    indices = [i for i in indices if i in df.index]
    sub = df.loc[indices]

    for col in sub.columns:
        values = sub[col]

        # Extract conditions
        rmsd_le_2 = values.apply(lambda x: x[0] <= 2)
        is_true = values.apply(lambda x: x[1] is True)

        # Compute portions
        portion_rmsd = rmsd_le_2.mean()  # fraction True
        portion_rmsd_and_true = (rmsd_le_2 & is_true).mean()

        result[col] = (portion_rmsd, portion_rmsd_and_true)

    return result

def group_tools_pairwise_cliques(df, subset_ids, alpha=0.05, method='holm'):

    subset_ids = [i for i in subset_ids if i in df.index]
    sub = df.loc[subset_ids]
    tool_list = list(df.columns)
    m = len(tool_list)
    N = len(subset_ids)  


    successes = np.array([sub[tool].apply(lambda x: bool(x[1])).sum() for tool in tool_list], dtype=int)
    proportions = {tool_list[i]: successes[i] / float(N) for i in range(m)}
    raw_p = np.ones((m, m))
    for i in range(m):
        for j in range(i+1, m):
            count = np.array([successes[i], successes[j]])
            nobs = np.array([N, N])
            # If N is zero somehow, p=1.0
            if N == 0:
                p = 1.0
            else:
                stat, p = proportions_ztest(count, nobs)
            raw_p[i, j] = p

    tri_idx = np.triu_indices(m, k=1)
    raw_p_flat = raw_p[tri_idx]

    if raw_p_flat.size == 0:
        return [[t] for t in tool_list], proportions, {'raw_p': raw_p, 'p_corrected': raw_p}

    reject, p_corrected_flat, _, _ = multipletests(raw_p_flat, alpha=alpha, method=method)

    p_corrected = np.ones((m, m))
    p_corrected[tri_idx] = p_corrected_flat
    i_upper, j_upper = tri_idx
    p_corrected[j_upper, i_upper] = p_corrected_flat

    G = nx.Graph()
    G.add_nodes_from(tool_list)
    for i in range(m):
        for j in range(i+1, m):
            if p_corrected[i, j] >= alpha:
                G.add_edge(tool_list[i], tool_list[j])

    groups = []
    remaining_nodes = set(tool_list)

    while remaining_nodes:
        H = G.subgraph(remaining_nodes)
        cliques = list(nx.find_cliques(H))
        if not cliques:
            for node in sorted(remaining_nodes):
                groups.append([node])
            break

        def clique_score(clq):
            size = len(clq)
            mean_prop = np.mean([proportions[n] for n in clq])
            return (size, mean_prop)

        cliques.sort(key=clique_score, reverse=True)
        chosen = cliques[0]
        groups.append(sorted(chosen, key=lambda t: -proportions[t])) 
        remaining_nodes -= set(chosen)

    groups.sort(key=lambda g: -np.mean([proportions[t] for t in g]))

    details = {'raw_p': raw_p, 'p_corrected': p_corrected, 'N': N, 'tool_list': tool_list, 'successes': successes}
    return groups, proportions, details

def getStatGroups(df, subset_uids, all_tools = False):
    if all_tools == False:
        df = df.loc[:, ['matcha_fromTrue_fast_40', 'diffdock', 
       'gnina_ligand_box', 'vina_ligand_box', 'smina_ligand_box', 'af3',
       'chai', 'boltz_pocket_10A']]
    groups, props, details = group_tools_pairwise_cliques(df, subset_uids)
    return groups

def plot_grouped_bars(data, title, w=1500, h=500):

    # Tool labels converted to LaTeX text
    tool_labels = {
    'matcha_fromTrue_fast_40': 'Matcha',
    'diffdock': 'DiffDock',
    'unimol_p2rank' : 'Uni-Mol (P2Rank)', #Docking V2 (P2Rank)',
    'gnina_ligand_box' : 'Gnina',
    'vina_ligand_box' : 'Vina', 
    'smina_ligand_box' : 'smina', 
    'af3' : 'AlphaFold3',
    'chai' : 'Chai-1',
    'neuralplexer' : 'NeuralPLexer',
    'FD': 'FlowDock',
    'boltz_pocket_10A' : "Boltz-2"
    }

    tol = 0.01
    subsets = list(data.keys())

    x_vals, first_vals, second_vals, x_tick_labels, subset_for_bar, tool_for_bar = [], [], [], [], [], []
    bar_index = 0

    # Flatten data
    for subset in subsets:
        subset_dict, subset_groups = data[subset]
        for subgroup in subset_groups:
            tools_in_data = [t for t in subgroup if t in subset_dict]
            for tool in tools_in_data:
                f_val, s_val = subset_dict[tool]
                x_vals.append(bar_index)
                first_vals.append(f_val)
                second_vals.append(s_val)
                # Tick labels cannot render MathJax → store LaTeX-style literal
                x_tick_labels.append(tool_labels.get(tool, tool))
                subset_for_bar.append(subset)
                tool_for_bar.append(tool)
                bar_index += 1
            if tools_in_data:
                bar_index += 1
        bar_index += 1

    max_y = max(max(first_vals, default=0), max(second_vals, default=0))
    fig = go.Figure()

    # Best tools selection
    best_first, best_second = {}, {}
    for subset in subsets:
        subset_dict, subset_groups = data[subset]
        subset_tools = [t for subgroup in subset_groups for t in subgroup if t in subset_dict]
        if subset_tools:
            max_first_val = max(subset_dict[t][0] for t in subset_tools)
            max_second_val = max(subset_dict[t][1] for t in subset_tools)
            best_first[subset] = {t for t in subset_tools if abs(subset_dict[t][0] - max_first_val) <= tol}
            best_second[subset] = {t for t in subset_tools if abs(subset_dict[t][1] - max_second_val) <= tol}

    # Bars
    for x, f_val, s_val, subset, tool in zip(x_vals, first_vals, second_vals,
                                             subset_for_bar, tool_for_bar):
        base_color = "#8ba8b7"
        hatched_color = "#9600ff" if tool in best_first.get(subset, set()) else base_color
        filled_color = "#9600ff" if tool in best_second.get(subset, set()) else base_color

        fig.add_trace(go.Bar(
            x=[x], y=[f_val],
            marker=dict(color="rgba(0,0,0,0)",line=dict(color=hatched_color, width=1.5),pattern=dict(shape="/", fgcolor=hatched_color)),showlegend=False))
        fig.add_trace(go.Bar(x=[x], y=[s_val], marker=dict(color=filled_color), showlegend=False))

    #  LaTeX FRACTION LABELS
    for x, f_val, s_val in zip(x_vals, first_vals, second_vals):
        f_pct = int(round(f_val * 100))
        s_pct = int(round(s_val * 100))

        top_height = max(f_val, s_val)
        y_pos = top_height + max_y * 0.02

        frac = f"{f_pct}<br>{s_pct}"
        fig.add_annotation(x=x, y=y_pos, text=frac, xref="x", yref="y", showarrow=False, font=dict(size=12, color='grey'), xanchor="center", yanchor="bottom")


    # Legend (LaTeX text)
    fig.add_trace(go.Bar(x=[0], y=[0],
                         marker=dict(color='white',
                                     pattern=dict(shape='/', fgcolor='#8ba8b7'),
                                     line=dict(color='#8ba8b7', width=1.5)),
                         name="RMSD ≤ 2Å",
                         showlegend=True))

    fig.add_trace(go.Bar(x=[0], y=[0],
                         marker=dict(color='#8ba8b7'),
                         name="RMSD ≤ 2Å and PB-valid",
                         showlegend=True))

    fig.add_trace(go.Bar(x=[0], y=[0],
                         marker=dict(color='white',
                                     pattern=dict(shape='/', fgcolor='#9600ff'),
                                     line=dict(color='#9600ff', width=1.5)),
                         name="Best RMSD ≤ 2Å",
                         showlegend=True))

    fig.add_trace(go.Bar(x=[0], y=[0],
                         marker=dict(color='#9600ff'),
                         name="Best RMSD ≤ 2Å and PB-valid",
                         showlegend=True))


    # Layout with LaTeX axis labels
    fig.update_layout(
        margin=dict(t=0, b=0, r=0, l=0),
        template="plotly_white",
        barmode="overlay",
        bargap=0.2,
        font=dict(size=18),
        xaxis=dict(
            tickmode="array",tickvals=x_vals, ticktext=x_tick_labels 
        ),
        yaxis=dict(
            title='Percent of predictions',
            tickmode="linear",
            tick0=0,
            dtick=0.2,
            tickformat=",.0%",
            range=[0, max_y * 1.3],
            ticks="outside",
            title_standoff=20
        ),
        legend=dict(
            orientation="h", x=0.5, y=1.25,xanchor="center",yanchor="top"
        ),
        width=w,height=h,yaxis_title_font=dict(size=18)
    )

    # Subset labels (LaTeX)
    cumulative = 0
    for subset in subsets:
        subset_dict, subset_groups = data[subset]
        n_bars = sum(len([t for t in subgroup if t in subset_dict])
                     for subgroup in subset_groups)
        if n_bars == 0:
            continue
        start = cumulative
        end = cumulative + n_bars - 1 + len(subset_groups)
        pos = (start + end) / 2
        fig.add_annotation(
            x=pos,
            y=max_y * 1.18,
            text=subset,
            font=dict(size=18),
            xref="x",
            yref="y",
            showarrow=False,
            xanchor="center",
            yanchor="bottom"
        )
        cumulative = end + 2

    fig.show()
    
    #fig.write_image(f"{title}.pdf")
    #fig.write_image(f"{title}.jpg")