from flask import Flask, render_template, request, url_for
import json
import networkx as nx
import numpy as np
import pandas as pd
import itertools
import random
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# custom functions. To understand: why if I put them in a separate file
# I get an error. For the moment, comment out and include functions below...
# from funcs import get_gen, get_many_gen, convert_to_float, plot_gen_plotly, plot_split_bar


app = Flask(__name__)
simul = None

@app.route('/callback1', methods=['POST', 'GET'])
def cb1():
    global simul
    return plot_gen_plotly(request.args.get('fgen1'), simul)
    
@app.route('/callback2', methods=['POST', 'GET'])
def cb2():
    global simul
    return plot_gen_plotly(request.args.get('fgen2'), simul)

@app.route("/")

def index():

    n_gen = request.args.get("n_gen", type=int) or 20
    pop_size = request.args.get("pop_size", type=int) or 20
    n_inter = request.args.get("n_inter", type=int) or 2
    start_split = request.args.get("split") or 0.8
    mult_c = request.args.get("mult_c") or 1.3
    mult_u = request.args.get("mult_u") or 1
    mult_adva = request.args.get("mult_adva") or 1
    mult_disa = request.args.get("mult_disa") or 1
    seed = request.args.get("seed")
    fgen1 = 0
    fgen2 = n_gen-1
    
    params = {'n_gen': n_gen,
              'pop_size': pop_size,
              'n_inter': n_inter,
              'start_split': start_split,
              'mult_c': mult_c,
              'mult_u': mult_u,
              'mult_adva': mult_adva,
              'mult_disa': mult_disa,
              'fgen1': fgen1,
              'fgen2': fgen2,
              'seed': seed}
    
    params = {k:(v if v is not None else '') for k,v in params.items()}
    
    try:
        start_split = convert_to_float(start_split)
        mult_c = convert_to_float(mult_c)
        mult_u = convert_to_float(mult_u)
        mult_disa = convert_to_float(mult_disa)
        mult_adva = convert_to_float(mult_adva)
        
        global simul
        simul = get_many_gen(
            n_gen = n_gen,
            pop_size = pop_size,
            mult_c = mult_c,
            mult_u = mult_u,
            mult_disa = mult_disa,
            mult_adva = mult_adva,
            start_split = start_split,
            n_inter = n_inter,
            seed = seed
        )

        splitJSON = plot_split_bar(simul)
        graphJSON1 = plot_gen_plotly(fgen1, simul)
        graphJSON2 = plot_gen_plotly(fgen2, simul)
        
        return render_template('index.html', params = params, splitJSON = splitJSON, graphJSON1 = graphJSON1, graphJSON2 = graphJSON2)
    except:
        return render_template('index.html', params = params, splitJSON = '', graphJSON1 = '', graphJSON2 = '') + 'Invalid input'

###############################
# custom functions start here #
###############################
def get_gen(pop_size, mult_c, mult_u, mult_disa, mult_adva, n_inter, split):
    
    gen_uncoop = int(np.round(pop_size*split))
    
    df = pd.DataFrame({'ID': range(pop_size),
                       'Cooperative': [True for i in range(pop_size-gen_uncoop)] + [False for i in range(gen_uncoop)],
                       'Fitness': 1})
    
    G1 = nx.Graph()

    G1.add_nodes_from(df.ID)

    nx.set_node_attributes(G1, pd.Series(df['Cooperative'], index = df['ID']).to_dict(), name = 'Cooperative')
    nx.set_node_attributes(G1, pd.Series(df['Fitness'], index = df['ID']).to_dict(), name = 'Fitness')

    for i in random.sample(list(G1.nodes), len(list(G1.nodes))):

        edge_list = G1.edges(i)
        n_edges_to_add = int(n_inter-len(edge_list))

        if n_edges_to_add > 0:

            nodes_to_ignore = set([i]+list(itertools.chain.from_iterable(edge_list)))

            node_list = list(G1.nodes())
            node_list = list(pd.Series(node_list)[[not i in nodes_to_ignore for i in node_list]])

            new_edges = [[i]+[x] for x in random.sample(node_list, n_edges_to_add)]

            G1.add_edges_from(new_edges)

    nx.set_node_attributes(G1, pd.Series(0, index = G1.nodes()).to_dict(), name = 'ncoop')
    nx.set_node_attributes(G1, pd.Series(0, index = G1.nodes()).to_dict(), name = 'nadva')
    nx.set_node_attributes(G1, pd.Series(0, index = G1.nodes()).to_dict(), name = 'ndisa')
    nx.set_node_attributes(G1, pd.Series(0, index = G1.nodes()).to_dict(), name = 'nunco')

    for i in G1.edges:
    
        if G1.nodes.data()[i[0]]['Cooperative'] & G1.nodes.data()[i[1]]['Cooperative']:
            G1.nodes.data()[i[0]]['Fitness'] *= mult_c
            G1.nodes.data()[i[1]]['Fitness'] *= mult_c
            G1.nodes.data()[i[0]]['ncoop'] += 1
            G1.nodes.data()[i[1]]['ncoop'] += 1
            
        elif (not G1.nodes.data()[i[0]]['Cooperative']) & (not G1.nodes.data()[i[1]]['Cooperative']):
            G1.nodes.data()[i[0]]['Fitness'] *= mult_u
            G1.nodes.data()[i[1]]['Fitness'] *= mult_u
            G1.nodes.data()[i[0]]['nunco'] += 1
            G1.nodes.data()[i[1]]['nunco'] += 1
            
        elif G1.nodes.data()[i[0]]['Cooperative']:
            G1.nodes.data()[i[0]]['Fitness'] *= mult_disa
            G1.nodes.data()[i[1]]['Fitness'] *= mult_adva
            G1.nodes.data()[i[0]]['ndisa'] += 1
            G1.nodes.data()[i[1]]['nadva'] += 1
            
        else:
            G1.nodes.data()[i[0]]['Fitness'] *= mult_adva
            G1.nodes.data()[i[1]]['Fitness'] *= mult_disa
            G1.nodes.data()[i[0]]['nadva'] += 1
            G1.nodes.data()[i[1]]['ndisa'] += 1

    fitvals = list(nx.get_node_attributes(G1, 'Fitness').values())
    new_fitvals = fitvals/np.max(fitvals)
    nx.set_node_attributes(G1, dict(zip(G1.nodes, new_fitvals)), 'Fitness')

    end_df = pd.DataFrame.from_dict(dict(G1.nodes(data=True)), orient='index').reset_index()
    end_df['Fitness'] = end_df['Fitness']/np.sum(end_df['Fitness'])
    
    try:
        end_split = end_df.groupby(by = 'Cooperative')['Fitness'].sum()[False]
    except:
        end_split = 0
    
    out = {'start_split': split,
           'end_split': end_split,
           'graph': G1}
    
    return out

def get_many_gen(n_gen, pop_size, mult_c, mult_u, mult_disa, mult_adva, n_inter, start_split, seed = None):

    if seed:
        random.seed(seed)
    
    out = {'F0': get_gen(pop_size, mult_c, mult_u, mult_disa, mult_adva, n_inter, start_split)}

    for i in range(n_gen-1):
        out[f'F{i+1}'] = get_gen(pop_size, mult_c, mult_u, mult_disa, mult_adva, n_inter, out[f'F{i}']['end_split'])
    
    return out

def convert_to_float(frac_str):
    if frac_str == None:
        return None
    else:
        try:
            return float(frac_str)
        except:
            num, denom = frac_str.split('/')
            return float(num)/float(denom)

def plot_gen_plotly(gen, sim):

    G = sim[f'F{gen}']['graph']
    pos = nx.spring_layout(G, seed = 1234)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='black'),
        hoverinfo='none',
        mode='lines')
    
    coop = nx.get_node_attributes(G,'Cooperative')
    node_color = ['lightgreen' if i == True else 'coral' for i in coop.values()]

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color = node_color,
            size = 20,
            line_width=1))

    node_text = []
    
    for i in G.nodes.data():

        if i[1]['Cooperative']:
            node_text.append(
                f"<b>Cooperative</b><br>"+
                f"Fitness: {str(np.round(i[1]['Fitness'],2))}<br>"+
                f"# of cooperative interactions: {i[1]['ncoop']}<br>"+
                f"# of disadvanteous interactions: {i[1]['ndisa']}"
            )

        else:
            node_text.append(
                f"<b>Uncooperative</b><br>"+
                f"Fitness: {str(np.round(i[1]['Fitness'],2))}<br>"+
                f"# of uncooperative interactions: {i[1]['nunco']}<br>"+
                f"# of advantageous interactions: {i[1]['nadva']}"
            )

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        #title=f'Generation F{gen}',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=5,l=20,r=20,t=5),
                        #template = 'plotly_white',
                        plot_bgcolor = 'rgba(0,0,0,0)',
                        paper_bgcolor = 'rgba(0,0,0,0)',
                        xaxis=dict(linecolor='#d0d0d0', showline=True, zeroline=False, showgrid=False, mirror=True, showticklabels=False),
                        yaxis=dict(linecolor='#d0d0d0', showline=True, zeroline=False, showgrid=False, mirror=True, showticklabels=False),
                        height = 500,
                        width = 500
                    )
                   )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def plot_split_bar(sim):
    gen_splits = [i['start_split'] for i in sim.values()]

    simul_split = pd.DataFrame({'x': list(range(len(gen_splits))),
                                'y': gen_splits})

    bar_trace = go.Bar(x = [str(i) for i in simul_split['x']],
                       y = simul_split['y'],
                       marker=dict(color = 'steelblue'),
                       hovertemplate="Generation <b>F%{x}</b>: "+"%{y:.2f}<extra></extra>")

    line_trace = go.Scatter(x = simul_split['x'],
                            y = simul_split['y'],
                            mode = 'lines',
                            line = dict(color='black'),
                            hoverinfo = 'none')

    fig = go.Figure(data=[bar_trace,line_trace],
                    layout = dict(showlegend=False,
                                  height = 500,
                                  width = 500,
                                  margin = dict(b=0,l=20,r=20,t=20),
                                  hovermode = 'x',
                                  hoverlabel = dict(bgcolor='white',
                                                    bordercolor='#808080'),
                                  template = 'plotly_white',
                                  xaxis = dict(linecolor='#d0d0d0',
                                               range=(-1.5,len(gen_splits)),
                                               dtick = 5),
                                  yaxis = dict(linecolor='#d0d0d0', gridcolor='#d0d0d0', showline=True, range=(0,1)),
                                  xaxis_title = 'Generation',
                                  yaxis_title = 'Proportion of uncooperative individuals'))

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


#########################################
#        Development server             #
# comment out next lines for deployment #
#########################################

# if __name__ == "__main__":

#     app.run(host="127.0.0.1", port=8080, debug=True)