# Contains code for plotting attention heatmap and predictions in a table
import plotly.express as px
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from torchaudio.functional import edit_distance as edit_dist
from plotly.subplots import make_subplots

'''
    function to return the 3 X 3 grid of plotly attention heatmaps for the word with number in
    word_numbers
'''
def generate_heatmap_grid(source_words, targets, preds, attn_matrices, word_numbers):
    indexes = [(w-1) for w in word_numbers]
    fig = make_subplots(3, 3, horizontal_spacing=0.1, vertical_spacing=0.1)
    for i in [1,2,3]:
        for j in [1,2,3]:
            index = indexes[3 * (i-1) + j-1]
            # we add a dummy number preceding the char to circumvent unique label bug in plotly
            # if a label is repeated (say 'a' appears twice in source), it is ignored. We avoid this
            # by adding the position in front in the label!
            x_labels = list(source_words[index]); x_labels = [(str(k+1) + ':' + l) for k, l in enumerate(x_labels)]
            y_labels = list(preds[index]) + ['$']; y_labels = [(str(k+1) + ':' + l) for k, l in enumerate(y_labels)]
            z = np.array(attn_matrices[index].cpu().numpy())
            z = z[1:(len(y_labels)+1), 1:(len(x_labels) + 1)]
            fig.add_trace(go.Heatmap(z=z, x=x_labels, y=y_labels, coloraxis = "coloraxis"), i, j)
            fig.update_xaxes(title=dict(text=f'Id : {index+1}; Source : {source_words[index]}; Target : {targets[index]}', font=dict(size=10)), row=i, col=j, side='top')
            fig.update_yaxes(title=dict(text=f'Pred : {preds[index]}', font=dict(size=10)), row=i, col=j, side='left',autorange="reversed")
    fig.update_layout(coloraxis = {'colorscale':'Greens'}, height=1500, width=1800)
    return fig

'''
    Function that will assign a color to each prediction in the matrix (rows = number of test examples, cols = (x, y_true, y_pred1, y_pred2 ...)). 
    This will be useful for plotting the test errors in the report using plotly tables.
'''
def color_code(pd_df):
    colorlist = ["mediumseagreen", "lightgreen", "yellow", "orange", "tomato"]
    rws, cols = len(pd_df), len(pd_df.columns)
    colors = [['#FFFFFF' for _ in range(rws)] for _ in range(cols)]
    for i in range(rws):
        for j in range(2, cols):
            # compute edit distance between true word and predicted word
            true_word = pd_df.iat[i,2]
            pred_word = pd_df.iat[i,j]
            edit_distance = edit_dist(true_word, pred_word)
            clip_edit = min(edit_distance, 4)
            colors[j][i] = colorlist[clip_edit]
    return colors

'''
    pd_df is a dataframe where 1st col = X, 2nd col = y, 3rd col = y_pred1, 4th col = y_pred2 ....
    locs are the locations that we want to display.
'''
def generate_table_and_legend(pd_df, locs):
    # get the subframe with only the rows with index in locs
    df_fil = pd_df.filter(items=locs, axis=0)
    # generate colors and plot the table in plotly
    colors = color_code(df_fil)
    table = go.Table(header=dict(values=df_fil.columns),
                     cells=dict(values=[list(df_fil[c]) for c in df_fil.columns], fill_color=colors, height=25))
    fig1 = go.Figure(data=[table])

    # also generate the legend for the color coding as a plotly table
    colorlist = ["mediumseagreen", "lightgreen", "yellow", "orange", "tomato"]
    table = go.Table(header=dict(values=['Color', 'Levenshtein distance']),
                     cells=dict(values=[['' for _ in range(len(colorlist))], [i for i in range(4)] + ['>= 4']], 
                                fill_color=[colorlist, ['#FFFFFF' for _ in range(len(colorlist))]], height=20
                                ))
    fig2 = go.Figure(data=[table])
    return fig1, fig2

'''
here, we will save all the predictions made in a local .csv file
'''
def save_predictions_file(src_list, tar_true_list, tar_pred_list, fname='predictions'):
    edit_distances = [edit_dist(pred,tar) for pred, tar in zip(tar_pred_list, tar_true_list)]
    pred_df = pd.DataFrame(zip(src_list, tar_true_list, tar_pred_list, edit_distances),
                            columns=['Source', 'Target', 'Predicted', 'Levenshtein Distance'])
    pred_df.to_csv('./'+fname+'.csv', index=False, encoding='utf-8')

'''
function to check the accuracy from a predictions file
'''
def check_accuracy(folname='predictions_vanilla', fname='predictions.csv'):
    df = pd.read_csv(f'./{folname}/{fname}')
    crct = 0
    pred = df['Predicted']
    tar = df['Target']
    tot = 0
    for x, y in zip(pred, tar):
        tot += 1
        if x == y:
            crct += 1
    return crct / tot