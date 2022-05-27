import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import LinearNDInterpolator
from random import randint
from functools import reduce
from iteration_utilities import deepflatten
import additional_methods as add_methods
import data_methods
import layout_methods as lay_methods

app = dash.Dash(__name__)
server = app.server

data, columns = data_methods.load_data()

# Set signal and noise variance
signal_variance_all = { x: 0.5 for x in columns }
noise_variance_all = { x: 0 for x in columns }

app.layout = lay_methods.define_layout(columns)

# Update data
@app.callback(
    Output('x-data', 'data'),
    Output('y-data', 'data'),
    Output('outliers-index', 'data'),

    Input('input-variables', 'value'),
    Input('output-variable', 'value'),
    Input('gpr-prediction-plot', 'clickData'),
    Input('apply', 'n_clicks'),
    State('x-data', 'data'),
    State('y-data', 'data'),
    State('outliers-index', 'data')
)
def update_data(input, output, outlier_plot, apply, X_data, Y_data, outliers_index):
    # Interactive outlier removal
    if dash.callback_context.triggered[0]['prop_id'] == 'gpr-prediction-plot.clickData':
        if 'customdata' in dash.callback_context.triggered[0]['value']['points'][0]:
            # Convert JSON strings to pandas DataFrame-objects
            X_data, Y_data = pd.read_json(X_data, orient='split'), pd.read_json(Y_data, orient='split')
            
            outlier = dash.callback_context.triggered[0]['value']['points'][0]['customdata']

            # Convert pandas DataFrame-objects to JSON strings so they can be used outside of this method
            return X_data.drop(index=outlier).to_json(orient='split'), Y_data.drop(index=outlier).to_json(orient='split'), outliers_index + [outlier]
        
        return dash.no_update, dash.no_update, dash.no_update
    
    X_data = pd.DataFrame({ x: y for x, y in zip(columns, data) })
    Y_data = pd.DataFrame({ x: y for x, y in zip(columns, data) })

    X_dropped_col = [ x for x in X_data.columns if x not in input ]
    Y_dropped_col = [ x for x in Y_data.columns if x not in output ]

    # Convert pandas DataFrame-objects to JSON strings so they can be used outside of this method
    return X_data.drop(columns=X_dropped_col).to_json(orient='split'), Y_data.drop(columns=Y_dropped_col).to_json(orient='split'), dash.no_update

# Update length scale
@app.callback(
    Output('length-scale-1', 'min'),
    Output('length-scale-1', 'max'),
    Output('length-scale-1', 'value'),
    Output('length-scale-1', 'step'),
    Output('length-scale-1', 'disabled'),
    Output('length-scale-1', 'tooltip'),
    Output('length-scale-1-text', 'children'),

    Output('length-scale-2', 'min'),
    Output('length-scale-2', 'max'),
    Output('length-scale-2', 'value'),
    Output('length-scale-2', 'step'),
    Output('length-scale-2', 'disabled'),
    Output('length-scale-2', 'tooltip'),
    Output('length-scale-2-text', 'children'),

    Output('length-scale-3', 'min'),
    Output('length-scale-3', 'max'),
    Output('length-scale-3', 'value'),
    Output('length-scale-3', 'step'),
    Output('length-scale-3', 'disabled'),
    Output('length-scale-3', 'tooltip'),
    Output('length-scale-3-text', 'children'),

    Output('length-scale-bounds', 'data'),

    Input('x-data', 'data'),
    Input('apply', 'n_clicks')
)
def update_length_scale(X_data, apply):
    # Convert JSON strings to pandas DataFrame-objects
    X_data = pd.read_json(X_data, orient='split')

    # For further reading on why the length scale bounds are defined like this:
    # https://stats.stackexchange.com/questions/297673/how-to-pick-length-scale-bounds-for-rbc-kernels-in-gaussian-process-regression
    length_scale_min = [ add_methods.min_dist_arr(X_data[x]) for x in X_data.columns ]
    length_scale_max = [ add_methods.max_dist_arr(X_data[x]) for x in X_data.columns ]

    length_scale_value = [ round((x + y) / 2, 2) for x, y in zip(length_scale_max, length_scale_min) ]

    length_scale_step = [ round((x - y) / 100, 3) for x, y in zip(length_scale_max, length_scale_min) ]

    length_scale_disabled = [ False for x in X_data.columns ]

    length_scale_tooltip = [ {'placement': 'bottom', 'always_visible': True} for x in X_data.columns ]

    length_scale_text = [ 'Length scale of {}:'.format(x) for x in X_data.columns ]

    length_scale_bounds = [ (x, y) for x, y in zip(length_scale_min, length_scale_max) ]

    return list(reduce(lambda a, b: a + b, [ x for x in zip(length_scale_min, length_scale_max, length_scale_value, length_scale_step, length_scale_disabled, length_scale_tooltip, length_scale_text) ])) + \
        list(reduce(lambda a, b: a + b, [ (0, 2, 1, 0.1, True, {'placement': 'bottom', 'always_visible': False}, 'Choose more variables to adjust') for x in range(0, 3 - len(X_data.columns)) ], ())) + \
            [length_scale_bounds]

# Update input value
@app.callback(
    Output('input-value-1', 'min'),
    Output('input-value-1', 'max'),
    Output('input-value-1', 'value'),
    Output('input-value-1', 'step'),
    Output('input-value-1', 'disabled'),
    Output('input-value-1', 'tooltip'),
    Output('input-value-1-text', 'children'),

    Output('input-value-2', 'min'),
    Output('input-value-2', 'max'),
    Output('input-value-2', 'value'),
    Output('input-value-2', 'step'),
    Output('input-value-2', 'disabled'),
    Output('input-value-2', 'tooltip'),
    Output('input-value-2-text', 'children'),

    Output('input-value-3', 'min'),
    Output('input-value-3', 'max'),
    Output('input-value-3', 'value'),
    Output('input-value-3', 'step'),
    Output('input-value-3', 'disabled'),
    Output('input-value-3', 'tooltip'),
    Output('input-value-3-text', 'children'),

    Input('x-data', 'data'),
    Input('apply', 'n_clicks')
)
def update_input_value(X_data, apply):
    # Convert JSON strings to pandas DataFrame-objects
    X_data = pd.read_json(X_data, orient='split')

    input_min = [ X_data[x].min() - ((X_data[x].max() - X_data[x].min()) / 10) for x in X_data.columns ]

    input_max = [ X_data[x].max() + ((X_data[x].max() - X_data[x].min()) / 10) for x in X_data.columns ]

    input_value = [ round((x + y) / 2, 2) for x, y in zip(input_max, input_min) ]

    input_step = [ round((x - y) / 100, 3) for x, y in zip(input_max, input_min) ]

    input_disabled = [ False for x in X_data.columns ]

    input_tooltip = [ {'placement': 'bottom', 'always_visible': True} for x in X_data.columns ]

    input_text = [ 'Input value of {}:'.format(x) for x in X_data.columns ]

    return list(reduce(lambda a, b: a + b, [ x for x in zip(input_min, input_max, input_value, input_step, input_disabled, input_tooltip, input_text) ])) + \
        list(reduce(lambda a, b: a + b, [ (0, 2, 1, 0.1, True, {'placement': 'bottom', 'always_visible': False}, 'Choose more variables to adjust') for x in range(0, 3 - len(X_data.columns)) ], ()))

# Update seed
@app.callback(
    Output('seed', 'data'),

    Input('re-shuffle', 'n_clicks')
)
def update_seed(re_shuffle):
    return randint(0, 1000)

# Update graphs
@app.callback(
    Output('gpr-prediction-plot', 'figure'),
    Output('gpr-prediction-plot-pca-2PC', 'figure'),
    Output('gpr-prediction-plot-pca-3PC', 'figure'),
    Output('gpr-prediction-plot-par-coo', 'figure'),
    Output('PCA-table', 'data'),
    Output('data-table', 'data'),
    Output('data-table', 'style_data_conditional'),
    Output('output-value', 'children'),
    Output('training-error', 'children'),
    Output('validation-error', 'children'),

    Input('y-data', 'data'),
    Input('length-scale-1', 'value'),
    Input('length-scale-2', 'value'),
    Input('length-scale-3', 'value'),
    Input('length-scale-1', 'disabled'),
    Input('length-scale-2', 'disabled'),
    Input('length-scale-3', 'disabled'),
    Input('length-scale-bounds', 'data'),
    Input('fix-length-scale-values', 'on'),
    Input('test-size', 'value'),
    Input('seed', 'data'),
    Input('input-value-1', 'value'),
    Input('input-value-2', 'value'),
    Input('input-value-3', 'value'),
    Input('apply', 'n_clicks'),
    State('x-data', 'data'),
    State('outliers-index', 'data')
)
def update_graphs(Y_data, length_scale_1, length_scale_2, length_scale_3, length_scale_1_disabled, length_scale_2_disabled, length_scale_3_disabled, length_scale_bounds, fix_length_scale_values, test_size, seed, input_value_1, input_value_2, input_value_3, apply, X_data, outliers_index):
    # Convert JSON strings to pandas DataFrame-objects
    X_data, Y_data = pd.read_json(X_data, orient='split'), pd.read_json(Y_data, orient='split')

    # Configure length scales, signal variance and noise variance
    length_scale = [ x[1] for x in zip([length_scale_1_disabled, length_scale_2_disabled, length_scale_3_disabled], [length_scale_1, length_scale_2, length_scale_3]) if x[0] == False ]
    signal_variance = signal_variance_all[Y_data.columns[0]]
    noise_variance = noise_variance_all[Y_data.columns[0]]

    # Initialize kernel and GP regressor
    if fix_length_scale_values:
        length_scale_bounds = 'fixed'
    
    rbf = ConstantKernel(constant_value=signal_variance) * \
    RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + \
    WhiteKernel(noise_level=noise_variance)

    gpr = GaussianProcessRegressor(kernel=rbf, alpha=0.0)

    # Split the data
    X_train, X_val, Y_train, Y_val = add_methods.split_data(X_data, Y_data, test_size, seed)

    # Create GP model
    gpr.fit(X_train, Y_train)

    if len(X_data.columns) == 1:
        fig = go.Figure()

        # Create grid
        x_val = np.linspace(
            X_train[X_data.columns[0]].min() - ((X_train[X_data.columns[0]].max() - X_train[X_data.columns[0]].min()) / 10),
            X_train[X_data.columns[0]].max() + ((X_train[X_data.columns[0]].max() - X_train[X_data.columns[0]].min()) / 10),
            100
        )
        X_pred = x_val.reshape(-1, 1)

        # Interpolate
        # Color: https://waldyrious.net/viridis-palette-generator/
        mu_val, std_val = gpr.predict(X_pred, return_std=True)

        fig.add_trace(go.Scatter(
            x=x_val, y=mu_val - std_val, line_width=0,
            line_color='rgba(33, 145, 140, 1)', name=''
        ))
        fig.add_trace(go.Scatter(
            x=x_val, y=mu_val + std_val, fill='tonexty', line_width=0,
            line_color='rgba(33, 145, 140, 1)', name='Standard deviation'
        ))
        fig.add_trace(go.Scatter(
            x=x_val, y=mu_val,
            line_color='rgba(33, 145, 140, 1)', name='Interpolation function'
        ))
        fig.add_trace(go.Scatter(
            x=X_train[X_train.columns[0]], y=Y_train[Y_train.columns[0]], customdata=X_train.index.tolist(),
            line_color='rgba(16, 24, 32, 0.7)', mode='markers', name='Training data'
        ))
        fig.update_layout(
            title='Interpolation of the training data set',
            xaxis_title=X_data.columns[0],
            yaxis_title=Y_data.columns[0]
        )

        # Calculate output value
        X_input = pd.DataFrame({X_data.columns[0]: [input_value_1]})
        output_value = gpr.predict(X_input)[0]
    elif len(X_data.columns) == 2:
        # Create grid
        n = 100
        x_val, y_val = [ np.linspace(
            X_train[x].min() - ((X_train[x].max() - X_train[x].min()) / 10),
            X_train[x].max() + ((X_train[x].max() - X_train[x].min()) / 10),
            n
        ) for x in X_data.columns ]
        xx, yy = np.meshgrid(x_val, y_val)
        X_pred = np.c_[xx.ravel(), yy.ravel()]

        # Interpolate
        mu_val = gpr.predict(X_pred)
        mu_val = np.array(mu_val).reshape((n, n))

        fig = go.Figure(data=[
            go.Surface(
                z=mu_val, x=x_val, y=y_val, coloraxis='coloraxis', opacity=0.8, name='Interpolation function'),
            go.Scatter3d(
                x=X_train[X_train.columns[0]], y=X_train[X_train.columns[1]], z=Y_train[Y_train.columns[0]], customdata=X_train.index.tolist(),
                mode='markers', marker={'size': 6, 'color': Y_train[Y_train.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data')
        ])
        
        fig.update_layout(
            title='Interpolation of the training data set',
            scene={
                'xaxis_title': X_data.columns[0],
                'yaxis_title': X_data.columns[1],
                'zaxis_title': Y_data.columns[0],
                'camera': {'eye': {'x': 2, 'y': 2, 'z': 2}}
            },
            coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'}
        )

        # Calculate output value
        X_input = pd.DataFrame({ x: y for x, y in zip(X_data.columns, [[input_value_1], [input_value_2]]) })
        output_value = gpr.predict(X_input)[0]
    elif len(X_data.columns) == 3:
        # Create grid
        n = 10
        x_val, y_val, z_val = [ np.linspace(
            X_train[x].min() - ((X_train[x].max() - X_train[x].min()) / 10),
            X_train[x].max() + ((X_train[x].max() - X_train[x].min()) / 10),
            n
        ) for x in X_data.columns ]
        xx, yy, zz = np.meshgrid(x_val, y_val, z_val)
        X_pred = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        # Interpolate
        mu_val = gpr.predict(X_pred)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), text=[ 'color: {}'.format(x) for x in mu_val ],
                mode='markers', marker={'size': 4, 'color': mu_val, 'coloraxis': 'coloraxis'}, name='Interpolation function'),
            go.Scatter3d(
                x=X_train[X_train.columns[0]], y=X_train[X_train.columns[1]], z=X_train[X_train.columns[2]], text=[ 'color: {}'.format(x) for x in Y_train[Y_train.columns[0]] ], customdata=X_train.index.tolist(),
                mode='markers', marker={'size': 12, 'color': Y_train[Y_train.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data')
        ])
        
        fig.update_layout(
            title='Interpolation of the training data set',
            scene={
                'xaxis_title': X_data.columns[0],
                'yaxis_title': X_data.columns[1],
                'zaxis_title': X_data.columns[2],
                'camera': {'eye': {'x': 2, 'y': 2, 'z': 2}}
            },
            coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'},
            showlegend=False
        )

        # Calculate output value
        X_input = pd.DataFrame({ x: y for x, y in zip(X_data.columns, [[input_value_1], [input_value_2], [input_value_3]]) })
        output_value = gpr.predict(X_input)[0]
    
    # Model error
    mu_val_train = gpr.predict(X_train)
    mu_val_val, std_val_val = gpr.predict(X_val, return_std=True)

    train_error, val_error = add_methods.gp_error(mu_val_train, mu_val_val, std_val_val, X_val, Y_train, Y_val)

    # Draw PCA figure
    X_data_pca = pd.DataFrame({ x: y for x, y in zip(columns, data) })
    X_data_pca = X_data_pca.drop(columns=Y_data.columns[0])
    Y_data_pca = pd.DataFrame({ x: y for x, y in zip(columns, data) })
    Y_data_pca = Y_data_pca.drop(columns=[ x for x in columns if not x == Y_data.columns[0] ])
    
    X_data_pca[X_data_pca.columns] = StandardScaler().fit_transform(X_data_pca[X_data_pca.columns])
    
    # PCA with 2 principal components
    pca = PCA(n_components=2)
    pca.fit(X_data_pca)
    X_data_pca_2pc = pca.transform(X_data_pca)

    # Create grid
    n = [150, 50]
    x_val, y_val = [ np.linspace(
        X_data_pca_2pc[:, x].min() - ((X_data_pca_2pc[:, x].max() - X_data_pca_2pc[:, x].min()) / 10),
        X_data_pca_2pc[:, x].max() + ((X_data_pca_2pc[:, x].max() - X_data_pca_2pc[:, x].min()) / 10),
        y
    ) for x, y in zip(range(0, 2), n) ]
    xx, yy = np.meshgrid(x_val, y_val)

    # Interpolate
    interp = LinearNDInterpolator(X_data_pca_2pc, Y_data_pca[Y_data_pca.columns[0]])
    col_interp = interp(xx, yy)
    xx = [ [ b for a, b in zip(x, y) if not np.isnan(a) ] for x, y in zip(col_interp, xx) ]
    yy = [ [ b for a, b in zip(x, y) if not np.isnan(a) ] for x, y in zip(col_interp, yy) ]
    color_interp = [ [ y for y in x if not np.isnan(y) ] for x in col_interp ]
    xx = list(deepflatten(xx))
    yy = list(deepflatten(yy))
    color_interp = list(deepflatten(color_interp))

    fig_pca_2pc = go.Figure()

    fig_pca_2pc.add_trace(go.Scatter(
        x=xx, y=yy, text=[ 'color: {}'.format(x) for x in color_interp ],
        mode='markers', marker={'size': 4, 'color': color_interp, 'coloraxis': 'coloraxis'}, name='Linear interpolation'
    ))
    fig_pca_2pc.add_trace(go.Scatter(
        x=X_data_pca_2pc[:, 0], y=X_data_pca_2pc[:, 1], text=[ 'color: {}'.format(x) for x in Y_data_pca[Y_data_pca.columns[0]] ],
        mode='markers', marker={'size': 12, 'color': Y_data_pca[Y_data_pca.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data'
    ))
    fig_pca_2pc.update_layout(
        title='PCA with 2 principal components',
        xaxis_title='PC1',
        yaxis_title='PC2',
        xaxis_range=[x_val[0], x_val[-1]],
        yaxis_range=[y_val[0], y_val[-1]],
        coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'},
        showlegend=False
    )

    # PCA with 3 principal components
    pca = PCA(n_components=3)
    pca.fit(X_data_pca)
    X_data_pca_3pc = pca.transform(X_data_pca)

    # Create grid
    n = 10
    x_val, y_val, z_val = [ np.linspace(
        X_data_pca_3pc[:, x].min() - ((X_data_pca_3pc[:, x].max() - X_data_pca_3pc[:, x].min()) / 10),
        X_data_pca_3pc[:, x].max() + ((X_data_pca_3pc[:, x].max() - X_data_pca_3pc[:, x].min()) / 10),
        n
    ) for x in range(0, 3) ]
    xx, yy, zz = np.meshgrid(x_val, y_val, z_val)

    # Interpolate
    interp = LinearNDInterpolator(X_data_pca_3pc, Y_data_pca[Y_data_pca.columns[0]])
    col_interp = interp(xx, yy, zz)
    xx = [ [ [ j for i, j in zip(a, b) if not np.isnan(i) ] for a, b in zip(x, y) ] for x, y in zip(col_interp, xx) ]
    yy = [ [ [ j for i, j in zip(a, b) if not np.isnan(i) ] for a, b in zip(x, y) ] for x, y in zip(col_interp, yy) ]
    zz = [ [ [ j for i, j in zip(a, b) if not np.isnan(i) ] for a, b in zip(x, y) ] for x, y in zip(col_interp, zz) ]
    color_interp = [ [ [ i for i in a if not np.isnan(i) ] for a in x ] for x in col_interp ]
    xx = list(deepflatten(xx))
    yy = list(deepflatten(yy))
    zz = list(deepflatten(zz))
    color_interp = list(deepflatten(color_interp))

    fig_pca_3pc = go.Figure(data=[
        go.Scatter3d(
            x=xx, y=yy, z=zz, text=[ 'color: {}'.format(x) for x in color_interp ],
            mode='markers', marker={'size': 4, 'color': color_interp, 'coloraxis': 'coloraxis'}, name='Linear interpolation'),
        go.Scatter3d(
            x=X_data_pca_3pc[:, 0], y=X_data_pca_3pc[:, 1], z=X_data_pca_3pc[:, 2], text=[ 'color: {}'.format(x) for x in Y_data_pca[Y_data_pca.columns[0]] ],
            mode='markers', marker={'size': 12, 'color': Y_data_pca[Y_data_pca.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data')
    ])

    fig_pca_3pc.update_layout(
        title='PCA with 3 principal components',
        scene={
            'xaxis_title': 'PC1',
            'yaxis_title': 'PC2',
            'zaxis_title': 'PC3',
            'xaxis_range': [x_val[0], x_val[-1]],
            'yaxis_range': [y_val[0], y_val[-1]]
        },
        coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'},
        showlegend=False
    )

    # PCA table
    pca_table_columns = ['Feature', 'Component 1', 'Component 2', 'Component 3']
    comp = pca.components_
    pca_table_data = [ {
        pca_table_columns[0]: col,
        pca_table_columns[1]: round(pc1, 3),
        pca_table_columns[2]: round(pc2, 3),
        pca_table_columns[3]: round(pc3, 3)
    } for col, pc1, pc2, pc3 in zip(X_data_pca.columns, comp[0, :], comp[1, :], comp[2, :]) ]

    # Draw parallel coordinates figure
    fig_par_coo = go.Figure(data=
        go.Parcoords(
            line={
                'color': Y_train[Y_train.columns[0]],
                'colorscale': 'viridis',
                'showscale': True
            },
            dimensions=[ {'label': x, 'values': X_train[x]} for x in X_train.columns ] + [{'label': Y_train.columns[0], 'values': Y_train[Y_train.columns[0]]}]
        )
    )

    # Data set table
    data_table_columns = ['Data point'] + columns
    XY_data = pd.DataFrame({ x: y for x, y in zip(columns, data) })
    data_table_data = [ dict(zip(data_table_columns, [col] + [ round(x, 3) for x in data ])) for col, data in zip(range(0, len(XY_data.index)), zip(*np.transpose(XY_data.to_numpy()).tolist())) ]

    data_table_style = [{
        'if': {'row_index': [ x for x in XY_data.index if not x in X_train.index ]},
        'backgroundColor': 'rgba(51, 51, 51, 0.7)',
        'color': 'white'
    }, {
        'if': {'row_index': outliers_index},
        'backgroundColor': 'rgba(222, 6, 26, 0.7)',
        'color': 'white'
    }]

    return [fig] + [fig_pca_2pc] + [fig_pca_3pc] + [fig_par_coo] + [pca_table_data] + [data_table_data] + [data_table_style] + \
        ['Output value of {}: {}'.format(Y_data.columns[0], round(output_value, 3))] + ['Relative training error: {}%'.format(round(train_error, 3))] + ['Relative validation error: {}%'.format(round(val_error, 3))]

if __name__ == '__main__':
    app.run_server(debug=True)