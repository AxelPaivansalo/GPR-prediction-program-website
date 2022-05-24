from dash import dash_table
from dash import dcc
from dash import html
import dash_daq as daq

def define_layout(columns):
    text_style = {'font-family': 'Open Sans', 'font-size': 14}

    layout = html.Div([
        # Header
        html.H1('Design the perfect FoamWood block suited for your needs using AI', style=text_style | {'font-size': 28}),

        # General view
        html.H1('General view', style=text_style | {'font-size': 20}),

        # Note
        html.Div('Hint: Click a point to remove it and rerun the script', style=text_style),
        html.Br(),

        # Output graph
        dcc.Graph(id='gpr-prediction-plot', style={'width': 'min(100vw, 80vh)', 'height': 'min(100vw, 80vh)', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),

        # Basic options
        html.H1('Basic options', style=text_style | {'font-size': 20}),

        # Apply button
        html.Div('Refresh the page without losing your settings:', style=text_style),
        html.Div(html.Button('Apply', id='apply', n_clicks=0), style=text_style | {'text-align': 'center'}),
        html.Br(),

        # Input variable(s)
        html.Div('Input variable(s):', style=text_style),
        dcc.Dropdown(
            id='input-variables',
            options=[ {'label': x, 'value': x} for x in columns ],
            value=['Min grad angle(rad/C)', 'Storage modulus at min grad angle(Pa)'],
            multi=True,
            style=text_style
        ),
        html.Br(),

        # Output variable
        html.Div('Output variable:', style=text_style),
        dcc.Dropdown(
            id='output-variable',
            options=[ {'label': x, 'value': x} for x in columns ],
            value='Yield stress(Pa*10^6)',
            style=text_style
        ),
        html.Br(),

        # Horizontal line
        html.Hr(style={'height': '2px', 'background-color': 'black', 'border': 'none'}),
        html.Br(),

        # Advanced options
        html.H1('Advanced options', style=text_style | {'font-size': 20}),

        # Fix length scales button
        html.Div('Override the built-in optimizer and fix length scale values:', style=text_style),
        daq.BooleanSwitch(id='fix-length-scale-values', on=True),
        html.Br(),

        # Length scale 1
        html.Div(id='length-scale-1-text', style=text_style),
        dcc.Slider(id='length-scale-1', marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, disabled=True),
        html.Br(),

        # Length scale 2
        html.Div(id='length-scale-2-text', style=text_style),
        dcc.Slider(id='length-scale-2', marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, disabled=True),
        html.Br(),

        # Length scale 3
        html.Div(id='length-scale-3-text', style=text_style),
        dcc.Slider(id='length-scale-3', marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, disabled=True),
        html.Br(),

        # Specific point view
        html.H1('Specific point view', style=text_style | {'font-size': 20}),

        # Input value 1
        html.Div(id='input-value-1-text', style=text_style),
        dcc.Slider(id='input-value-1', marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, disabled=True),
        html.Br(),

        # Input value 2
        html.Div(id='input-value-2-text', style=text_style),
        dcc.Slider(id='input-value-2', marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, disabled=True),
        html.Br(),

        # Input value 3
        html.Div(id='input-value-3-text', style=text_style),
        dcc.Slider(id='input-value-3', marks=None, tooltip={'placement': 'bottom', 'always_visible': True}, disabled=True),
        html.Br(),

        # Output value
        html.Div(id='output-value', style=text_style),
        html.Br(),

        # Error
        html.H1('Model error', style=text_style | {'font-size': 20}),

        # Model error
        html.Div(id='training-error', style=text_style),
        html.Br(),
        html.Div(id='validation-error', style=text_style),
        html.Br(),

        # Further options
        html.H1('Further options', style=text_style | {'font-size': 20}),

        # Validation set size
        html.Div('Validation set size:', style=text_style),
        dcc.Slider(id='test-size', min=0, max=1, value=0.1, step=0.01, marks=None, tooltip={'placement': 'bottom', 'always_visible': True}),
        html.Br(),

        # Re-shuffle button
        html.Div('Re-shuffle the data by changing the seed used to generate random behaviour:', style=text_style),
        html.Div(html.Button('Re-shuffle Data', id='re-shuffle', n_clicks=0), style=text_style | {'text-align': 'center'}),
        html.Br(),

        # PCA
        html.H1('PCA', style=text_style | {'font-size': 20}),

        # Further plots
        dcc.Graph(id='gpr-prediction-plot-pca-2PC'),
        dcc.Graph(id='gpr-prediction-plot-pca-3PC'),

        # Principal components of each of the features
        html.H1('Principal components of each of the features', style=text_style | {'font-size': 20}),

        dash_table.DataTable(id='PCA-table', export_format='xlsx', export_headers='display'),

        # Parallel coordinates
        html.H1('Parallel coordinates', style=text_style | {'font-size': 20}),

        dcc.Graph(id='gpr-prediction-plot-par-coo'),

        # Header 3
        html.H1('Table of the data set', style=text_style | {'font-size': 20}),

        # Table of the data set
        html.Div('black: validation data (points that are not included in the training data)', style=text_style),
        html.Br(),
        html.Div('red: outlier (points that the user has clicked)', style=text_style),
        html.Br(),

        dash_table.DataTable(id='data-table', export_format='xlsx', export_headers='display', style_header={'white-space': 'normal'}),

        # Stored variables
        dcc.Store(id='x-data'),
        dcc.Store(id='y-data'),
        dcc.Store(id='length-scale-bounds'),
        dcc.Store(id='seed'),
        dcc.Store(id='outliers-index', data=[])
    ])

    return layout