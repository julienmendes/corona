import datetime
import os
import yaml

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from dash.exceptions import PreventUpdate

# Lecture du fichier d'environnement
ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

# Lecture du fichier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df: _df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020, 3, 21)]
              )
epidemie_df['Country/Region'].replace('China', 'Mainland China', inplace=True)

countries = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]

app = dash.Dash('Corona Virus Explorer')
app.layout = html.Div([
    html.H1(['Corona Virus Explorer'], style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
            html.Div([
                dcc.Dropdown(
                    id='country',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country2',
                    options=countries
                )
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable',
                    options=[
                        {'label': 'Confirmed', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')
            ]),   
        ]),
        dcc.Tab(label='Map', children=[
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                #marks={i:str(date) for i, date in enumerate(epidemie_df['day'].unique())}
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )  
        ]),
        dcc.Tab(label='Model', children=[
            html.P(['We choose to multiply by 0.00001 the population in order to have faster dashboard'], style={'textAlign': 'center'}),
            html.P(['Moreover by default the contry selected is South Korea'], style={'textAlign': 'center'}),
            html.H6(['default population: 51470000, default Beta: 0.01, default Alpha: 0.1'], style={'textAlign': 'center'}),
            html.Div([
                dcc.Dropdown(
                    id='country3',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Input(
                    id='input1',
                    placeholder='Country Population...',
                    type='number'
                ),
                dcc.Input(
                    id='input2',
                    placeholder='choose an alpha...',
                    type='number'
                ),
                dcc.Input(
                    id='input3',
                    placeholder='Choose a Beta...',
                    type='number'
                ),
            ], style={'display': 'inline-block','padding': 10, 'width':'100%'}),
            html.Div([
                dcc.RadioItems(
                    id='check',
                    options=[
                        {'label': 'Auto', 'value': 'auto'},
                        {'label': 'Manual', 'value': 'manu'}
                    ],
                    value='manu',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph2')
            ]),
        ]),
    ]),
])


def sumsq_error(parameters):
    beta, gamma = parameters
    def SIR(t,y):
        S=y[0]
        I=y[1]
        R=y[2]
        return([-beta*S*I, beta*S*I-gamma*I, gamma*I])
    
    solution = solve_ivp(SIR,[0,nb_steps-1],[pop,1,0],t_eval=np.arange(0,nb_steps,1))
        
    return(sum((solution.y[1]-model_df['infected'])**2))

def SIR(t,y):
    S = y[0]
    I = y[1]
    R = y[2]
    return([-beta*S*I, beta*S*I-gamma*I, gamma*I])


@app.callback(
    Output('graph2', 'figure'),
    [
        Input('country3', 'value'),
        Input('input1', 'value'),
        Input('input2', 'value'),
        Input('input3', 'value'),
        Input('check', 'value')
    ]
)

def SIRmodel(country3, input1, input2, input3, check):
    global model_df
    global beta
    global gamma
    global pop
    global nb_steps
    if input3 is None:
        beta = 0.01
    else:
        beta=input3
    if input2 is None:
        gamma = 0.1
    else:
        gamma = input2
    if input1 is None:
        pop = 51470000*0.00001
    else:
        pop = input1*0.00001
    
    
    if country3 is not None:
        model_df = (epidemie_df[epidemie_df['Country/Region'] == country3])
        model_df = (model_df
            .groupby(['Country/Region', 'day'])
            .agg({'Confirmed': 'sum'})
            .reset_index()
            )
        model_df['infected'] = model_df['Confirmed'].diff()
        model_df = model_df.loc[2:]
    else:
        model_df = (epidemie_df[epidemie_df['Country/Region'] == 'South Korea'])
        model_df = (model_df
            .groupby(['Country/Region', 'day'])
            .agg({'Confirmed': 'sum'})
            .reset_index()
            )
        model_df['infected'] = model_df['Confirmed'].diff()
        model_df = model_df.loc[2:]
    
    nb_steps = len(model_df)
    
    if check == 'manu':
        solution = solve_ivp(SIR,[0,len(model_df)],[pop,1,0],t_eval=np.arange(0,len(model_df),1))
    else:
        msol = minimize(sumsq_error,[beta,gamma],method='Nelder-Mead')
        beta = msol.x[0]
        gamma = msol.x[1]
        solution = solve_ivp(SIR,[0,len(model_df)],[pop,1,0],t_eval=np.arange(0,len(model_df),1))

    
    model_df['range'] = [i for i in range(0,len(model_df))]
    #model_df['range'] = l
    return {
        'data': [
            dict(
                x=model_df['range'],
                y=model_df['infected'],
                type='line',
                name='Original Data'
            )
        ] + ([
            dict(
                x=solution.t,
                y=solution.y[0],
                type='line',
                name="Susceptible"
            )            
        ]) + ([
            dict(
                x=solution.t,
                y=solution.y[1],
                type='line',
                name="Infected"
            
            )            
        ]) + ([
            dict(
                x=solution.t,
                y=solution.y[2],
                type='line',
                name="Removed"
            )            
        ]),
        'layout':{
            'title': (country3 if country3 is not None else 'South Korea' + '1'),
            'title': country3 if country3 is not None else 'South Korea',
            'xaxis':{
                'title':'Days'
            },
            'yaxis':{
                'title':'Population'
            }
        }
    }



@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),
        Input('country2', 'value'),
        Input('variable', 'value'),        
    ]
)
def update_graph(country, country2, variable):
    print(country)
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )
    if country2 is not None:
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                     .groupby(['Country/Region', 'day'])
                     .agg({variable: 'sum'})
                     .reset_index()
                    )

        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )            
        ] if country2 is not None else [])
    }

@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day', 'value'),
    ]
)
def update_map(map_day):
    day = epidemie_df['day'].unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({'Confirmed': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )
    print(map_day)
    print(day)
    print(map_df.head())
    return {
        'data': [
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + ' (' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed'] / 1_000, 5)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True),
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
