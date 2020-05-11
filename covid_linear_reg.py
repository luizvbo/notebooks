# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% execution={"iopub.status.busy": "2020-05-11T05:59:16.231896Z", "iopub.execute_input": "2020-05-11T05:59:16.232686Z", "iopub.status.idle": "2020-05-11T05:59:16.242204Z", "shell.execute_reply.started": "2020-05-11T05:59:16.232615Z", "shell.execute_reply": "2020-05-11T05:59:16.240507Z"}
import pandas as pd
import cufflinks as cf
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from IPython.display import HTML, display

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# %% execution={"iopub.status.busy": "2020-05-11T05:59:16.513882Z", "iopub.execute_input": "2020-05-11T05:59:16.514631Z", "iopub.status.idle": "2020-05-11T05:59:16.522901Z", "shell.execute_reply.started": "2020-05-11T05:59:16.514562Z", "shell.execute_reply": "2020-05-11T05:59:16.520653Z"}
cases = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
         "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
death = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
         "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")

# %% execution={"iopub.status.busy": "2020-05-11T05:59:17.306307Z", "iopub.execute_input": "2020-05-11T05:59:17.307342Z", "iopub.status.idle": "2020-05-11T05:59:18.042803Z", "shell.execute_reply.started": "2020-05-11T05:59:17.307251Z", "shell.execute_reply": "2020-05-11T05:59:18.042178Z"}
df_death = pd.read_csv(death)
df_cases = pd.read_csv(cases)


# %% execution={"iopub.status.busy": "2020-05-11T05:59:20.990298Z", "iopub.execute_input": "2020-05-11T05:59:20.991614Z", "iopub.status.idle": "2020-05-11T05:59:21.056502Z", "shell.execute_reply.started": "2020-05-11T05:59:20.991498Z", "shell.execute_reply": "2020-05-11T05:59:21.055901Z"}
def get_same_origin(df):
    n_days = df.shape[0]

    def _pad_days(s):
        s = s.astype(float)
        s_pad = s[s.cumsum() != 0]
        return np.pad(s_pad, (0, n_days-s_pad.shape[0]),
                      'constant', constant_values=np.nan)

    df = (
        df
        .apply(_pad_days, raw=True)
        .reset_index(drop=True)
    ).dropna(how='all')

    # df.index += 1
    return df


def set_index(df):
    """Set the index for the data frame using the date

    Args:
        df: Pandas data frame obtained from John Hopkins repo
    """
    # Set region, country, lat and long as index
    index = pd.MultiIndex.from_frame(df.iloc[:, :4])
    # Set the index and transpose
    df = df.iloc[:, 4:].set_index(index).T
    # Set date as index
    return df.set_index(pd.to_datetime(df.index, dayfirst=False))


df = set_index(df_death.copy())

# Groupy territories per country
df = df.groupby(level=1, axis=1).sum()

# # Drop all-zeros columns
df = df[df.sum()[lambda s: s > 0].index]

# # Shift all series to the origin (first death)
df = get_same_origin(df)

# %% execution={"iopub.status.busy": "2020-05-11T06:37:01.337062Z", "iopub.execute_input": "2020-05-11T06:37:01.338417Z", "iopub.status.idle": "2020-05-11T06:37:06.779008Z", "shell.execute_reply.started": "2020-05-11T06:37:01.338296Z", "shell.execute_reply": "2020-05-11T06:37:06.778115Z"}


min_diff = 7
all_columns = df.columns
results = {}

for i, x_label in enumerate(tqdm(all_columns)):
    for y_label in all_columns[i+1:]:
        x =  df[x_label].dropna().values
        y =  df[y_label].dropna().values

        if x_label == 'Brazil':
            print(i, y_label)

        # Keep the largest array in x
        if x.shape[0] < y.shape[0]:
            x, y = y, x
            x_label, y_label = y_label, x_label

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)

        if x.shape[0] - y.shape[0] < min_diff:
            pass

        lr = LinearRegression()
        # The weights increase linearly from 1 to 2
        weights = np.linspace(1, 1, y.shape[0])
        lr.fit(x[:y.shape[0]], y, weights)

        results[(x_label, y_label)] = dict(
            lr_model=lr, r_score=lr.score(x[:y.shape[0]], y),
            predictions=lr.predict(x),
            len_x = x.shape[0],
            len_y = y.shape[0],
        )

# %% execution={"iopub.status.busy": "2020-05-11T06:24:02.094820Z", "iopub.execute_input": "2020-05-11T06:24:02.095602Z", "iopub.status.idle": "2020-05-11T06:24:02.121294Z", "shell.execute_reply.started": "2020-05-11T06:24:02.095531Z", "shell.execute_reply": "2020-05-11T06:24:02.120128Z"}
df_results = pd.DataFrame.from_dict(results, orient='index')

# %% execution={"iopub.status.busy": "2020-05-11T06:24:02.307430Z", "iopub.execute_input": "2020-05-11T06:24:02.308580Z", "iopub.status.idle": "2020-05-11T06:24:02.478998Z", "shell.execute_reply.started": "2020-05-11T06:24:02.308460Z", "shell.execute_reply": "2020-05-11T06:24:02.478349Z"}
df_results.r_score[lambda s: s > 0.8].plot.hist(bins=30)

# %% execution={"iopub.status.busy": "2020-05-11T06:24:02.943618Z", "iopub.execute_input": "2020-05-11T06:24:02.943833Z", "iopub.status.idle": "2020-05-11T06:24:02.976954Z", "shell.execute_reply.started": "2020-05-11T06:24:02.943814Z", "shell.execute_reply": "2020-05-11T06:24:02.976028Z"}
df_results[lambda df: (df.index.get_level_values(1) == 'Brazil')].sort_values('r_score', ascending=False).head(40)


# %% execution={"iopub.execute_input": "2020-05-10T20:12:10.928882Z", "iopub.status.busy": "2020-05-10T20:12:10.928684Z", "iopub.status.idle": "2020-05-10T20:12:11.025983Z", "shell.execute_reply": "2020-05-10T20:12:11.025202Z", "shell.execute_reply.started": "2020-05-10T20:12:10.928865Z"}
def plot_predictions(df, df_results, col_1, col_2):
    n_rows = df.shape[0]

    arr_c1 = df[col_1].values
    arr_c2 = df[col_2].values
    arr_pred = df_results.loc[(col_1, col_2), 'predictions']
    arr_pred = np.pad(arr_pred.flatten(), (0, n_rows-arr_pred.shape[0]),
                      constant_values=np.nan)
    df_ = pd.DataFrame([arr_c1, arr_c2, arr_pred]).T
    df_.columns = [col_1, col_2, 'Predicted']
    fig = df_.iplot(asFigure=True)
    display(HTML(fig.to_html()))

plot_predictions(df, df_results, 'India', 'Brazil')

# %%
