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

# %% execution={"iopub.status.busy": "2020-05-11T16:06:37.924277Z", "iopub.execute_input": "2020-05-11T16:06:37.924626Z", "iopub.status.idle": "2020-05-11T16:06:37.928441Z", "shell.execute_reply.started": "2020-05-11T16:06:37.924605Z", "shell.execute_reply": "2020-05-11T16:06:37.927918Z"}
import pandas as pd
import cufflinks as cf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from functools import partial
from zipfile import ZipFile
from glob import glob

from IPython.display import HTML, display

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)

# %% execution={"iopub.status.busy": "2020-05-11T16:04:54.106492Z", "iopub.execute_input": "2020-05-11T16:04:54.106707Z", "iopub.status.idle": "2020-05-11T16:04:54.109691Z", "shell.execute_reply.started": "2020-05-11T16:04:54.106687Z", "shell.execute_reply": "2020-05-11T16:04:54.109155Z"}
url_cases = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
             "csse_covid_19_data/csse_covid_19_time_series/"
             "time_series_covid19_confirmed_global.csv")
url_death = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
             "csse_covid_19_data/csse_covid_19_time_series/"
             "time_series_covid19_deaths_global.csv")
url_pop = ("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv")

# %%
# !unzip help

# %% execution={"iopub.status.busy": "2020-05-11T16:17:25.023377Z", "iopub.execute_input": "2020-05-11T16:17:25.023795Z", "iopub.status.idle": "2020-05-11T16:17:26.450974Z", "shell.execute_reply.started": "2020-05-11T16:17:25.023757Z", "shell.execute_reply": "2020-05-11T16:17:26.448008Z"}
# Getting the population information
# !wget {url_pop} -O pop.zip
# !unzip -o pop.zip -d pop_csv
# Load the population file
df_pop_ = pd.read_csv(glob('pop_csv/API_SP.POP.TOTL*.csv')[0],  skiprows=4)
df_pop = df_pop_[['Country Name', '2018']].set_index('Country Name')
# Delete the files downloaded
# !rm -r pop_csv

# %% execution={"iopub.status.busy": "2020-05-11T15:50:37.879460Z", "iopub.execute_input": "2020-05-11T15:50:37.880059Z", "iopub.status.idle": "2020-05-11T15:50:38.685680Z", "shell.execute_reply.started": "2020-05-11T15:50:37.880003Z", "shell.execute_reply": "2020-05-11T15:50:38.684529Z"}
df_death = pd.read_csv(death)
df_cases = pd.read_csv(cases)


# %% execution={"iopub.status.busy": "2020-05-11T16:21:48.942045Z", "iopub.execute_input": "2020-05-11T16:21:48.942250Z", "iopub.status.idle": "2020-05-11T16:21:48.982072Z", "shell.execute_reply.started": "2020-05-11T16:21:48.942232Z", "shell.execute_reply": "2020-05-11T16:21:48.981488Z"}
def iplot_nb(self, *args, **kwargs):
    fig = self.iplot(asFigure=True, *args, **kwargs)
    display(HTML(fig.to_html()))
    

pd.DataFrame.iplot_nb = iplot_nb

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

# %% execution={"iopub.status.busy": "2020-05-11T16:25:32.880337Z", "iopub.execute_input": "2020-05-11T16:25:32.880589Z", "iopub.status.idle": "2020-05-11T16:25:32.889061Z", "shell.execute_reply.started": "2020-05-11T16:25:32.880571Z", "shell.execute_reply": "2020-05-11T16:25:32.888485Z"}
df_pop.sort_values('2018', ascending=True).head(40)

# %% execution={"iopub.status.busy": "2020-05-11T11:28:46.210991Z", "iopub.execute_input": "2020-05-11T11:28:46.211226Z", "iopub.status.idle": "2020-05-11T11:28:46.216577Z", "shell.execute_reply.started": "2020-05-11T11:28:46.211208Z", "shell.execute_reply": "2020-05-11T11:28:46.215930Z"}
import matplotlib.pyplot as plt

def plot_us_br(arr_us, arr_br, arr_br_pred):
    plt.plot(arr_us[:arr_br.shape[0]], arr_br)
    plt.plot(arr_us, arr_br_pred, '--')
    plt.xlabel('US')
    plt.ylabel('Brazil')
    plt.grid(True)

arr_us = df['US'].dropna().values.reshape(-1, 1)
arr_br = df['Brazil'].dropna().values.reshape(-1, 1)

lr = LinearRegression()

# %% execution={"iopub.status.busy": "2020-05-11T15:48:22.322452Z", "iopub.execute_input": "2020-05-11T15:48:22.322688Z", "iopub.status.idle": "2020-05-11T15:48:22.466810Z", "shell.execute_reply.started": "2020-05-11T15:48:22.322671Z", "shell.execute_reply": "2020-05-11T15:48:22.466269Z"}
weigths = np.geomspace(1, 1000, arr_br.shape[0])
# weigths[-1] = 200
lr.fit(arr_us[:arr_br.shape[0]], arr_br, sample_weight=weigths)
arr_br_pred = lr.predict(arr_us)

print(lr.score(arr_us[:arr_br.shape[0]], arr_br))
plot_us_br(arr_us, arr_br, arr_br_pred)

# %% execution={"iopub.status.busy": "2020-05-11T11:26:16.790283Z", "iopub.execute_input": "2020-05-11T11:26:16.791638Z", "iopub.status.idle": "2020-05-11T11:26:16.950017Z", "shell.execute_reply.started": "2020-05-11T11:26:16.791519Z", "shell.execute_reply": "2020-05-11T11:26:16.949501Z"}
lr.fit(arr_us[:arr_br.shape[0]], arr_br)
arr_br_pred = lr.predict(arr_us)
plot_us_br

# %% execution={"iopub.status.busy": "2020-05-11T11:20:24.719023Z", "iopub.execute_input": "2020-05-11T11:20:24.719794Z", "iopub.status.idle": "2020-05-11T11:20:24.870903Z", "shell.execute_reply.started": "2020-05-11T11:20:24.719723Z", "shell.execute_reply": "2020-05-11T11:20:24.870397Z"}
ax = df[['Brazil', 'US']].dropna().set_index('US').rename(columns={'Brazil': 'N. of deaths'}).plot(figsize=(8,8))
ax.set_ylabel('Brazil')
ax.grid(True)
# ax.axis('equal')

# %% execution={"iopub.status.busy": "2020-05-11T16:29:33.126148Z", "iopub.execute_input": "2020-05-11T16:29:33.126997Z", "iopub.status.idle": "2020-05-11T16:29:33.174221Z", "shell.execute_reply.started": "2020-05-11T16:29:33.126919Z", "shell.execute_reply": "2020-05-11T16:29:33.173596Z"}
df

# %% execution={"iopub.status.busy": "2020-05-11T16:27:46.807687Z", "iopub.execute_input": "2020-05-11T16:27:46.808496Z", "iopub.status.idle": "2020-05-11T16:27:46.828200Z", "shell.execute_reply.started": "2020-05-11T16:27:46.808389Z", "shell.execute_reply": "2020-05-11T16:27:46.827120Z"}
df_pop

# %% execution={"iopub.status.busy": "2020-05-11T16:39:30.353989Z", "iopub.execute_input": "2020-05-11T16:39:30.354544Z", "iopub.status.idle": "2020-05-11T16:39:32.794499Z", "shell.execute_reply.started": "2020-05-11T16:39:30.354493Z", "shell.execute_reply": "2020-05-11T16:39:32.793888Z"}
from multiprocessing import Pool

min_diff = 7
all_columns = df.columns
results = {}

# Ignore countries with less than 1M 
countries = [c for c in df.columns if c not in 
             df_pop[lambda df_: df_['2018'] < 10**6].index]

# def compute_lr_parallel(df_covid, min_df=7):
def compute_lr(country, df_covid):
    """        
    Params:
        country: It's  a tuple with the index of the 
            country and the country
    """
    i, col_1 = country
    results = {}
    
    for col_2 in df_covid.columns[i+1:]:

        x = df_covid[col_1].dropna().values
        y = df_covid[col_2].dropna().values
      
        # Keep the largest array in x
        if x.shape[0] < y.shape[0]:
            x, y = y, x
            x_label, y_label = col_2, col_1
        else:
            x_label, y_label = col_1, col_2

        x, y = x.reshape(-1, 1), y.reshape(-1, 1)

        if x.shape[0] - y.shape[0] > min_diff:
            lr = LinearRegression()
            # The weights increase linearly from 1 to 2
            weights = np.linspace(1, 1, y.shape[0])
            lr.fit(x[:y.shape[0]], y, weights)

            results[(x_label, y_label)] = dict(
                    lr_model=lr, r_score=lr.score(x[:y.shape[0]], y),
                    predictions=lr.predict(x),
                    len_x=x.shape[0],
                    len_y=y.shape[0],
            )
            
    return results




compute_lr_parallel = partial(compute_lr, df_covid=df[countries])

with Pool(8) as pool:
    results = {}
    for res_dict in tqdm(pool.imap_unordered(compute_lr_parallel,
                                             enumerate(countries)), 
                         total=df.shape[0]):
        results.update(res_dict)
        

# for i, col_1 in enumerate(tqdm(all_columns)):
#     for col_2 in all_columns[i+1:]:

#         x = df[col_1].dropna().values
#         y = df[col_2].dropna().values

#         # Keep the largest array in x
#         if x.shape[0] < y.shape[0]:
#             x, y = y, x
#             x_label, y_label = col_2, col_1
#         else:
#             x_label, y_label = col_1, col_2

#         x, y = x.reshape(-1, 1), y.reshape(-1, 1)

#         if x.shape[0] - y.shape[0] > min_diff:
#             lr = LinearRegression()
#             # The weights increase linearly from 1 to 2
#             weights = np.linspace(1, 1, y.shape[0])
#             lr.fit(x[:y.shape[0]], y, weights)

#             results[(x_label, y_label)] = dict(
#                 lr_model=lr, r_score=lr.score(x[:y.shape[0]], y),
#                 predictions=lr.predict(x),
#                 len_x=x.shape[0],
#                 len_y=y.shape[0],
#             )

# %% execution={"iopub.status.busy": "2020-05-11T16:39:33.045365Z", "iopub.execute_input": "2020-05-11T16:39:33.046467Z", "iopub.status.idle": "2020-05-11T16:39:33.127107Z", "shell.execute_reply.started": "2020-05-11T16:39:33.046260Z", "shell.execute_reply": "2020-05-11T16:39:33.126271Z"}
df_results = pd.DataFrame.from_dict(results, orient='index')

# %% execution={"iopub.status.busy": "2020-05-11T16:39:55.139597Z", "iopub.execute_input": "2020-05-11T16:39:55.139813Z", "iopub.status.idle": "2020-05-11T16:39:55.143234Z", "shell.execute_reply.started": "2020-05-11T16:39:55.139794Z", "shell.execute_reply": "2020-05-11T16:39:55.142743Z"}
df_results.shape

# %% execution={"iopub.status.busy": "2020-05-11T16:39:35.236752Z", "iopub.execute_input": "2020-05-11T16:39:35.236967Z", "iopub.status.idle": "2020-05-11T16:39:35.415298Z", "shell.execute_reply.started": "2020-05-11T16:39:35.236948Z", "shell.execute_reply": "2020-05-11T16:39:35.414763Z"}
df_results[lambda df: (df.index.get_level_values(1) == 'Brazil')].sort_values('r_score', ascending=False).head(40)

# %% execution={"iopub.status.busy": "2020-05-11T16:41:29.184699Z", "iopub.execute_input": "2020-05-11T16:41:29.185010Z", "iopub.status.idle": "2020-05-11T16:41:29.195118Z", "shell.execute_reply.started": "2020-05-11T16:41:29.184983Z", "shell.execute_reply": "2020-05-11T16:41:29.194325Z"}
df_results.r_score[lambda s: s > 0.95]#.plot.hist(bins=30)


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
