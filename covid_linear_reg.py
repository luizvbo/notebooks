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

from multiprocessing import Pool


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
        df.apply(_pad_days, raw=True)
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


# %% execution={"iopub.status.busy": "2020-05-11T17:04:36.679984Z", "iopub.execute_input": "2020-05-11T17:04:36.680202Z", "iopub.status.idle": "2020-05-11T17:04:39.481860Z", "shell.execute_reply.started": "2020-05-11T17:04:36.680184Z", "shell.execute_reply": "2020-05-11T17:04:39.480482Z"}
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
            pred = lr.predict(x)

            results[(x_label, y_label)] = dict(
                    lr_model=lr, r_score=lr.score(x[:y.shape[0]], y),
                    predicted=lr.predict(x),
                    x=x,
                    y=y,
            )
            
    return results

min_diff = 7
all_columns = df.columns
results = {}

# Ignore countries with less than 1M 
countries = [c for c in df.columns if c not in 
             df_pop[lambda df_: df_['2018'] < 10**6].index]

compute_lr_parallel = partial(compute_lr, df_covid=df[countries])

with Pool(8) as pool:
    results = {}
    for res_dict in tqdm(pool.imap(compute_lr_parallel, enumerate(countries)), 
                         total=df.shape[0]):
        results.update(res_dict)
df_results = pd.DataFrame.from_dict(results, orient='index')

# %% execution={"iopub.status.busy": "2020-05-11T17:14:06.452581Z", "iopub.execute_input": "2020-05-11T17:14:06.452782Z", "iopub.status.idle": "2020-05-11T17:14:06.647023Z", "shell.execute_reply.started": "2020-05-11T17:14:06.452765Z", "shell.execute_reply": "2020-05-11T17:14:06.646500Z"}
df_ = df_results[lambda df: (df.index.get_level_values(1) == 'Brazil') &
                 (df.r_score > 0.95)].sort_values('r_score', ascending=False).head(10)


# %% execution={"iopub.status.busy": "2020-05-11T17:26:27.733957Z", "iopub.execute_input": "2020-05-11T17:26:27.734248Z", "iopub.status.idle": "2020-05-11T17:26:29.092966Z", "shell.execute_reply.started": "2020-05-11T17:26:27.734226Z", "shell.execute_reply": "2020-05-11T17:26:29.092404Z"}
from scipy.stats import pearsonr

# lambda row: pearsonr(row['x'][:row['y'].shape[0]], row['y'])

df_ = (
    df_results
    .assign(rho=lambda df: df.apply(lambda row: pearsonr(row['x'][:row['y'].shape[0]].flatten(), 
                                                         row['y'].flatten())[0], axis=1))
    .assign(rho_sq=lambda df: df.rho**2)
    .assign(diff=lambda df: abs(df.rho_sq-df.r_score))
)


# %% execution={"iopub.status.busy": "2020-05-11T17:38:24.549622Z", "iopub.execute_input": "2020-05-11T17:38:24.550518Z", "iopub.status.idle": "2020-05-11T17:38:25.397595Z", "shell.execute_reply.started": "2020-05-11T17:38:24.550430Z", "shell.execute_reply": "2020-05-11T17:38:25.396991Z"}
def plot_candidates(df_candidates, nrows=4, ncols=2, over_days=True):
    fig, axs = plt.subplots(nrows, ncols)
    df_ = df_candidates.head(nrows * ncols)
    for (i, row), ax in zip(df_.iterrows(), axs.flatten()):
        if over_days:
            ax.plot(row['x'], label=i[0])
            ax.plot(row['y'], label=i[1])
            ax.plot(row['predicted'], '--', label=f"{i[1]} (predicted)")
            ax.title.set_text("$r^2={:.3f}$".format(row['r_score']))
        else:
            ax.plot(row['x'][:row['y'].shape[0]], row['y'], label='True value')
            ax.plot(row['x'], row['predicted'], '--', label='Predicted')
            ax.set_xlabel(i[0])
            ax.set_ylabel(i[1])
        
        ax.grid(True)
        legend = ax.legend(loc='upper left')#, shadow=True, fontsize='x-large')

    fig.set_size_inches(10, 15)

df_ = df_results[lambda df: (df.index.get_level_values(1) == 'Brazil')].sort_values('r_score', ascending=False)
plot_candidates(df_, over_days=False)


# %%
def plot_candidates(df_candidates, nrows=4, ncols=2):
    fig, axs = plt.subplots(nrows, ncols)
    df_ = df_candidates.head(nrows * ncols)
    for (i, row), ax in zip(df_.iterrows(), axs.flatten()):
        ax.plot(row['x'], label=i[0])
        ax.plot(row['y'], label=i[1])
        ax.plot(row['predicted'], '--', label=f"{i[1]} (predicted)")
        ax.title.set_text("$r^2={:.3f}$".format(row['r_score']))
        
        ax.grid(True)
        legend = ax.legend(loc='upper left')#, shadow=True, fontsize='x-large')

    fig.set_size_inches(10, 15)

df_ = df_results[lambda df: (df.index.get_level_values(1) == 'Brazil')].sort_values('r_score', ascending=False)
plot_candidates(df_)

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
