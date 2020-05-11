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

# %% [markdown]
# # Predicting COVID-19 Deaths by Similarity
#
# I have been seen some discussion about epidemiologic models to predict the number of cases and deaths by COVID-19, which made me give some thought about the subject.
#
# Since we have countries in different stages of the pandemic, my hypothesis was that we could use information from countries in advanced stages to predict information for countries at the beginning of the crisis.
#
# Considering the number of deaths by COVID-19 registered, we can align the data for all countries, such that the day the first death by COVID-19 was registered for each country coincides with the origin.
#
# With the data shifted, we can compute the correlation between each pair of countries and then find those pairs with high correlation. We can fit a linear regression to each of these pairs considering the country with more data as the independent variable and the one with fewer data as the response variable.
#
# We can then use the data for the country with more data to predict the number of deaths for the other country. For, instance, for Brazil, the 8 highest correlated countries are Canada, USA, Egypt, Japan, Argentina, UK, Italy and Iran (all of them with correlation > 0.95).
#
# I used the data provided by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) in their [GitHub repository](https://github.com/CSSEGISandData/COVID-19).

# %% execution={"iopub.status.busy": "2020-05-11T18:58:09.905534Z", "iopub.execute_input": "2020-05-11T18:58:09.906362Z", "iopub.status.idle": "2020-05-11T18:58:10.741596Z", "shell.execute_reply.started": "2020-05-11T18:58:09.906300Z", "shell.execute_reply": "2020-05-11T18:58:10.740943Z"}
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from functools import partial
from glob import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt


# %% execution={"iopub.status.busy": "2020-05-11T18:58:10.742459Z", "iopub.execute_input": "2020-05-11T18:58:10.742652Z", "iopub.status.idle": "2020-05-11T18:58:10.745437Z", "shell.execute_reply.started": "2020-05-11T18:58:10.742633Z", "shell.execute_reply": "2020-05-11T18:58:10.744833Z"}
# URL to the CSSE repository
url_covid_death = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"
                   "master/csse_covid_19_data/csse_covid_19_time_series/"
                   "time_series_covid19_deaths_global.csv")
# URL to the population data from Worldbank
url_pop = ("http://api.worldbank.org/v2/en/indicator/"
           "SP.POP.TOTL?downloadformat=csv")


# %% execution={"iopub.status.busy": "2020-05-11T18:58:10.860735Z", "iopub.execute_input": "2020-05-11T18:58:10.861635Z", "iopub.status.idle": "2020-05-11T18:58:10.898437Z", "shell.execute_reply.started": "2020-05-11T18:58:10.861551Z", "shell.execute_reply": "2020-05-11T18:58:10.897608Z"}
def get_same_origin(df):
    """

    """
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


def compute_lr(country, df_covid, min_diff=7):
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
                lr_model=lr,
                r_score=lr.score(x[:y.shape[0]], y),
                predicted=lr.predict(x),
                x=x, y=y,
            )

    return results

def plot_candidates(df_candidates, nrows=4, ncols=2, over_days=True, figsize=(12, 15)):
    fig, axs = plt.subplots(nrows, ncols)
    df_ = df_candidates.head(nrows * ncols)
    for (i, row), ax in zip(df_.iterrows(), axs.flatten()):
        if over_days:
            ax.plot(row['x'], label=i[0])
            ax.plot(row['y'], label=i[1])
            ax.plot(row['predicted'], '--', label=f"{i[1]} (predicted)")
            ax.set_xlabel("Days since the first death by COVID-19")
            ax.set_ylabel("Number of deaths")
        else:
            ax.plot(row['x'][:row['y'].shape[0]], row['y'], label='True value')
            ax.plot(row['x'], row['predicted'], '--', label='Predicted')
            ax.set_xlabel(i[0])
            ax.set_ylabel(i[1])

        ax.grid(True)
        legend = ax.legend(title="$r^2={:.3f}$".format(row['r_score']),
                           loc='upper left')

    fig.set_size_inches(*figsize)
    return fig


# %% execution={"iopub.status.busy": "2020-05-11T18:58:11.644956Z", "iopub.execute_input": "2020-05-11T18:58:11.645974Z", "iopub.status.idle": "2020-05-11T18:58:14.226952Z", "shell.execute_reply.started": "2020-05-11T18:58:11.645875Z", "shell.execute_reply": "2020-05-11T18:58:14.225506Z"}
# Getting the population information
# !wget {url_pop} -O pop.zip
# !unzip -o pop.zip -d pop_csv
# Load the population file
df_pop_ = pd.read_csv(glob('pop_csv/API_SP.POP.TOTL*.csv')[0],  skiprows=4)
df_pop = df_pop_[['Country Name', '2018']].set_index('Country Name')
# Delete the files downloaded
# !rm -r pop_csv

# %% execution={"iopub.status.busy": "2020-05-11T18:58:14.229227Z", "iopub.execute_input": "2020-05-11T18:58:14.229597Z", "iopub.status.idle": "2020-05-11T18:58:14.677704Z", "shell.execute_reply.started": "2020-05-11T18:58:14.229561Z", "shell.execute_reply": "2020-05-11T18:58:14.676594Z"}
# Loading the data for the number of Deaths
df_death = pd.read_csv(url_covid_death)

# %% execution={"iopub.status.busy": "2020-05-11T18:58:14.679170Z", "iopub.execute_input": "2020-05-11T18:58:14.679466Z", "iopub.status.idle": "2020-05-11T18:58:14.732718Z", "shell.execute_reply.started": "2020-05-11T18:58:14.679440Z", "shell.execute_reply": "2020-05-11T18:58:14.732050Z"}
df = set_index(df_death.copy())

# Groupy territories per country
df = df.groupby(level=1, axis=1).sum()

# # Drop all-zeros columns
df = df[df.sum()[lambda s: s > 0].index]

# # Shift all series to the origin (first death)
df = get_same_origin(df)

# %% execution={"iopub.status.busy": "2020-05-11T18:58:14.783169Z", "iopub.execute_input": "2020-05-11T18:58:14.783391Z", "iopub.status.idle": "2020-05-11T18:58:17.879591Z", "shell.execute_reply.started": "2020-05-11T18:58:14.783373Z", "shell.execute_reply": "2020-05-11T18:58:17.878952Z"}
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

# %% execution={"iopub.status.busy": "2020-05-11T18:58:18.340275Z", "iopub.execute_input": "2020-05-11T18:58:18.340799Z", "iopub.status.idle": "2020-05-11T18:58:19.490409Z", "shell.execute_reply.started": "2020-05-11T18:58:18.340746Z", "shell.execute_reply": "2020-05-11T18:58:19.489746Z"}
df_brazil = df_results[lambda df: (df.index.get_level_values(1) == 'Brazil')].sort_values('r_score', ascending=False)
fig1 = plot_candidates(df_, over_days=True)

# %% execution={"iopub.status.busy": "2020-05-11T18:58:27.907924Z", "iopub.execute_input": "2020-05-11T18:58:27.909053Z", "iopub.status.idle": "2020-05-11T18:58:28.995842Z", "shell.execute_reply.started": "2020-05-11T18:58:27.908950Z", "shell.execute_reply": "2020-05-11T18:58:28.995325Z"}
fig2 = plot_candidates(df_brazil, over_days=False)

# %%
