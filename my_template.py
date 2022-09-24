import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

def tot_exec_time(time_start):
    print("-"*50)
    time_end = time.time()
    time_c = time_end - time_start
    print(f"time cost: {time_c:.3f} s")

def added_section_separate_line():
    print("-"*30)

## column type
def str_column_to_float(dataset, column):
    for r in dataset:
        r[column] = float(r[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, val in enumerate(unique):
        lookup[val] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

## describe data
def describe_class_distribution(df, class_col):
    class_counts = df.groupby(class_col).size()
    return class_counts

def describe_features_correlation(df, corr_method='pearson'):
    pd.set_option('display.width', 100)
    pd.set_option('precision', 3)
    correlations = df.corr(method=corr_method)
    return correlations

def describe_skew_unvariate_distribution(df, column_list):
    data = df[column_list]
    skew = df.skew()   # result: positive->right skew, negative->left skew
    print(skew)

## visualize data
def visualize_features_hist(df, column_list):
    data = df[column_list]
    print(f"Histogram of features:")
    data.hist()
    plt.show()

def visualize_countplot(df, column_list):
    sns.set_theme(style="darkgrid")
    for col in column_list:
        ax = sns.countplot(x=col, data=df)
        plt.show()

def visualize_density_plot(df, column_list, layout=(3,3), legend=False):
    data = df[column_list]
    print(f"Density Plots of features:")
    data.plot(kind='density', subplots=True, layout=layout, sharex=False, legend=legend)
    plt.show()

def visualize_box_whisker_plot(df, column_list, subplots=True, layout=(3,3), legend=False, title=''):
    data = df[column_list]
    print(f"Box-whisker Plots of features:")
    #layout_len = math.ceil(math.sqrt(len(column_list)))
    if subplots==True:
        data.plot(kind='box', subplots=True, layout=layout, sharex=False, legend=legend)
        plt.show()
    else:
        data.plot(kind='box', subplots=False, sharex=False, legend=legend, xticks=[])
        plt.title(title)
        plt.show()

def visualize_correlation_matrix(df, column_list, features_show=False, annot=True, cmap='RdYlGn'):
    data = df[column_list]
    print(f"Correlation matrix of features:")
    correlations = data.corr()
    sns.heatmap(correlations, annot=annot, cmap=cmap, vmin=-1, vmax=1)
    plt.show()

def visualize_scatter_plot(df, column_list):
    data = df[column_list]
    print(f"Scatter Plots of features:")
    pd.plotting.scatter_matrix(data)
    plt.show()

def visualize_subplots_by_diff_df(df_list, set_subtitle_list, kind='barh', nrows=2, ncols=2, sharex=True, sharey=True, rotat_deg=15, scientific_notate=[None, None]):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=False)
    arrang_idx = 0
    arrang_idy = 0
    for idx, df_sub in enumerate(df_list):
        df_sub.plot(kind=kind, ax=axes[arrang_idx, arrang_idy])
        axes[arrang_idx, arrang_idy].set_title(set_subtitle_list[idx])
        if scientific_notate[0]==False:
            axes[arrang_idx, arrang_idy].get_xaxis().get_major_formatter().set_scientific(False)
        elif scientific_notate[0]==True:
            axes[arrang_idx, arrang_idy].get_xaxis().get_major_formatter().set_scientific(True)
        if scientific_notate[1]==False:
            axes[arrang_idx, arrang_idy].get_yaxis().get_major_formatter().set_scientific(False)
        elif scientific_notate[1]==True:
            axes[arrang_idx, arrang_idy].get_yaxis().get_major_formatter().set_scientific(True)
        axes[arrang_idx, arrang_idy].tick_params(labelrotation=rotat_deg)
        if arrang_idy < ncols-1:
            arrang_idy += 1
        else:
            arrang_idx += 1
            arrang_idy = 0
    plt.show()

## data cleaning
def detect_column_only_single_value(df):
    counts = df.nunique()
    to_del = [i for i, v in enumerate(counts) if v==1]
    print(f"Record columns to delete (only single value): {to_del}")

def detect_column_very_few_value(df):
    counts = df.nunique()
    to_del = [i for i,v in enumerate(counts) if (float(v)/df.shape[0]*100)<1]
    print(f"Record columns to delete (very few value): {to_del}")

def remove_column(df, column_list):
    for col in column_list:
        if col in df.columns:
            df.drop([col], axis=1, inplace=True)
        else:
            continue
    return df

def remove_duplicate_rows(df):
    dups = df.duplicated()
    if dups.any():
        print(df[dups])
    print("-"*20)
    print(f"Before duplicate removed: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"After duplicate removed: {df.shape}")

def detect_outlier_over3std(df, col):
    data_mean, data_std = np.mean(df[col]), np.std(df[col])
    cut_off = data_std * 3
    lower, upper = data_mean-cut_off, data_mean+cut_off
    # identify outliers
    outliers = [x for x in df[col] if x > upper or x < lower]
    # removed outliers
    outliers_removed = [x for x in df[col] if x>=lower or x<=upper]
    print(f"Identified outliers: {len(outliers)}, Non-outlier observation: {len(outliers_removed)}")

def detect_outlier_quartile(df, col):
    q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    iqr = q75 - q25
    print(f"Percentile: 25th = {q25:.2f}, 75th = {q75:.2f}, IQR = {iqr:.3f}")
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25-cut_off, q75+cut_off
    # identify outliers
    outliers = [x for x in df[col] if x > upper or x < lower]
    # removed outliers
    outliers_removed = [x for x in df[col] if x <= upper or x >= lower]
    print(f"Identified outliers: {len(outliers)}, Non-outlier observations: {len(outliers_removed)}")

def detect_have_null_value(df):
    if df.isnull().values.any():
        print(f"There are null values in dataframe: ")
        print(df.isnull().sum())
    else:
        print("There is no null value in dataframe.")

def detect_zero_value_counts(df, column_list):
    print("Detect 0 value in these features:")
    for feature in column_list:
        print(f"> Count of zeros in Column {feature}: {(df[feature]==0).sum()}")

def remove_null_value(df):
    df.dropna(inplace=True)
    print(f"After remove null values: \n{df.isnull().sum()}")
    return df

def remove_outlier_over3std(df, col):
    data_mean, data_std = np.mean(df[col]), np.std(df[col])
    cut_off = data_std * 3
    lower, upper = data_mean-cut_off, data_mean+cut_off
    idx = df[((df[col] > upper) | (df[col] < lower))].index
    df = df.drop(idx).reset_index(drop=True)
    return df

## data transform
# rescale to range (0,1)
def transform_by_minmax_rescale(fit_data, transform_data, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range).fit(fit_data)
    rescaled_data = scaler.transform(transform_data)
    return rescaled_data

# useful for Gaussian distribution with differing means and deviations to standard Gaussian distribution (mean=0, stdev=1)
def transform_by_standardize_rescale(fit_data, transform_data):
    scaler = StandardScaler().fit(fit_data)
    rescaled_data = scaler.transform(transform_data)
    return rescaled_data

# useful for sparse data (a lot of zeros) with attributes of varying scales, algorithms such as nueral networks, knn
def transform_by_normalize_rescale(fit_data, transform_data):
    scaler = Normalizer().fit(fit_data)
    rescaled_data = scaler.transform(transform_data)
    return rescaled_data

# binarize value using threshold
def transform_by_binarize(fit_data, transform_data, threshold=0.0):
    scaler = Binarizer(threshold=threshold).fit(fit_data)
    rescaled_data = scaler.transform(transform_data)
    return rescaled_data

## baseline model
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)   # 找出比例最大的分類，直接作為所有驗證集的baseline預測結果
    prediction = [prediction for i in range(len(test))]
    return prediction

def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values))   # 計算所有結果的平均值作為驗證集的baseline預測結果
    prediction = [prediction for i in range(len(test))]
    return prediction

## feature selection
def select_features(X_train, y_train, X_test, score_func=f_classif, k_num='all'):
    fs = SelectKBest(score_func=score_func, k=k_num)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

## evaluate metrics
def rmse_metric(actual, predicted):
    sum_error = 0
    for i in range(len(actual)):
        predict_error = predicted[i] - actual[i]
        sum_error += (predict_error**2)
    mean_error = sum_error / float(len(actual))
    rmse = math.sqrt(mean_error)
    return rmse