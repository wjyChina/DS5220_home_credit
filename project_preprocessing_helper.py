import pandas as pd
import gc


def df_agg(df: pd.DataFrame, main_key, df_name):
    for col in df:
        if col != main_key and "SK_ID" in col:
            df = df.drop(columns=col)

    df_id = df[main_key]
    numberic_df = df.select_dtypes('number')
    categorical_df = pd.get_dummies(df.select_dtypes('object'))
    count = pd.DataFrame(df.groupby(main_key).size())
    count.columns = ['count']
    gc.enable()
    del df
    gc.collect()

    numberic_df[main_key] = df_id
    stats = ['mean', 'max', 'min', 'sum']
    agg = numberic_df.groupby(main_key).agg(stats)
    # agg = agg.reset_index()
    # print(agg.head(3))
    numberic_df = rename(agg, main_key, df_name, stats)

    categorical_df[main_key] = df_id
    stats = ['mean']
    agg = categorical_df.groupby(main_key).agg(stats)
    # agg = agg.reset_index()
    categorical_df = rename(agg, main_key, df_name, stats)

    numberic_df = count.merge(numberic_df, right_index=True, left_on=main_key, how='outer')

    agg = numberic_df.merge(categorical_df, right_index=True, left_on=main_key, how='outer')
    gc.enable()
    del numberic_df, categorical_df
    gc.collect()
    return agg


def rename(df: pd.DataFrame, main_key, df_name, stats):
    columns = []
    for val in df.columns.levels[0]:
        if val == main_key:
            continue
        for stat in stats:
            columns.append("{}_{}_{}".format(df_name, val, stat))
    df.columns = columns
    return df