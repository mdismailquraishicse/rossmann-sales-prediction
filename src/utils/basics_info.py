import pandas as pd



def display_basic_info(df:pd.DataFrame):

    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"duplicates: {df.duplicated().sum()}")
    print()
    print("INFO:")
    df.info()
    print()
    print("NULLS:")
    print(df.isna().sum())
    print()
    print("UNIQUE")
    print(df.nunique())
    print()
    print("DESCRIBE:")
    print(df.describe())


def extend_date(df:pd.DataFrame, colname:str):

    df[colname] = pd.to_datetime(df[colname])
    df["year"] = df[colname].dt.year
    df["month"] = df[colname].dt.month
    df["day"] = df[colname].dt.day
    df["week_of_year"] = df[colname].dt.isocalendar().week
    return df


def distance_bucket(df: pd.DataFrame):

    bins = [0, 1000, 5000, 10000, float("inf")]
    labels = ["very_close", "close", "medium", "far"]
    df["CompetitionDistanceBucket"] = pd.cut(
        df["CompetitionDistance"],
        bins=bins,
        labels=labels
    )
    return df