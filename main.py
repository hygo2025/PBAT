import pandas as pd

if __name__ == '__main__':


    df = pd.read_parquet(
        '/Users/hygo/s3/n-recommendations/recs/GRUPOZAP/archive/events/2026/05/19/part-00000-f2c689c5-7806-4342-8bf4-554df62efa03-c000.snappy.parquet')
    print(df.columns.tolist())
    print(df.dtypes)
    print(df.head(10).to_string())
    print("---")
    for col in df.columns:
        n = df[col].nunique()
        print(f"{col}: {n} unique")
        if n < 30: print(f"  {df[col].unique().tolist()}")

