"""Convert GRUPOZAP parquet events (7 days) to PBAT-compatible TXT format."""
import pandas as pd
from pathlib import Path

DATA_ROOT = Path('/Users/hygo/s3/n-recommendations/recs/GRUPOZAP/archive/events/2026/05')
DAYS = ['14', '15', '16', '17', '18', '19', '20']
OUTPUT = Path(__file__).resolve().parent.parent / 'data' / 'grupozap.txt'

dfs = []
for day in DAYS:
    day_path = DATA_ROOT / day
    parquets = list(day_path.glob('*.parquet'))
    print(f"Day {day}: {len(parquets)} files")
    df = pd.read_parquet(day_path, columns=['anonymous_id', 'listing_id', 'event_type', 'captured_at'])
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.rename(columns={'anonymous_id': 'uid', 'listing_id': 'sid', 'event_type': 'behavior', 'captured_at': 'timestamp'})
df['behavior'] = df['behavior'].str.lower()
df['timestamp'] = df['timestamp'] / 1000.0
df = df.sort_values(['uid', 'timestamp']).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, sep='\t', header=False, index=False)

print(f"\n=== Statistics ===")
print(f"Total interactions: {len(df)}")
print(f"Unique users: {df['uid'].nunique()}")
print(f"Unique items: {df['sid'].nunique()}")
print(f"Behaviors: {df['behavior'].value_counts().to_dict()}")
print(f"Output: {OUTPUT}")
