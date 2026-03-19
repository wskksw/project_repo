import pandas as pd

df = pd.read_excel('cfpb.xlsx', header=1)

# Parse dates
df['_date'] = pd.to_datetime(df['Date received'], format='%m/%d/%y', errors='coerce')

# Filter: year >= 2025
df_recent = df[df['_date'].dt.year >= 2025]
print(f'Rows with year >= 2025: {len(df_recent)}')

# Filter: narrative more than 300 words
df_recent = df_recent[df_recent['Consumer complaint narrative'].notna()]
df_recent = df_recent.copy()
df_recent['_word_count'] = df_recent['Consumer complaint narrative'].str.split().str.len()
df_filtered = df_recent[df_recent['_word_count'] > 300]
print(f'Rows also with >300 words: {len(df_filtered)}')

# Sample 100
if len(df_filtered) >= 100:
    df_sample = df_filtered.sample(n=100, random_state=42)
else:
    df_sample = df_filtered
    print(f'WARNING: Only {len(df_filtered)} rows match criteria, taking all')

# Drop helper columns and save
df_sample = df_sample.drop(columns=['_date', '_word_count'])
df_sample.to_excel('cfpb_100_samples.xlsx', index=False)
print(f'Saved {len(df_sample)} rows to cfpb_100_samples.xlsx')
