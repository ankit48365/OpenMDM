

import pandas as pd

# 5.1. Prepare Your DataFrame
# Read CSV file into a DataFrame & first row as header

# df.rename(columns={'index':'record_id'}, inplace=True)
# df.set_index('record_id', inplace=True)

df = pd.read_csv('/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/Data_Set_2_20_modified.csv', header=0)
# Add record_id directly as an index from the position
df.reset_index(drop=True, inplace=True)
df.index.name = 'record_id'


# Display the DataFrame
# print(df)

# 5.2. Blocking (Reduce Comparisons) We block on last_name + zip_code (if present) to cut down candidate pairs.
# Ensure a unique index for linkage
import recordlinkage
indexer = recordlinkage.Index()
indexer.block(['last_name', 'zip_code'])
candidate_pairs = indexer.index(df, df)

# 5.3. Pairwise Comparison (Feature Generation) Compute similarity scores on key fields.
compare = recordlinkage.Compare()

compare.string('first_name', 'first_name', method='jarowinkler', label='fn_sim')
compare.string('middle_name', 'middle_name', method='jarowinkler', label='mn_sim')
compare.string('last_name', 'last_name', method='jarowinkler', label='ln_sim')
compare.string('address', 'address', method='levenshtein', label='addr_sim')
compare.string('city', 'city', method='jarowinkler', label='city_sim')
compare.exact('zip_code', 'zip_code', label='zip_match')
compare.string('phone', 'phone', method='damerau_levenshtein', label='phone_sim')
compare.string('email', 'email', method='jarowinkler', label='email_sim')

features = compare.compute(candidate_pairs, df, df)

# 5.4. Classification & Thresholding Use a simple rule: sum of normalized similarities above a threshold signals a match

# Normalize exact match to 1/0, others range [0,1]
features['score'] = (
    features['fn_sim'] +
    features['mn_sim'] +
    features['ln_sim'] +
    features['addr_sim'] +
    features['city_sim'] +
    features['zip_match'] +
    features['phone_sim'] +
    features['email_sim']
) / 8

matches = features[features['score'] >= 0.8].drop(columns='score')
matched_pairs = matches.index.tolist()

# 5.5. Clustering into Entity Groups Build a graph of link pairs and extract connected components as clusters.
import networkx as nx

G = nx.Graph()
G.add_edges_from(matched_pairs)
clusters = list(nx.connected_components(G))

# Step 6. Survivorship & Merge, For each cluster, we’ll apply field-level rules to build a “golden record.
# 6.1. Define Survivorship Rules
# - Original record wins: prefer the row where Original == 'Y'.
# - Fallback by completeness:
# - Non-null > null
# - Longer strings (for address)
# - Mode (for city, state, email, zip)

def choose_value(series):
    # 1. Any Original == 'Y'?
    orig = series.loc[df.loc[series.index, 'Original'] == 'Y']
    if not orig.dropna().empty:
        return orig.dropna().iloc[0]
    # 2. Else, pick the mode (or longest for strings)
    non_null = series.dropna()
    if non_null.empty:
        return None
    if series.name in ['address']:
        return non_null.loc[non_null.str.len().idxmax()]
    return non_null.mode().iloc[0]


# 6.2. Build Golden Records

golden_records = []

for cluster in clusters:
    group = df.loc[list(cluster)]  # extract all records in the current cluster
    merged = {col: choose_value(group[col]) for col in group.columns if col != 'Original'}  # apply field-level merge
    golden_records.append(merged)

master_df = pd.DataFrame(golden_records)


# golden_records = []
# for cluster in clusters:
#     group = df.loc[list(cluster)]
#     merged = {col: choose_value(group[col]) for col in cols if col != 'Original'}
#     golden_records.append(merged)

# master_df = pd.DataFrame(golden_records)

print(master_df)