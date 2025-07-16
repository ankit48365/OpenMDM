import pandas as pd
import recordlinkage
import networkx as nx

# 5.1. Load and prepare your data
df = pd.read_csv('/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/Data_Set_1.csv')
df.reset_index(drop=True, inplace=True)
df.index.name = 'record_id'

# 5.2. Blocking
indexer = recordlinkage.Index()
indexer.block(['first_name','middle_name','last_name'])
candidate_pairs = indexer.index(df, df)
candidate_pairs = candidate_pairs[candidate_pairs.get_level_values(0) != candidate_pairs.get_level_values(1)]
print(f"🧮 Total candidate pairs after blocking: {len(candidate_pairs)}")

# 5.3. Pairwise similarity
compare = recordlinkage.Compare()
compare.string('first_name', 'first_name', method='jarowinkler', label='fn_sim')
compare.string('middle_name', 'middle_name', method='jarowinkler', label='mn_sim')
compare.string('last_name', 'last_name', method='jarowinkler', label='ln_sim')
compare.string('address', 'address', method='levenshtein', label='addr_sim')
# compare.string('city', 'city', method='jarowinkler', label='city_sim')
# compare.exact('zip_code', 'zip_code', label='zip_match')
# compare.string('phone', 'phone', method='damerau_levenshtein', label='phone_sim')
# compare.string('email', 'email', method='jarowinkler', label='email_sim')

features = compare.compute(candidate_pairs, df, df)
features['score'] = (
    features['fn_sim'] + features['mn_sim'] + features['ln_sim'] +
    features['addr_sim'] #+ features['city_sim'] + features['zip_match'] 
    # + features['phone_sim'] + features['email_sim']
) / 4 # this is total fields being compared

features['match_category'] = pd.cut(
    features['score'],
    bins=[0, 0.7, 0.88, 1.0],
    labels=['non_match', 'review', 'auto_merge']
)

# 5.4. Write summary function
def write_pairwise_summary(df, features, category, output_path):
    subset = features[features['match_category'] == category].reset_index()

    # Normalize pair direction (min/max)
    subset[['id_min', 'id_max']] = subset[['record_id_1', 'record_id_2']].apply(sorted, axis=1, result_type='expand')
    subset = subset.drop_duplicates(subset=['id_min', 'id_max'])

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'\n--- {category.upper()} RECORDS ---\n')
        for _, row in subset.iterrows():
            id1, id2 = row['record_id_1'], row['record_id_2']
            rec1 = df.loc[id1].to_dict()
            rec2 = df.loc[id2].to_dict()
            f.write(f"🔹 Record 1 ({id1}): {rec1}\n")
            f.write(f"🔸 Record 2 ({id2}): {rec2}\n")
            f.write(f"💡 Similarity Score: {round(row['score'], 4)} | Match Category: {category}\n")
            f.write('-' * 80 + '\n')


output_path = '/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/mdm_similarity_summary.txt'
write_pairwise_summary(df, features, 'auto_merge', output_path)
write_pairwise_summary(df, features, 'review', output_path)
# write_pairwise_summary(df, features, 'non_match', output_path)

# 5.5. 🎯 Cluster auto_merge pairs to remove reverse duplicates
auto_merge_pairs = features[features['match_category'] == 'auto_merge'].reset_index()
auto_merge_pairs[['id_min', 'id_max']] = auto_merge_pairs[['record_id_1', 'record_id_2']].apply(
    sorted, axis=1, result_type='expand'
)
auto_merge_pairs = auto_merge_pairs.drop_duplicates(subset=['id_min', 'id_max'])


# Build clusters using graph traversal
G = nx.Graph()
G.add_edges_from(auto_merge_pairs[['record_id_1', 'record_id_2']].values)
clusters = list(nx.connected_components(G))

# 5.5.1 This just takes care of single no match records 
# Identify all record IDs
all_ids = set(df.index)
# Get all clustered record IDs from auto_merge clusters
clustered_ids = set().union(*clusters)
# Identify singleton records: those not involved in any auto_merge cluster
singleton_ids = sorted(all_ids - clustered_ids)


# 5.6. Merge records in each cluster
def merge_cluster(df, cluster):
    members = list(cluster)
    records = df.loc[members]
    preferred = records[records['Original'] == 'Y']
    fallback = records[records['Original'] != 'Y']

    merged = {}
    for col in df.columns:
        if col == 'Original':
            continue
        val = (
            preferred[col].dropna().iloc[0]
            if not preferred[col].dropna().empty
            else fallback[col].dropna().iloc[0]
            if not fallback[col].dropna().empty
            else None
        )
        merged[col] = str(val).strip() if pd.notnull(val) else None

    merged['source_ids'] = members
    merged['merge_score'] = 1.0
    return pd.Series(merged)

singleton_goldens = []

for rid in singleton_ids:
    record = df.loc[rid].copy()
    clean_record = {col: str(record[col]).strip() if pd.notnull(record[col]) else None
                    for col in df.columns if col != 'Original'}
    clean_record['source_ids'] = [rid]
    clean_record['merge_score'] = 0.0
    singleton_goldens.append(clean_record)


golden_clusters = [merge_cluster(df, cluster) for cluster in clusters]
# golden_df = pd.DataFrame(golden_clusters)
# golden_df.to_csv('/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/golden_auto_records.csv', index=False)

singleton_goldens_series = [pd.Series(rec) for rec in singleton_goldens]
golden_df = pd.DataFrame(golden_clusters + singleton_goldens_series)

# golden_df = pd.DataFrame(golden_clusters + singleton_goldens)
golden_df.to_csv('/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/golden_auto_records.csv', index=False)



with open(output_path, 'a', encoding='utf-8') as f:
    f.write('\n--- NON_MATCHED SINGLE RECORDS ---\n')
    for rid in singleton_ids:
        rec = df.loc[rid].to_dict()
        f.write(f"🧍 Record ({rid}): {rec}\n")
        f.write(f"💡 Similarity Score: 0.0 | Match Category: non_match (singleton)\n")
        f.write('-' * 80 + '\n')