"""mdm_dynamic.py : requires config_mdm.yml in same directory"""
import pandas as pd
import recordlinkage
import networkx as nx
import sqlite3
import yaml
from collections import Counter
from datetime import datetime

# variables
DB_PATH = 'D:\\mygit\\OpenMDM\\database\\'
DB_NAME = 'dbt_etl_dev.db'
TIMEIS = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
OUTPUT_PATH = 'D:\\mygit\\OpenMDM\\mdm\\output\\'
CONFIG_PATH = 'D:\\mygit\\OpenMDM\\mdm\\matching_identity\\config_mdm.yml'

# Load configuration from YAML
# with open('config_mdm.yml', 'r') as file:
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
blocking_columns = config['blocking']['columns']
similarity_configs = config['similarity']
thresholds = config['thresholds']
survivorship_rules = {rule['column']: rule['strategy'] for rule in config['survivorship']['rules']}

# 1. Load and prepare data
# conn = sqlite3.connect(r'D:\mygit\OpenMDM\database\dbt_etl_dev.db')
FULL_DB_PATH = f"{DB_PATH}{DB_NAME}"
conn = sqlite3.connect(FULL_DB_PATH)

df = pd.read_sql_query("SELECT * FROM main.slvr_personal_info", conn)
df.reset_index(drop=True, inplace=True)
df.index.name = 'record_id'
conn.close()

# 2. Blocking
indexer = recordlinkage.Index()
indexer.block(blocking_columns)
candidate_pairs = indexer.index(df, df)
candidate_pairs = candidate_pairs[candidate_pairs.get_level_values(0) != candidate_pairs.get_level_values(1)] # pylint: disable=line-too-long
print(f"🧮 Total candidate pairs after blocking: {len(candidate_pairs)}")

# 3. Pairwise similarity
compare = recordlinkage.Compare()
for sim in similarity_configs:
    column = sim['column']
    method = sim['method']
    if method == 'exact':
        compare.exact(column, column, label=f"{column}_match")
    else:
        compare.string(column, column, method=method, label=f"{column}_sim")

features = compare.compute(candidate_pairs, df, df)

# 4. Calculate score
similarity_labels = [f"{sim['column']}_{'match' if sim['method'] == 'exact' else 'sim'}" for sim in similarity_configs] # pylint: disable=line-too-long
features['score'] = features[similarity_labels].mean(axis=1)

# 5. Categorize based on thresholds
features['match_category'] = pd.cut(
    features['score'],
    bins=[0, thresholds['review'], thresholds['auto_merge'], 1.0],
    labels=['non_match', 'review', 'auto_merge']
)

# 6. Write summary function
def write_pairwise_summary(df, features, category, output_path):
    subset = features[features['match_category'] == category].reset_index()
    subset[['id_min', 'id_max']] = subset[['record_id_1', 'record_id_2']].apply(sorted, axis=1, result_type='expand') # pylint: disable=line-too-long
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

summary = f"detail_summary_{TIMEIS}.txt"
output_path = f"{OUTPUT_PATH}{summary}"
write_pairwise_summary(df, features, 'auto_merge', output_path)
write_pairwise_summary(df, features, 'review', output_path)
print(f"📊 MDM Summary Report saved here  {OUTPUT_PATH}{summary}")


# 7. Cluster auto_merge pairs
auto_merge_pairs = features[features['match_category'] == 'auto_merge'].reset_index()
auto_merge_pairs[['id_min', 'id_max']] = auto_merge_pairs[['record_id_1', 'record_id_2']].apply(
    sorted, axis=1, result_type='expand'
)
auto_merge_pairs = auto_merge_pairs.drop_duplicates(subset=['id_min', 'id_max'])

# 8. Build clusters
G = nx.Graph()
G.add_edges_from(auto_merge_pairs[['record_id_1', 'record_id_2']].values)
clusters = list(nx.connected_components(G))

# 9. Dynamic merge function with survivorship rules
def merge_cluster(df, cluster, rules):
    members = list(cluster)
    records = df.loc[members]
    merged = {}
    for col in df.columns:
        if col in rules:
            strategy = rules[col]
            if strategy == 'prefer_Y':
                preferred = records[records[col] == 'Y'][col].dropna()
                val = preferred.iloc[0] if not preferred.empty else records[col].dropna().iloc[0] if not records[col].dropna().empty else None # pylint: disable=line-too-long
            elif strategy == 'longest_string':
                val = max(records[col].dropna(), key=len, default=None)
            elif strategy == 'mode':
                val = Counter(records[col].dropna()).most_common(1)[0][0] if not records[col].dropna().empty else None # pylint: disable=line-too-long
            else:
                val = None
            merged[col] = str(val).strip() if pd.notnull(val) else None
    merged['source_ids'] = members
    merged['merge_score'] = 1.0
    return pd.Series(merged)

# 10. Handle singletons
all_ids = set(df.index)
clustered_ids = set().union(*clusters)
review_pairs = features[features['match_category'] == 'review'].reset_index()
review_ids = set(review_pairs['record_id_1']) | set(review_pairs['record_id_2'])
singleton_ids = sorted(all_ids - clustered_ids - review_ids)

singleton_goldens = []
for rid in singleton_ids:
    record = df.loc[rid].copy()
    clean_record = {col: str(record[col]).strip() if pd.notnull(record[col]) else None for col in df.columns if col != 'original'} # pylint: disable=line-too-long
    clean_record['source_ids'] = [rid]
    clean_record['merge_score'] = 0.0
    singleton_goldens.append(clean_record)

# 11. Write non-matched single records
with open(output_path, 'a', encoding='utf-8') as f:
    f.write('\n--- NON_MATCHED SINGLE RECORDS ---\n')
    for rid in singleton_ids:
        rec = df.loc[rid].to_dict()
        f.write(f"🧍 Record ({rid}): {rec}\n")
        f.write(f"💡 Similarity Score: 0.0 | Match Category: non_match (singleton)\n")
        f.write('-' * 80 + '\n')

# 12. Merge clusters and combine golden records
unique_clusters = {frozenset(cluster) for cluster in clusters}
golden_clusters = [merge_cluster(df, list(cluster), survivorship_rules) for cluster in unique_clusters] # pylint: disable=line-too-long
singleton_goldens_series = [pd.Series(rec) for rec in singleton_goldens]
golden_df = pd.DataFrame(golden_clusters + singleton_goldens_series)
# golden_df = pd.DataFrame(golden_clusters + singleton_goldens)
golden = f"golden_rec_{TIMEIS}.txt"
output_path_2 = f"{OUTPUT_PATH}{golden}"
# golden_df.to_csv(r'D:\mygit\OpenMDM\mdm\source_data\golden_auto_records.csv', index=False)
golden_df.to_csv(output_path_2, index=False)
print(f"🥇 Golden Records Document Saved here: {output_path_2}")