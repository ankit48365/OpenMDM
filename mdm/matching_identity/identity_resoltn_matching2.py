

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

# below section helps later, # to avoid self-matching pairs, like record 1 being compared to itself
candidate_pairs = candidate_pairs[
    candidate_pairs.get_level_values(0) != candidate_pairs.get_level_values(1)
]
total_pairs = len(candidate_pairs)
print(f"🧮 Total candidate pairs after blocking: {total_pairs}")


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

features['match_category'] = pd.cut(
    features['score'],
    bins=[0, 0.6, 0.78, 1.0],
    labels=['non_match', 'review', 'auto_merge']
)

def write_pairwise_summary(df, features, category, output_path):
    subset = features[features['match_category'] == category]
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f'\n--- {category.upper()} RECORDS ---\n')
        for (id1, id2), row in subset.iterrows():
            rec1 = df.loc[id1].to_dict()
            rec2 = df.loc[id2].to_dict()
            f.write(f"🔹 Record 1 ({id1}): {rec1}\n")
            f.write(f"🔸 Record 2 ({id2}): {rec2}\n")
            f.write(f"💡 Similarity Score: {round(row['score'], 4)} | Match Category: {category}\n")
            f.write('-' * 80 + '\n')


def merge_auto_pairs(df, features):
    golden_records = []
    auto_pairs = features[features['match_category'] == 'auto_merge']

    for (id1, id2), row in auto_pairs.iterrows():
        r1, r2 = df.loc[id1], df.loc[id2]
        preferred = r1 if r1.get('Original') == 'Y' else r2
        fallback = r2 if preferred is r1 else r1

        merged = {}
        for col in df.columns:
            if col == 'Original':
                continue
            val_pref = str(preferred[col]).strip() if pd.notnull(preferred[col]) else None
            val_fallback = str(fallback[col]).strip() if pd.notnull(fallback[col]) else None
            merged[col] = val_pref if val_pref else val_fallback

        merged['source_ids'] = [id1, id2]
        merged['merge_score'] = round(row['score'], 4)
        golden_records.append(merged)

    return pd.DataFrame(golden_records)

# # Separate datasets based on match categories
# review_pairs = features[features['match_category'] == 'review']
# auto_merge_pairs = features[features['match_category'] == 'auto_merge']
# non_match_pairs = features[features['match_category'] == 'non_match']

output_path = '/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/mdm_similarity_summary.txt'

write_pairwise_summary(df, features, 'auto_merge', output_path)
write_pairwise_summary(df, features, 'review', output_path)
write_pairwise_summary(df, features, 'non_match', output_path)

golden_df = merge_auto_pairs(df, features)
golden_df.to_csv('/home/ankiz/Documents/mygit/OpenMDM/mdm/source_data/golden_auto_records.csv', index=False)

# # Define the output file path
# output_file = 'all_similarity_buckets.csv'

# # Open file in write mode for first section, then append mode for others
# with open(output_file, 'w', encoding='utf-8') as f:
#     f.write('--- AUTO MERGED (≥ 95%) ---\n')
#     auto_merge_pairs.to_csv(f)

# with open(output_file, 'a', encoding='utf-8') as f:
#     f.write('\n\n--- USER REVIEW (80% – 95%) ---\n')
#     review_pairs.to_csv(f)

# with open(output_file, 'a', encoding='utf-8') as f:
#     f.write('\n\n--- NON MATCH (< 80%) ---\n')
#     non_match_pairs.to_csv(f)

