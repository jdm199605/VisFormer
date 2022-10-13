import pandas as pd

input_file_path = '../data/dataset.tsv'
output_file_path = '../data/dataset.tsv'

df = pd.read_table(input_file_path,sep='\t')

uids = []
fids = df['fid']

for fid in fids:
    uids.append('_'.join(fid.split('_')[:-1]))

df.insert(loc=0, column='uid', value=uids)
subset = ['uid','num_traces','num_fields','num_rows','pairs','types']
df.drop_duplicates(subset=subset,keep='first',inplace=True)
df.to_csv(output_file_path,index=False,sep='\t')

