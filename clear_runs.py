import os
import pandas as pd


#
base_path = './search/ptbxl'
all_search_dir_list = os.listdir(base_path)

#
wandb_path = 'tmp'
wandb_path_list = os.listdir(wandb_path)
all_timeid_list = []
for each_f in wandb_path_list:
    each_data = pd.read_csv(os.path.join(wandb_path,each_f))
    print(each_data.shape)
    group_names = each_data['Name'].values
    group_timeid = [n.split('_')[0] for n in group_names]
    all_timeid_list += group_timeid
    #print(group_names)
print(all_timeid_list)
#
print('Search dir not use now')
count = 0
for search_dir in all_search_dir_list:
    search_ids = search_dir.split('_')[0].split('-')
    search_id = search_ids[0]+'-'+search_ids[1]
    if search_id not in all_timeid_list and 'grid' not in search_dir:
        count += 1
        search_dir_path = os.path.join(base_path,search_dir)
        print(search_dir_path)
        os.system(f'rm -rf {search_dir_path}')
print(f'Move {count} files')