from pprint import pprint as pp
from service import pred_api
from service import scheme_api
from pprint import pprint
from tqdm import tqdm
import pandas as pd 
from service.file_config import DEPOT_FILE, DEMAND_FILE
from datetime import datetime, timedelta
from itertools import combinations

pred_id = 1
start_date = '2022-12-05'
end_date = '2022-12-05'

depot_ids = ['d1']
demand_ids = [1, 2, 3, 4, 5]

# depot_ids = ['d2']
# demand_ids = [6, 7, 8, 9, 10]

# 测试pred接口
# result = pred_api.main(pred_id, start_date, end_date, depot_ids, demand_ids)
# pp(result)

# 测试scheme接口
# start_date = '2023-05-10'
# end_date = '2023-05-10'

# depot_ids = ['d1']
# demand_ids = [3,4]
# result = scheme_api.main(start_date, end_date, depot_ids, demand_ids)
# pprint(result)

# 预先计算所有可能的结果
start_date = '2023-05-10'
dep_df = pd.read_excel(DEPOT_FILE, index_col='depot_id')
dem_df = pd.read_excel(DEMAND_FILE, index_col='demand_id')
# all_depot_ids = dep_df.index.tolist()
# all_demand_ids = dem_df.index.tolist()
all_depot_ids = ['d1']
all_demand_ids = [1,2,3,4,5]

for depot_id in all_depot_ids:
	depot_ids = [depot_id]
	for days in tqdm(range(31)):
		end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')
		for demand_num in range(1, len(all_demand_ids)+1):
			for demand_ids in tqdm(combinations(all_demand_ids, demand_num)):
				print(depot_ids, demand_ids, start_date, end_date) 
				result = scheme_api.main(start_date, end_date, depot_ids, demand_ids)

	




