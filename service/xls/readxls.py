import pandas as pd


depot_file = './depot.xls'  # 供热点表格
demand_file = './demand.xls'  # 用热点表格
dep_df = pd.read_excel(depot_file, index_col="depot_id")
dem_df = pd.read_excel(demand_file, index_col="demand_id")

x =[]
print(dep_df)

print(dep_df.loc["d1"]["lng"],dep_df.loc["d1"]["lat"])
x.append((dep_df.loc["d1"]["lng"],dep_df.loc["d1"]["lat"]))
print(x)
# print(dep_df.loc["d1"]["lat"])

# print((dem_df))

# print((dem_df.loc[2]["lng"]))