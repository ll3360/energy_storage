import matplotlib.pyplot as plt
import pandas as pd 


dep_df = pd.read_excel('./xls/depot_new.xls', index_col='depot_id')
dem_df = pd.read_excel('./xls/demand_new.xls', index_col='demand_id')

# 期望的组合
all_depot_ids = [['d1']]
# all_demand_ids = [[1],[2],[1,2],[1,2,3],[1,2,4],[1,2,3,4],[1,2,3,4,5]]
all_demand_ids = [[1,2]]
all_mode_ids = [3,2,0]
all_days = [1,7,14,30]
mode_names = {0:'仅储能车模式', 2:'两罐模式', 3:'三罐模式'}


x = all_days


report_df = pd.read_excel('report.xls', index_col=None)
y_mode3 = report_df.loc[0:3,'总成本(万元)']
y_mode2 = report_df.loc[4:7,'总成本(万元)']
y_mode0 = report_df.loc[8:11,'总成本(万元)']


print(y_mode3)
plt.plot(x,y_mode3, marker = 'o',color ='r', label='三罐模式')
plt.plot(x,y_mode2, marker = 'o',label='两罐模式')
plt.plot(x,y_mode0, marker='o',color='g',label='仅储能车模式' )
plt.title('平顶山中电-平棉纺织 两罐/三罐/仅储能车模式总成本对比')
plt.xlabel('运营时间/天')
plt.ylabel('总成本/万元')
plt.legend()
plt.savefig('平顶山中电-平棉纺织.png')
# plt.show()




# p_node1_mode3 = [0.09668,1.028739,2.042643,4.528132]
# p_node1_mode2 = [-0.066853,-0.216638,-0.48166,-0.881088]
# p_node1_mode0 = [0.163483,1.446798,2.956316,6.353855]
# plt.plot(x,p_node1_mode3, marker = 'o',color ='r', label='三罐模式')
# plt.plot(x,p_node1_mode2, marker = 'o',label='两罐模式')
# plt.plot(x,p_node1_mode0, marker='o',color='g',label='仅储能车模式' )
# plt.title('平顶山中电-平棉纺织 两罐/三罐/仅储能车模式总利润对比')
# plt.xlabel('运营时间/天')
# plt.ylabel('总利润/万元')
# plt.rcParams['axes.unicode_minus'] = False 
# plt.legend()
# plt.show()