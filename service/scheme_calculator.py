from collections import deque, OrderedDict
# from typing import List, Tuple, Union
import math

import numpy as np
import pandas as pd
from geopy.distance import geodesic
# import matplotlib.pyplot as plt

from service.file_config import DEPOT_FILE, DEMAND_FILE, DEMAND_HISTORY_FLOW_FOLDER
from service.mip_model import getTasksDict, calMinVehNum


# 挂车(含移动罐)单价(元/辆) = 移动罐(310000) + 骨架车(85900) + 金属软管及快装接头(8000)
#                          + 移动罐运输费用(10000) + 骨架车运输费用(2500) 
#                          + 移动罐手续费用(500) + 骨架车手续费用 (7600)
TRAILER_UNIT = 424500  # 挂车(含移动罐)单价(元/辆)
TRACTOR_SERVICE_UNIT = 19000  # 牵引车手续费用(元/辆)
TANK_UNIT_COST = 300000  # 固定罐单价(元/台)
TANK_ISTLN_UNIT_COST = 500000  # 固定罐安装单价(元/台)
DEPOT_TRANSFORM = 500000  # 热源改造原值(元)
DEMAND_TRANSFORM = 23000  # 用户改造原值(元) = 流量计(5000)+市场开发费用(15000)
DEPOT_TRANSFORM_LIFESPAN = 5  # 热源改造使用年限(年)
DEMAND_TRANSFORM_LIFESPAN = 5  # 用户改造使用年限(年)
VEH_LIFESPAN = 8  # 车预计使用年限(年)
TANK_LIFESPAN = 8  # 罐预计使用年限(年)
EST_NET_RESIDUAL_VALUE_RATE = 0.05  # 预计净残值率(5%)
STEAM_UNIT_COST = 160  # 蒸汽单价(元/t)
ELECT_UNIT_COST = 0.83  # 电费单价(元/kW·h)
MAINT_ANNUAL_RATIO = 0.03  # 年维修费用比例(每年按固定设备3%计算)
MGMT_ANNUAL_COST = 99600  # 年管理费用(元/年)
STEAM_UNIT_PRICE = 350  # 用户购买热蒸汽的价格(元/t)
FUEL_UNIT_COST = 7.44  # 0号柴油的价格(元/L)
FUEL_CONSUMPTION_100KM = 55  # 百公里油耗(L/100km)


class Scheme:
    """Operation scheme"""

    def __init__(self):
        self.days = 0  # 方案天数(天)
        self.depot_ids = []  # 供热点列表
        self.demand_ids = []  # 用热点列表
        self.dep_df = None  # 供热点数据
        self.dem_df = None  # 用热点数据
        self.models = {}  # {demand_id:model}
        self.mileage = 0  # 储能车行驶的里程数(km)
        self.tank_num = 0  # 固定罐数(个)
        self.tractor_num = 0  # 牵引车数(个)
        self.trailer_num = 0  # 挂车(含移动罐)数(个)
        self.elec_qty = 0  # 用电量(kW·h)
        self.steam_qty = 0  # 蒸汽用量(t)
        self.shipping_qty = 0  # 运输量(t)
        self.t_v_cap_ratio = 2  # 固定罐容量与车容量之比(按int算)
        self.tasks_list = []  # 储能车(牵引车)任务列表
        self.tasks_list_trailer = []  # 挂车(含移动罐)任务列表
        self.total_cost = 0  # 总成本(元)
        self.total_profit = 0  # 总利润(元)
        self.usage = 0  # 方案的用热量(t)
        self.share_flag = False  # 方案不分摊
        self.tractor_sol = []  # 挂车的解
        self.trailer_sol = []  # 移动罐的解


class Tanks:
    """Tanks mode"""

    def __init__(self):
        self.days = 0  # 天数
        self.depot_id = None  # 供热点id
        self.demand_id = 0  # 用热点id
        self.mode_id = 0  # 模式id
        self.tank_num = 0  # 罐数
        self.tractor_num = 0  # 牵引车数(个)
        self.trailer_num = 0  # 挂车(含移动罐)数(个)        
        self.charging_time = 30  # 充热时间(分钟)
        self.pumping_time = 30  # 抽水时间(分钟)
        self.pump_worktime = 0  # 水泵工作时间(h), 充热和抽水都需要水泵
        self.pump_power = 22  # 水泵功率(kW)
        self.traveling_time = 0  # 时间 = 距离 / 速度
        self.recharging_time = 45  # 移动储能车再充热时间(分钟)
        self.drop_and_pull_time = 15  # 储能车甩挂时间(分钟)
        self.service_time_with_p = 0 # 用户侧的服务时间(车带罐来)(分钟)
        self.service_time_without_p = 0 # 用户侧的服务时间(车空车头来)(分钟)
        self.tank_steam_cap = 7.36  # 单罐最大蒸汽释放量(t) 2.5Mpa to 0.4Mpa
        self.tank_steam_loss = 0.6  # 固定罐热损: 0.6t/个
        self.veh_steam_cap = 3.2  # 单车最大蒸汽释放量(t) 2.5Mpa to 0.5Mpa
        self.veh_steam_loss = 0.3  # 储能车热损: 0.3t/辆
        self.veh_max_flow = 1.8  # 单车最大放热流量(t/h)
        self.demand_flow = []  # 用户侧流量
        self.demand_timeline = []  # 需求时间线, 存放每分钟累积用热量
        self.init_p_tank_num = 0  # 初始的移动罐数
        self.t_v_cap_ratio = 2  # 固定罐容量与车容量之比(按int算)
        self.tasks_list = []  # 储能车任务列表
        self.tasks_list_trailer = []  # 挂车(含移动罐)任务列表
        self.tank_consume = 0  # 固定罐蒸汽消耗个数(个)
        self.p_tank_consume_num = 0  # 移动罐蒸汽消耗个数(个)
        self.elec_qty = 0  # 用电量(kW·h)
        self.shipping_qty = 0  # 运输量(t), 储能车运输的蒸汽量
        self.steam_qty = 0  # 蒸汽量(t), 从供热点流量计上累积获得的蒸汽量
        self.usage = 0  # 用热点的用热量(t)
        self.distance = 0  # 用热点到用热点的距离(km)
        self.mileage = 0  # 储能车行驶的里程数(km)
        self.total_cost = 0  # 成本(万元)
        self.total_profit = 0  # 利润(万元)
        self.share_flag = True  # 点位分摊
        self.none_p_tank_judge = None  # 空车头判断条件


def getTravelingTime(model):
    """get model.traveling_time"""
    # depot_file = './xls/depot_new.xls'  # 供热点表格
    # demand_file = './xls/demand_new.xls'  # 用热点表格
    dep_df = pd.read_excel(DEPOT_FILE, index_col="depot_id")
    dem_df = pd.read_excel(DEMAND_FILE, index_col="demand_id")

    # 供热点GPS(lat, lng)
    # depot_gps = dep_df.loc[model.depot_id, ["depot_lat", "depot_lng"]]
    depot_gps = dep_df.loc[model.depot_id, ["lat", "lng"]]

    # 用热点GPS(lat, lng)
    # demand_gps = dem_df.loc[model.demand_id, ["demand_lat", "demand_lng"]]
    demand_gps = dem_df.loc[model.demand_id, ["lat", "lng"]]

    # 路径系数
    cof = 1.5392156862745099

    # 两点间距 = 路径系数 * 两点测地线距离(km)
    distance = cof * geodesic(depot_gps, demand_gps).km

    # 根据距离确定平均车速
    if distance < 10:
        veh_velocity = 25
    else:
        veh_velocity = 30

    # 入站时间(3min)和出站时间(3min)
    in_time = 3
    out_time = 3

    # 行驶时间(min) = 出站时间 + 供用距离/车速 + 入站时间
    model.traveling_time = out_time + int(distance / veh_velocity * 60) + in_time
    model.distance = distance
    # print('model.traveling_time:', model.traveling_time)


def mape(y_true, y_pred):
    """Return MAPE(Mean Absolute Percentage Error)"""
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def findFirstEligibleIndex(seq, value):
    """Use dichotomy to find the index of the first value greater
    than or equal to a value in a sequence. Return the index if succeeded,
    otherwise return None.
    """
    left_index = 0
    right_index = len(seq) - 1
    while left_index < right_index:
        mid_index = int(left_index + (right_index - left_index) / 2)
        if seq[mid_index] < value:
            left_index = mid_index + 1
        else:
            right_index = mid_index
    if 0 <= left_index <= len(seq) - 1:
        result = left_index
    else:
        result = None
    return result


def getTasksListOfThreeTanks(model):
    """Get tasks_list of three-tank mode"""
    timeline = model.demand_timeline
    # print(f'用户{model.demand_id}使用蒸汽量: {timeline[-1]}')
    model.tank_consume = math.floor(timeline[-1] / model.tank_steam_cap)
    # print('model.tank_consume:', model.tank_consume)

    # 判断空车头的条件
    # 用热点的服务时间

    # 初始状态: A,B,C三罐. A-正在使用, B-空罐, C-已使用完,装有剩余饱和水
    # 生成配送任务列表
    tasks_list = []
    switch_time_list = []
    switch_time_previous = 0  # 前一次的switch_time
    for i in range(1, model.tank_consume + 1):
        # 罐的转换时刻(一个罐用完切换到下一个罐的时刻) index + 1
        switch_time = findFirstEligibleIndex(timeline, i * model.tank_steam_cap) + 1
        switch_time_list.append(switch_time)
        # print('switch_time:', switch_time)

        # 任务一:车(满蒸汽)从热源出发, 到达用户侧, 将车上的热水充入空罐B中
        # 然后将罐C的剩余饱和水抽到车上的移动罐中, 车(剩余饱和水)从用户侧出发, 回到热源, 向移动罐中补充蒸汽, 至补充完毕
        task1_start = switch_time - model.charging_time - model.traveling_time
        task1_end = switch_time + model.pumping_time + model.traveling_time + model.recharging_time
        task1 = task1_start, task1_end, model.depot_id, model.depot_id
        tasks_list.append(task1)

        # 额外任务:根据罐车容积比增加额外任务, 如果时间够, 则不增加车辆, 如果不够, 则增加车辆
        ex_task_end = task1_start
        for j in range(model.t_v_cap_ratio - 1):
            ex_task_start = ex_task_end - model.recharging_time - model.traveling_time \
                            - model.pumping_time - model.charging_time - model.traveling_time
            pre_task_end = switch_time_previous + model.pumping_time + model.traveling_time + model.recharging_time
            if pre_task_end <= ex_task_start:
                # print(f'单车运{model.t_v_cap_ratio}次')
                ex_task = ex_task_start, ex_task_end, model.depot_id, model.depot_id
                ex_task_end = ex_task_start
            else:
                # print('增加车加运一次')
                ex_task = task1
                ex_task_end = task1_start
            tasks_list.append(ex_task)
        switch_time_previous = switch_time
    model.tasks_list = tasks_list

    # 生成挂车(含移动罐)的任务, 固定罐模式下, 车罐不分离
    model.tasks_list_trailer = tasks_list


def getTasksListOfTwoTanks(model):
    """Get tasks_list of two-tank mode"""
    timeline = model.demand_timeline
    # print(f'用户{model.demand_id}使用蒸汽量: {timeline[-1]}')
    model.tank_consume = math.floor(timeline[-1] / model.tank_steam_cap)
    # print('model.tank_consume:', model.tank_consume)

    # 初始状态: A,B两罐. A-正在使用, B-已使用完,装有剩余饱和水
    # 生成配送任务列表
    tasks_list = []
    switch_time_list = []
    switch_time_previous = 0
    for i in range(1, model.tank_consume + 1):
        # 罐的转换时刻(一个罐用完切换到下一个罐的时刻) index + 1
        switch_time = findFirstEligibleIndex(timeline, i * model.tank_steam_cap) + 1
        switch_time_list.append(switch_time)
        # print('switch_time:', switch_time)

        # 任务一:车(满)从热源出发, 到达用户侧, 将车上的热水充入固定罐B中, 再以空罐状态返回热源,开始工位1, 结束工位1
        task1_start = switch_time - model.charging_time - model.traveling_time
        task1_end = switch_time + model.traveling_time
        task1 = task1_start, task1_end, model.depot_id, model.depot_id
        tasks_list.append(task1)

        # 任务二: 在给罐B充热前, 必须先从B中抽水. 车(空)从热源出发, 到达用户侧, 将B中的剩余饱和水抽出, 再返回热源, 补充热至完毕
        task2_end = switch_time - model.charging_time - model.traveling_time
        task2_start = (task2_end - model.recharging_time - model.traveling_time
                       - model.pumping_time - model.traveling_time)
        pre_task1_end = switch_time_previous + model.traveling_time

        # 如果时间够, 则单车抽水加充热, 如果不够, 则一辆抽水, 一辆充热
        if pre_task1_end <= task2_start:
            task2 = task2_start, task2_end, model.depot_id, model.depot_id
            # print('单车抽水加充热')
        else:
            task2_end = switch_time - model.charging_time + model.traveling_time + model.recharging_time
            task2_start = switch_time - model.charging_time - model.pumping_time - model.traveling_time
            task2 = task2_start, task2_end, model.depot_id, model.depot_id
            # print('一辆抽水, 一辆充热')
        tasks_list.append(task2)

        # 根据车罐容积比添加额外任务
        for j in range(model.t_v_cap_ratio - 1):
            ex_task1 = task1
            ex_task2 = task2
            tasks_list.append(ex_task1)
            tasks_list.append(ex_task2)
        switch_time_previous = switch_time
    model.tasks_list = tasks_list

    # 生成挂车(含移动罐)的任务, 固定罐模式下, 车罐不分离
    model.tasks_list_trailer = tasks_list


def getSwitchReqList(switch_time_list: list, model) -> list:
    """Get switch_req_list"""
    # 根据本次switch到下一次switch期间的流量, 计算本次switch所需的移动罐数, 单车最大释放流量1.8t/h
    req_veh_list = [math.ceil(x / model.veh_max_flow) for x in model.demand_flow]
    result = []
    queue = deque(switch_time_list)
    switch_time = 0  # 初始值
    left_index = 0  # 初始值
    while True:
        try:
            switch_time = queue[0]
            left_index = math.floor(switch_time / 60)
            right_index = left_index + 1
            switch_time_next = queue[req_veh_list[right_index]]
            right_index_next = math.floor(switch_time_next / 60) + 1
            req_p_tank_num = max(req_veh_list[right_index:right_index_next])
            result.append((switch_time, req_p_tank_num))
            for k in range(req_p_tank_num):
                queue.popleft()
        except IndexError:
            result.append((switch_time, req_veh_list[left_index]))
            break
    return result


def getTasksListOfVehOnly(model):
    """Get tasks_list of vehicle-only mode"""
    timeline = model.demand_timeline
    # print(f'用户{model.demand_id}使用蒸汽量: {timeline[-1]}')
    model.p_tank_consume_num = math.floor(timeline[-1] / model.veh_steam_cap)
    # print('model.p_tank_consume_num:', model.p_tank_consume_num)

    # 判断空车头的条件, 如果(结束时间-开始时间) < (traveling + drop_and_pull), 则为空车头
    model.none_p_tank_judge = model.traveling_time + model.drop_and_pull_time

    # 用热点的服务时间(车带满罐来) = 甩挂时间
    model.service_time_with_p = model.drop_and_pull_time

    # 用热点的服务时间(空车头来) = 甩挂时间/2
    model.service_time_without_p = math.ceil(model.drop_and_pull_time / 2)

    # 所有的转换时刻
    switch_time_list = []
    for i in range(1, model.p_tank_consume_num + 1):
        # 罐的转换时刻(一个罐用完切换到下一个罐的时刻) index + 1
        switch_time = findFirstEligibleIndex(timeline, i * model.veh_steam_cap) + 1
        switch_time_list.append(switch_time)
    # print('switch_time_list:', switch_time_list)

    # 根据流量flow获取switch_req_list := [(switch_time, req_p_tank_num)]
    switch_req_list = getSwitchReqList(switch_time_list, model)
    # print('switch_req_list:', switch_req_list)

    # 生成任务列表
    tasks_list = []  # 牵引车(拖车)的任务
    tasks_list_trailer = []  # 生成挂车(含移动罐)的任务
    req_p_tank_num_pre = math.ceil(model.demand_flow[0] / model.veh_max_flow)
    model.init_p_tank_num = req_p_tank_num_pre
    # print(f'初始有{model.init_p_tank_num}罐正在使用')
    switch_time_previous = 0
    for i in range(len(switch_req_list)):
        switch_time, req_p_tank_num = switch_req_list[i]

        # 任务一: 车的任务, 车(载满罐)从热源侧出发, 在用户侧甩挂, 开始工位'd1', 结束工位1
        task1_start = switch_time - model.service_time_with_p - model.traveling_time
        task1_end = switch_time
        task1 = task1_start, task1_end, model.depot_id, model.demand_id
        tasks_list.append(task1)

        # 任务三: 罐的任务, 满罐被运往用热点, 被留在用热点, 直到使用完, 开始工位'd1', 结束工位1
        task3_start = task1_start
        try:
            task3_end = switch_req_list[i + 1][0]
            task3 = task3_start, task3_end, model.depot_id, model.demand_id
            tasks_list_trailer.append(task3)
        except IndexError:
            # task3_end = switch_time + switch_time - switch_time_previous  # 不知道最后的罐子何时用完, 暂定经1440分钟用完
            pass

        # 根据req_p_tank_num增加车和罐的送达任务
        for _ in range(req_p_tank_num - 1):
            tasks_list.append(task1)
            tasks_list_trailer.append(task3)

        # 增加空车的去任务
        if req_p_tank_num_pre > req_p_tank_num:
            for k in range(req_p_tank_num_pre - req_p_tank_num):
                ex_task1 = (switch_time - model.service_time_without_p - model.traveling_time,
                            switch_time, model.depot_id, model.demand_id)
                tasks_list.append(ex_task1)

        # 任务二:车(载空罐)从用户侧出发, 返回热源侧, 甩下空罐, 装上满罐 开始工位1, 结束工位'd1'
        task2_start = switch_time
        task2_end = task2_start + model.traveling_time + model.drop_and_pull_time
        task2 = task2_start, task2_end, model.demand_id, model.depot_id
        tasks_list.append(task2)

        # 任务四: 空罐从用户侧出发, 被运回热源, 重新充热至完毕, 开始工位1, 结束工位'd1'
        task4_start = switch_time
        task4_end = task4_start + model.traveling_time + model.drop_and_pull_time + model.recharging_time
        task4 = task4_start, task4_end, model.demand_id, model.depot_id
        tasks_list_trailer.append(task4)

        # 根据req_p_tank_num_pre增加车和罐的取回任务
        for j in range(req_p_tank_num_pre - 1):
            tasks_list.append(task2)
            tasks_list_trailer.append(task4)

        # 增加空车的回任务
        if req_p_tank_num > req_p_tank_num_pre:
            for k in range(req_p_tank_num - req_p_tank_num_pre):
                ex_task2 = (switch_time,
                            switch_time + math.ceil(model.drop_and_pull_time / 2) + model.traveling_time,
                            model.demand_id, model.depot_id)
                tasks_list.append(ex_task2)

        req_p_tank_num_pre = req_p_tank_num
        switch_time_previous = switch_time

    # 保存牵引车(拖车)任务和挂车(含移动罐)任务
    model.tasks_list = tasks_list
    model.tasks_list_trailer = tasks_list_trailer
    # print('model.tasks_list:', model.tasks_list)
    # print('length of model.tasks_list:', len(model.tasks_list))
    # print('model.tasks_list_trailer:', model.tasks_list_trailer)
    # print('length of model.tasks_list_trailer', len(model.tasks_list_trailer))

    # 画出更换移动罐的时间点
    # plt.figure()
    # plt.plot(switch_time_list, range(1, len(switch_time_list)+1), marker='o', label='更换移动罐的时间点(原)')
    # plt.xlabel('分钟')
    # plt.ylabel('罐数')
    # xdata = [x[0] for x in switch_req_list]
    # ydata = pd.Series([1]+[x[1] for x in switch_req_list][:-1]).cumsum()
    # t_num = [x[1] for x in switch_req_list]
    # plt.scatter(xdata, ydata, color='r', marker='*', linewidths=5, label='更换移动罐的时间点(新)')
    # for i in range(len(xdata)):
    #     plt.text(xdata[i]+25, ydata[i]-0.1, f'{t_num[i]}罐供热', color='red')   
    # plt.legend()
    # plt.show()


def getDemandTimeline(model):
    """Get model.demand_timeline"""
    df = pd.read_excel(DEMAND_HISTORY_FLOW_FOLDER + f'/{model.demand_id}.xlsx')

    # 累积用热量
    df_cumsum = df.cumsum()

    # 取最后一行里最大值对应的列标签, 即得到用热量最高的那一天, 用这一天的数据建立用热模型
    last_line = df_cumsum.iloc[-1, :]
    highest_day = last_line.idxmax()
    # print('highest_day:', highest_day)

    # 对highest_day对应的曲线y-x进行多项式拟合
    x = np.array(list(range(1, 25)))
    y = np.array(df_cumsum[highest_day])

    # get fit_func 拟合曲线函数
    deg = 1
    while True:
        try:
            fit_func = np.polynomial.polynomial.Polynomial.fit(x, y, deg=deg)  # 多项式拟合
            y_p = fit_func(x)
            y_mape = mape(y, y_p)
            if y_mape <= 5:  # 判断为真则break
                break
            else:
                deg += 1
        finally:
            pass

    # 画拟合曲线
    # plt.figure(1)
    # plt.plot(np.array(list(range(24))), df[highest_day])
    # plt.plot(np.array(list(range(24))), 1.8*np.ones(x.shape), linestyle='--', color='orange',label='2移动罐供热')
    # plt.plot(np.array(list(range(24))), 2*1.8*np.ones(x.shape), linestyle='--', color='red', label='3移动罐供热')
    # plt.plot(np.array(list(range(24))), 3*1.8*np.ones(x.shape), linestyle='--', color='purple', label='4移动罐供热')
    # plt.legend()
    # plt.title(f'用户{model.demand_id}(0-23时)流量')
    # plt.xlabel('时间/h')
    # plt.ylabel('流量(t/h)')
    # plt.show()

    # plt.figure(2)
    # plt.plot(x, y, color='g', label=f'{highest_day}真实值')
    # plt.scatter(x, y, marker='^', color='purple', label='0-24小时用热累积值')
    # for i in range(1, math.ceil(y[-1]/3.62)):
    #     plt.axhline(3.62*i, linestyle='--', color='red')
    # for i in np.array([1,2,4,3,4,2,2,1]).cumsum():
    #     plt.axhline(3.62*i, linestyle='--', color='red')
    # plt.plot(x, y_p, label=f'{deg}阶多项式拟合曲线')
    # plt.title(f'mape = {y_mape:.2f}%')
    # plt.xlabel('时间/h')
    # plt.ylabel('累积用量/t')
    # plt.legend()
    # plt.show()

    # 24小时=1440分钟, 0:00-1:00线性增长, 1:00-24:00按拟合曲线增长
    one_oclock = fit_func(1)  # 1点整的累积用量
    first_60 = np.array((np.arange(0, one_oclock, one_oclock / 60, dtype='float64')))
    rest_minute = np.array(range(60, 1440))
    rest_1380 = fit_func(rest_minute / 60)
    total_1440 = np.concatenate((first_60, rest_1380), axis=None)

    # 需求时间线
    demand_timeline = []
    for k in range(model.days):
        demand_timeline += [i + total_1440[-1] * k for i in total_1440]

    daily_usage = total_1440[-1]  # 该点的日用热量
    model.usage = daily_usage * model.days  # 该点的总用热量
    model.demand_timeline = demand_timeline
    model.demand_flow = df[highest_day].tolist() * model.days  # 用热流量


def getQtyOfThreeTanks(model):
    """get shipping_qty, steam_qty, elec_qty of three-tank mode"""
    # 运输量(t) = (固定罐消耗个数 + 初始的1罐) * (单罐最大蒸汽释放量 + 固定罐热损)
    model.shipping_qty = (model.tank_consume + 1) * (model.tank_steam_cap + model.tank_steam_loss)
    # print('运输量:', model.shipping_qty)

    # 蒸汽量(t) = (固定罐消耗个数 + 初始的1罐) * (单罐最大蒸汽释放量 + 固定罐热损 + 储能车热损)
    model.steam_qty = (model.tank_consume + 1) * (model.tank_steam_cap + model.tank_steam_loss + model.veh_steam_loss)
    # print('蒸汽量:', model.steam_qty)

    # 水泵工作时间(h) = 任务数 * (充热时间(min) + 抽水时间(min)) / 60
    model.pump_worktime = len(model.tasks_list) * (model.charging_time + model.pumping_time) / 60

    # 用电量 = 水泵功率 * 水泵工作时间
    model.elec_qty = model.pump_power * model.pump_worktime

    # 行驶里程(km) = 任务数 * 供用两点距离
    model.mileage = len(model.tasks_list) * model.distance


def getQtyOfTwoTanks(model):
    """get shipping_qty, steam_qty, elec_qty of two-tank mode"""
    # 运输量(t) = (固定罐消耗个数 + 初始的1罐) * (单罐最大蒸汽释放量 + 固定罐热损)
    model.shipping_qty = (model.tank_consume + 1) * (model.tank_steam_cap + model.tank_steam_loss)
    # print('运输量:', model.shipping_qty)

    # 蒸汽量(t) = (固定罐消耗个数 + 初始的1罐) * (单罐最大蒸汽释放量 + 固定罐热损 + 储能车热损)
    model.steam_qty = (model.tank_consume + 1) * (model.tank_steam_cap + model.tank_steam_loss + model.veh_steam_loss)
    # print('蒸汽量:', model.steam_qty)

    # 水泵工作时间(h) = 任务数 * 充热或抽水时间(min) / 60
    model.pump_worktime = len(model.tasks_list) * model.charging_time / 60

    # 用电量 = 水泵功率 * 水泵工作时间
    model.elec_qty = model.pump_power * model.pump_worktime

    # 行驶里程(km) = 任务数 * 供用两点距离 * 2
    model.mileage = len(model.tasks_list) * model.distance * 2


def getQtyOfVehOnly(model):
    """get shipping_qty, steam_qty, elec_qty of veh-only mode"""
    # 运输量(t) = (移动罐消耗个数 + 初始放置在用热点的罐数) * 单车最大蒸汽释放量
    model.shipping_qty = (model.p_tank_consume_num + model.init_p_tank_num) * model.veh_steam_cap
    # print('运输量:', model.shipping_qty)

    # 蒸汽量(t) = (移动罐消耗个数 + 初始放置在用热点的罐数) * (单车最大蒸汽释放量 + 储能车热损)
    model.steam_qty = (model.p_tank_consume_num + model.init_p_tank_num) * (model.veh_steam_cap + model.veh_steam_loss)
    # print('蒸汽量:', model.steam_qty)

    # 水泵工作时间(h) = 0 此模式不需要用到水泵
    model.pump_worktime = 0

    # 用电量 = 0 此模式不需要用到水泵
    model.elec_qty = 0

    # 行驶里程(km) = 任务数 * 供用两点距离
    model.mileage = len(model.tasks_list) * model.distance


def getTasksListStep(sol_step_previous: list, tasks_dict_previous: dict) -> list:
    """Get tasks_list_step"""
    result = []
    for i in sol_step_previous:
        start_id = i[0]
        end_id = i[-1]
        start_time = tasks_dict_previous[start_id][1]
        start_node = tasks_dict_previous[start_id][3]
        end_time = tasks_dict_previous[end_id][2]
        end_node = tasks_dict_previous[end_id][4]
        task = start_time, end_time, start_node, end_node
        result.append(task)
    return result


def getSol(tasks_list, days):
    """ Get solution of scheme.tasks_list"""
    tasks_dict = getTasksDict(tasks_list)

    # 第一步: 先将所有的任务分成每天的任务, 求解执行每天的任务所需的最小车辆数和出车安排, 将每天的解放进sol_step1
    # print('Step1求解中...')
    sol_step1 = []
    checker = []
    groupNum = 7
    for day in range(days):
        tasks_list_daily = []
        index = 1
        map_dict = {}
        for key, val in tasks_dict.items():
            if 1440 * day <= val[2] < 1440 * (day + 1):  # 结束时间val[2]
                tasks_list_daily.append(val)
                map_dict[index] = key
                index += 1
                checker.append(tuple(val[1:5]))
        # print('map_dict:', map_dict)
        tasks_dict_daily = OrderedDict()
        for i in range(len(tasks_list_daily)):
            tasks_dict_daily[i + 1] = tasks_list_daily[i]
        min_veh_num_daily, tmp_sol = calMinVehNum(tasks_dict_daily, groupNum, days=1)
        daily_sol = [[map_dict[x] for x in y] for y in tmp_sol]
        groupNum = len(daily_sol)
        # print(f'Day{day + 1}: {len(tasks_dict_daily)} tasks len(daily_sol): {len(daily_sol)}')
        sol_step1 += daily_sol
    # print('sol_step1:', sol_step1, len(sol_step1))

    # 第二步: 先将sol_step1转换为任务tasks_list_step2, 再分成周任务, 求解每个周任务的最小车辆数和出车安排, 将每周的解放入sol_step2
    # print('\nStep2求解中...')
    tasks_list_step2 = getTasksListStep(sol_step1, tasks_dict)
    tasks_dict_step2 = getTasksDict(tasks_list_step2)

    sol_step2 = []
    checker = []
    weeks = math.ceil(days / 7)
    for week in range(weeks):
        tasks_list_weekly = []
        index = 1
        map_dict = {}
        for key, val in tasks_dict_step2.items():
            if 1440 * 7 * week <= val[2] < 1440 * 7 * (week + 1):  # 结束时间val[2]
                tasks_list_weekly.append(val)
                map_dict[index] = key
                index += 1
                checker.append(tuple(val[1:5]))
        tasks_dict_weekly = OrderedDict()
        for i in range(len(tasks_list_weekly)):
            tasks_dict_weekly[i + 1] = tasks_list_weekly[i]
        min_veh_num_weekly, tmp_sol = calMinVehNum(tasks_dict_weekly, groupNum, days=7)
        weekly_sol = [[map_dict[x] for x in y] for y in tmp_sol]
        # print(f'Week{week + 1}: {len(tasks_dict_weekly)} tasks len(weekly_sol): {len(weekly_sol)}')
        sol_step2 += weekly_sol
    # print('sol_step2:', sol_step2, len(sol_step2))

    # 第三步: 将sol_step2转换为任务tasks_list_step3, 求解执行任务的最小车辆数和出车安排, 得到解sol_step3
    # print('\nStep3求解中...')
    tasks_list_step3 = getTasksListStep(sol_step2, tasks_dict_step2)
    tasks_dict_step3 = getTasksDict(tasks_list_step3)
    min_veh_num_step3, sol_step3 = calMinVehNum(tasks_dict_step3, groupNum, days)
    # print('sol_step3:', sol_step3, len(sol_step3))
    # print(f'任务拆分{len(tasks_list)}->{len(sol_step1)}->{len(sol_step2)}->{len(sol_step3)}\n')

    # 提取结果
    min_veh_num = min_veh_num_step3
    sol = []
    for i in sol_step3:
        tmp = []
        for j in i:
            tmp2 = []
            for k in sol_step2[j - 1]:
                tmp2 += sol_step1[k - 1]
            tmp += tmp2
        sol.append(tmp)
    # print('sol:',sol)
    return min_veh_num, sol


def calCostProfit(scheme):
    """Calculate cost and profit of the scheme"""
    days = scheme.days  # 方案天数(天)
    tank_num = scheme.tank_num  # 固定罐数(个)
    tractor_num = scheme.tractor_num  # 牵引车数(个)
    trailer_num = scheme.trailer_num  # 挂车数(个)
    elec_qty = scheme.elec_qty  # 用电量(kW·h)
    steam_qty = scheme.steam_qty  # 蒸汽用量(t)
    # shipping_qty = scheme.shipping_qty  # 运输量(t)
    mileage = scheme.mileage  # 行驶里程(km)
    years = days / 360  # 年

    # 储能车设备原值 = 牵引车数 * 牵引车手续费用 + 挂车(含移动罐)数 * 挂车(含移动罐)单价
    veh_cost = tractor_num * TRACTOR_SERVICE_UNIT + trailer_num * TRAILER_UNIT
    # print('储能车设备原值:', veh_cost)

    # 固定罐设备原值 = 固定罐数 * (固定罐单价 + 固定罐安装单价)
    tank_cost = tank_num * (TANK_UNIT_COST + TANK_ISTLN_UNIT_COST)
    # print('固定罐设备原值:', tank_cost)

    # 车年折旧额 = 储能车设备原值 / 1.13 * (1 - 预计净残值率) / 车预计使用年限
    veh_annual_depr = veh_cost / 1.13 * (1 - EST_NET_RESIDUAL_VALUE_RATE) / VEH_LIFESPAN

    # 罐年折旧额 = 固定罐设备原值 / 1.13 * (1 - 预计净残值率) / 罐预计使用年限
    tank_annual_depr = tank_cost / 1.13 * (1 - EST_NET_RESIDUAL_VALUE_RATE) / TANK_LIFESPAN

    # 折旧费 = (车年折旧额 + 罐年折旧额) * 年
    depreciated_cost = (veh_annual_depr + tank_annual_depr) * years
    # print('折旧费:', depreciated_cost)

    # 管理费 = 用热点个数 * 年管理费用 * 年
    global MGMT_ANNUAL_COST
    if len(scheme.demand_ids) >= 2:
        MGMT_ANNUAL_COST = 150000  # 1对2(及以上)的情况, 年管理费用为15万/年
    if scheme.share_flag:  # 输入为model
        mgmt_cost = MGMT_ANNUAL_COST / len(scheme.demand_ids) * years
    else:  # 输入为scheme
        mgmt_cost = MGMT_ANNUAL_COST * years
    # print('管理费:', mgmt_cost)

    # 热源改造摊年摊销额 = 热源改造原值 / 热源改造使用年限
    # 用户改造年摊销额 = 用户数 * 用户改造原值 / 用户改造使用年限
    if scheme.share_flag:  # 输入为model
        depot_transform_annual_amort = DEPOT_TRANSFORM * scheme.share_flag / DEPOT_TRANSFORM_LIFESPAN
        demand_transform_annual_amort = DEMAND_TRANSFORM / DEMAND_TRANSFORM_LIFESPAN
    else:  # 输入为scheme
        depot_transform_annual_amort = DEPOT_TRANSFORM / DEPOT_TRANSFORM_LIFESPAN
        demand_transform_annual_amort = len(scheme.demand_ids) * DEMAND_TRANSFORM / DEMAND_TRANSFORM_LIFESPAN
    # print('热源改造摊年摊销额:', depot_transform_annual_amort)
    # print('用户改造摊年摊销额:', demand_transform_annual_amort)

    # 摊销费 = (热源改造摊年摊销额 + 用户改造年摊销额) * 年
    amort_cost = (depot_transform_annual_amort + demand_transform_annual_amort) * years
    # print('摊销费:', amort_cost)

    # 维修费 = 设备总值 * 年维修费用占比 * 年
    maint_cost = (veh_cost + tank_cost) * MAINT_ANNUAL_RATIO * years
    # print('维护费:', maint_cost)

    # 电费 = 用电量 * 电费单价
    elec_cost = elec_qty * ELECT_UNIT_COST
    # print('电费:', elec_cost)

    # 蒸汽成本 = 蒸汽量 * 蒸汽单价
    steam_cost = steam_qty * STEAM_UNIT_COST
    # print('蒸汽成本:', steam_cost)

    # 油费 = 行驶里程 / 100 * 百公里油耗 * 油价
    fuel_cost = mileage / 100 * FUEL_CONSUMPTION_100KM * FUEL_UNIT_COST
    # print('油费:', fuel_cost)

    # 司机费 = 牵引车数 * 2 * 司机年薪 * 年
    driver_cost = tractor_num * 2 * 10000 * 12 * years

    # 运输费 = 油费 + 司机费 + 其他费用(保密)
    shipping_cost = fuel_cost + driver_cost + tractor_num * (23000 + 2000 * 10 / 2) * years + trailer_num * (
            1500 + 2000 / 36) * years
    # print('运输费:', shipping_cost)
    # print('运输单价:', shipping_cost / shipping_qty)

    # 固定成本 = 折旧费 + 管理费 + 摊销费
    fixed_cost = depreciated_cost + mgmt_cost + amort_cost
    # print('固定成本:', fixed_cost)

    # 变动成本 = 维修费 + 电费 + 蒸汽成本 + 运输费
    variable_cost = maint_cost + elec_cost + steam_cost + shipping_cost
    # print('变动成本:', variable_cost)

    # 总成本 = 固定成本 + 变动成本
    total_cost = fixed_cost + variable_cost
    # print('总成本:', total_cost)

    # 总利润 = 热蒸汽售价 * 总用热量 - 总成本
    total_profit = STEAM_UNIT_PRICE * scheme.usage - total_cost

    # 保存为万元
    scheme.total_cost = total_cost / 10000
    scheme.total_profit = total_profit / 10000


def calBestScheme(depot_ids: list, demand_ids: list, mode_ids: list, days: int, fast=False) -> Scheme:
    scheme = Scheme()
    scheme.depot_ids = depot_ids
    scheme.demand_ids = demand_ids
    scheme.mode_ids = mode_ids
    scheme.days = days
    dem_mode_zip = zip(demand_ids, mode_ids)
    if len(demand_ids) != len(mode_ids):
        raise ValueError('demand_ids要与mode_ids一一对应')

    for demand_id, mode_id in dem_mode_zip:
        model = Tanks()
        model.mode_id = mode_id
        model.tank_num = mode_id
        model.days = days
        model.demand_id = demand_id
        model.demand_ids = demand_ids
        model.depot_id = scheme.depot_ids[0]  # 先只考虑单供热点
        model.t_v_cap_ratio = scheme.t_v_cap_ratio
        getTravelingTime(model)
        getDemandTimeline(model)
        if mode_id == 3:
            # print('三罐模式计算中...')
            getTasksListOfThreeTanks(model)
            getQtyOfThreeTanks(model)
        elif mode_id == 2:
            # print('两罐模式计算中...')
            getTasksListOfTwoTanks(model)
            getQtyOfTwoTanks(model)
        elif mode_id == 0:
            # print('仅储能车模式计算中...')
            getTasksListOfVehOnly(model)
            getQtyOfVehOnly(model)
        else:
            raise ValueError('存在未知的mode_id')
        scheme.shipping_qty += model.shipping_qty
        scheme.steam_qty += model.steam_qty
        scheme.elec_qty += model.elec_qty
        scheme.mileage += model.mileage
        scheme.tank_num += model.tank_num
        scheme.tasks_list += model.tasks_list
        scheme.tasks_list_trailer += model.tasks_list_trailer
        scheme.usage += model.usage
        scheme.models[demand_id] = model  # 按demand_id保存model

    # 对方案的任务进行求解, 根据fast参数判断是否快速计算(7天以上只取7天来计算车数和罐数)
    if days >= 7 and fast:
        scheme.tractor_num, scheme.tractor_sol = getSol(scheme.tasks_list, days=7) 
        scheme.trailer_num, scheme.trailer_sol = getSol(scheme.tasks_list_trailer, days=7)
    else:
        scheme.tractor_num, scheme.tractor_sol = getSol(scheme.tasks_list, scheme.days)
        scheme.trailer_num, scheme.trailer_sol = getSol(scheme.tasks_list_trailer, scheme.days)
    # print(f 'depot_ids: {depot_ids}\n demand_ids: {demand_ids}\n mode_ids: {mode_ids}\n days: {days}\n')
    # print('总固定罐数:', scheme.tank_num)
    # print('总牵引车数:', scheme.tractor_num)
    # print('总挂车(含移动罐)数:', scheme.trailer_num)

    # 计算总成本和总利润
    calCostProfit(scheme)
    # print(f'总成本:{scheme.total_cost:.6f}万元')
    # print(f'总利润:{scheme.total_profit:.6f}万元\n')

    # print(f'{scheme.total_cost:.6f}\n{scheme.total_profit:.6f}')

    if len(demand_ids) >= 2:
        # 按里程数分摊总牵引车数, 按运输量分摊总挂车(含移动罐)数, 按用气量分摊热源改造费用, 计算每个用热点的成本和盈利
        for dem_id, dem_model in scheme.models.items():
            dem_model.tractor_num = scheme.tractor_num * dem_model.mileage / scheme.mileage
            dem_model.trailer_num = scheme.trailer_num * dem_model.shipping_qty / scheme.shipping_qty
            dem_model.share_flag = dem_model.usage / scheme.usage
            calCostProfit(dem_model)
    # print(f'用热点{dem_id}成本:{dem_model.total_cost:.6f}万元')
    # print(f'用热点{dem_id}利润:{dem_model.total_profit:.6f}万元\n')
    return scheme


def main():
    days = 1
    depot_ids = ['d1']
    demand_ids = [3,4]
    mode_ids = [0,0]
    best_scheme = calBestScheme(depot_ids, demand_ids, mode_ids, days)


if __name__ == '__main__':
    main()
