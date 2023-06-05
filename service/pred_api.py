from collections import OrderedDict
import json
import math
import os
from typing import Tuple

from geopy.distance import geodesic
import lkh
import numpy as np
import pandas as pd

from service.file_config import DEPOT_FILE, DEMAND_FILE, VEHICLE_FILE
from service.mip_model import getTasksDict, calMinVehNum


def getEdgeWeightSection(weight_dict: dict) -> list:
    """Get EDGE_WEIGHT_SECTION"""
    length = len(weight_dict)
    weight_key = list(weight_dict)
    col = math.floor(math.sqrt(length))
    row, rem = divmod(length, col)
    if rem > 0:
        ews_key_list = [weight_key[col * i:col * i + col] for i in range(row)] + [weight_key[-rem:]]
    else:
        ews_key_list = [weight_key[col * i:col * i + col] for i in range(row)]
    ews = [[weight_dict[i] for i in x] for x in ews_key_list]
    return ews


def numToTimeStr(num: int) -> str:
    """Transfer num to time-string"""
    hour, minute = divmod(num, 60)
    if hour <= 9:
        hour_str = '0' + str(hour)
    else:
        hour_str = str(hour)
    if minute <= 9:
        minute_str = '0' + str(minute)
    else:
        minute_str = str(minute)
    result = hour_str + ':' + minute_str
    return result


def getVehID(x: int, sol_dict: dict) -> str:
    """Get veh_id"""
    for key, val in sol_dict.items():
        if x in val:
            return 'v' + str(key)
        else:
            pass


def getPTankID(x: list, sol_dict: dict) -> list:
    """Get tank_id"""
    r_x = [i - 1 for i in x]  # route里的点位id-1
    result = []
    for item in r_x:
        for key, val in sol_dict.items():
            if item in val:
                result.append('p' + str(key))
            else:
                pass
    return result


def getVehStatusList(route_list: list, pd_dict: dict) -> list:
    """Get veh_status_list"""
    node_id_list = route_list[1:-1]
    status1 = "1"  # 出发-满罐
    status2 = "2"  # 到达-满罐
    status3 = "3"  # 出发-车头
    status4 = "4"  # 到达-车头
    status5 = "5"  # 出发-冷罐
    status6 = "6"  # 到达-冷罐
    sep = "-"
    if len(node_id_list) == 1:  # 如果车只去一个用热点就返回
        node_id = node_id_list[0]
        pdd = pd_dict[node_id]
        if pdd == (1, 0):  # 只取不送 ["3", "4-5", "6"]
            result = [status3, status4 + sep + status5, status6]
        elif pdd == (0, 1):  # 只送不取 ["1", "2-3", "4"]
            result = [status1, status2 + sep + status3, status4]
        elif pdd == [1, 1]:  # 同时取送 ["1", "2-5", "6"]
            result = [status1, status2 + sep + status5, status6]
        else:
            raise
    elif len(node_id_list) == 2:  # 如果车去两个点返回
        pdd = [pd_dict[node_id_list[0]], pd_dict[node_id_list[1]]]
        if pdd == [(0, 1), (1, 0)]:  # ["1", "2-3", "4-5", "6"]
            result = [status1, status2 + sep + status3, status4 + sep + status5, status6]
        else:
            raise ValueError(f'ERROR!pdd:{pdd}')
    else:
        raise
    return result


def getTimetableList(key: int, timetable_dict: dict) -> list:
    """Get timetable_list"""
    result = []
    for val in timetable_dict[key]:
        if val[0] == val[1]:
            result.append(numToTimeStr(val[0]))
        else:
            result.append(numToTimeStr(val[0]) + '-' + numToTimeStr(val[1]))
    return result


def getTravelingTime(x: tuple, y: tuple) -> int:
    """Get traveling time between x and y. The coordinates of are (x_lat, x_lng)."""
    veh_speed = 30 * 1000 / 60  # 车速(m/min) 30(km/h) = 30 * 1000 / 60 = 500(m/min)
    result = int(geodesic(x, y).m / veh_speed)
    return result


def createLKHProblem(depot_ids: list, demand_ids: list):
    """Create lkh.LKHProblem on the basis of depot.xls, demand.xls. Return the problem and 
    mapping dict{cal:real}.
    """
    dep_df = pd.read_excel(DEPOT_FILE, index_col="depot_id")
    dem_df = pd.read_excel(DEMAND_FILE, index_col="demand_id")
    veh_df = pd.read_excel(VEHICLE_FILE, index_col="vehicle_id")

    # non-negative check
    dep_nm = dep_df.apply(pd.to_numeric, errors='coerce')
    dem_nm = dem_df.apply(pd.to_numeric, errors='coerce')
    if (dep_nm < 0).any().any():
        raise ValueError('depot.xls中存在负值')
    if (dem_nm < 0).any().any():
        raise ValueError('demand.xls中存在负值')

    # 单供热点
    depot_id = depot_ids[0]
    dep_ncs = dep_df.loc[depot_id, ['lat', 'lng']].tolist()

    # NODE_COORD_SECTION
    ncs = {1: dep_ncs}

    # PICKUP_AND_DELIVERY_SECTION
    pds = [[1, 0, 0, 1440, 0, 0, 0]]
    cal_map_real = {1: depot_id}  # 计算用id映射真实id
    node_id = 2
    pd_dict = {1: (0, 0)}
    tasks_list_p_tank = []  # 移动罐的任务

    for demand_id in demand_ids:
        lat = dem_df.loc[demand_id, 'lat']
        lng = dem_df.loc[demand_id, 'lng']
        # delivery_time = dem_df.loc[demand_id, 'delivery_time']    # 应修改xls字段名
        # pickup_time = dem_df.loc[demand_id, 'pickup_time']        # 应修改xls字段名
        delivery_time = dem_df.loc[demand_id, 'ready_time']  # 用户开始用热时间
        pickup_time = dem_df.loc[demand_id, 'due_time']  # 用户结束用热时间

        service_time = dem_df.loc[demand_id, 'service_time']
        delivery = dem_df.loc[demand_id, 'delivery']
        pickup = dem_df.loc[demand_id, 'pickup']
        time_window = 20  # 软时间窗

        # type check
        if type(lat) != np.float64 or type(lng) != np.float64 or type(delivery_time) != np.int64 \
                or type(pickup_time) != np.int64 or type(service_time) != np.int64 or type(pickup) != np.int64 \
                or type(delivery) != np.int64:
            raise TypeError('demand.xls中存在数值类型错误')

        # 生成罐的任务, 任务一: 满罐从供热点被送往用热点, 直到用热完毕, 开始工位1, 结束工位2
        traveling_time = getTravelingTime(tuple(ncs[1]), (lat, lng))
        recharging_time = 50
        task1_start = delivery_time - time_window - service_time - traveling_time
        task1_end = pickup_time
        task1 = task1_start, task1_end, depot_id, demand_id

        # 生成罐的任务， 任务二: 从用热点送回供热点, 重新充热完毕, 开始工位2, 结束工位1
        task2_start = pickup_time
        task2_end = task2_start + time_window + service_time + traveling_time + recharging_time
        task2 = task2_start, task2_end, demand_id, depot_id

        # for delivery
        for i in range(delivery):
            ncs[node_id] = [lat, lng]
            cal_map_real[node_id] = demand_id
            pds.append([node_id, 0, delivery_time - time_window - service_time,
                        delivery_time - service_time, service_time, 0, 1])
            pd_dict[node_id] = (0, 1)
            node_id += 1
            tasks_list_p_tank.append(task1)

        # for pickup
        for i in range(pickup):
            ncs[node_id] = [lat, lng]
            cal_map_real[node_id] = demand_id
            pds.append([node_id, 0, pickup_time, pickup_time + time_window, service_time, 1, 0])
            pd_dict[node_id] = (1, 0)
            node_id += 1
            tasks_list_p_tank.append(task2)
            # print('ncs:', ncs)

    # weight_dict = {(i,j):dij/v}  dij = geodesic(i的坐标,j的坐标)  v = veh_speed
    weight_dict = {(i, j): getTravelingTime((ncs[i][0], ncs[i][1]), (ncs[j][0], ncs[j][1]))
                   for j in range(1, len(ncs) + 1) for i in range(1, j + 1)}

    # EDGE_WEIGHT_SECTION
    ews = getEdgeWeightSection(weight_dict)

    # DEPOT_SECTION
    ds = [1]

    # keyword arguments 
    NAME = 'prob'
    TYPE = 'VRPSPDTW'
    DIMENSION = len(ncs)
    CAPACITY = 1
    EDGE_WEIGHT_TYPE = 'EXPLICIT'
    EDGE_WEIGHT_FORMAT = 'LOWER_DIAG_ROW'
    EDGE_WEIGHT_SECTION = ews
    DEPOT_SECTION = ds
    PICKUP_AND_DELIVERY_SECTION = pds

    # create problem
    prob = lkh.LKHProblem(name=NAME, type=TYPE, dimension=DIMENSION, capacity=CAPACITY,
                          edge_weight_type=EDGE_WEIGHT_TYPE, edge_weight_format=EDGE_WEIGHT_FORMAT,
                          edge_weights=EDGE_WEIGHT_SECTION, depots=DEPOT_SECTION,
                          pickup_and_delivery=PICKUP_AND_DELIVERY_SECTION)
    prob.ncs = ncs
    prob.pds = pds
    prob.depot_id = depot_id
    prob.pd_dict = pd_dict
    prob.cal_map_real = cal_map_real
    prob.avail_veh_ids = veh_df.index  # 所有可用车辆id
    prob.tasks_list_p_tank = tasks_list_p_tank
    return prob


def solveLKHProblem(prob) -> list:
    """Solve LKH problem."""
    solver_path = './service/LKH/LKH'
    vehicles = prob.dimension
    routes = []
    try:
        output = lkh.solve(solver_path, problem=prob, vehicles=vehicles, runs=1)
        routes = [x for x in output if x != []]
        ub = len(routes)  # ub : upper bound
        vehicles = ub
        # print('vehicles = {} 有可行解!'.format(ub))
    except IndexError:
        print('WARNING! vehicles = dimension 无可行解')

    # vehicles = ub
    while vehicles > 1:
        try:
            vehicles -= 1
            output = lkh.solve(solver_path, problem=prob, vehicles=vehicles, runs=1)
            routes = [x for x in output if x != []]
            # print('vehicles = {} 有可行解!'.format(len(routes))
        except IndexError:
            # print('vehicles = {} 无可行解'.format(vehicles))
            break
    return routes


def createMIPProblem(routes_dict: dict, prob) -> Tuple[dict, OrderedDict[int, Tuple[str, int, int, str, str, float]]]:
    """Create MIP(Mixed Integer Programming) Problem"""
    # timetable_dict：按routes_dict的key存放车辆出发到达时间表timetable
    timetable_dict = {}
    tw_dict = {x[0]: x[2:5] for x in prob.pickup_and_delivery}
    for key, route in routes_dict.items():
        if len(route) <= 1:
            sec_node = route[0]
            tw = tw_dict[sec_node]
            start_time = tw[0] - prob.get_weight(0, sec_node - 1)
            end_time = tw[0] + tw[2] + prob.get_weight(0, sec_node - 1)
            timetable = [(start_time, start_time), (tw[0], tw[0] + tw[2]), (end_time, end_time)]
            timetable_dict[key] = timetable
        else:
            sec_node = route[0]
            start_time = tw_dict[sec_node][0] - prob.get_weight(0, sec_node - 1)
            timetable = [(start_time, start_time),
                         (tw_dict[sec_node][0], tw_dict[sec_node][0] + tw_dict[sec_node][2])]

            # 2nd node -> 3rd node -> 2nd to last node      
            leave_node = sec_node
            leave = tw_dict[sec_node][0] + tw_dict[sec_node][2]
            for arrive_node in route[1:]:
                arrive = leave + prob.get_weight(leave_node - 1, arrive_node - 1)
                tw = tw_dict[arrive_node]
                leave = max(arrive + tw[2], tw[0] + tw[2])
                timetable.append((arrive, leave))
                leave_node = arrive_node

            # 2nd to last node -> last node
            end_time = leave + prob.get_weight(leave_node - 1, 0)
            timetable.append((end_time, end_time))
            timetable_dict[key] = timetable
    # print('timetable_dict:', timetable_dict, len(timetable_dict))

    #  将timetable_dict转换为tasks_dict
    tasks_dict = OrderedDict()
    for key, val in timetable_dict.items():
        start = val[0][0]
        end = val[-1][0]
        hours = round((end - start) / 60, 2)
        tasks_dict[key] = 't' + str(key), start, end, 'd1', 'd1', hours
    return timetable_dict, tasks_dict


def outputResultJSON(routes: list, prob, pred_id: int, start_date: str, end_date: str) -> list:
    """Output result.json"""
    routes_dict = {x + 1: routes[x] for x in range(len(routes))}
    # print('routes_dict:', routes_dict, f'一共{len(routes_dict)}条路径')

    # create and solve mip problem for vehicle, save veh_sol as veh_sol_dict
    timetable_dict, veh_tasks_dict = createMIPProblem(routes_dict, prob)
    minVehNum, veh_sol = calMinVehNum(veh_tasks_dict, group_num=7, days=1)
    veh_sol_dict = {x + 1: veh_sol[x] for x in range(len(veh_sol))}
    # print('veh_sol_dict:', veh_sol_dict, f'一共{len(veh_sol_dict)}辆车')

    # create and solve mip problem for p_tank. save p_tank_sol as p_tank_sol_dict
    p_tank_tasks_dict = getTasksDict(prob.tasks_list_p_tank)
    minPTankNum, p_tank_sol = calMinVehNum(p_tank_tasks_dict, group_num=7, days=1)
    p_tank_sol_dict = {x + 1: p_tank_sol[x] for x in range(len(p_tank_sol))}  # 一期接口未使用
    # print('p_tank_sol_dict:', p_tank_sol_dict, f'一共{len(p_tank_sol_dict)}个罐')
    print('minVehNum:', minVehNum)
    print('minPTankNum:', minPTankNum)

    sd = np.datetime64(start_date)  # sd: start date
    ed = np.datetime64(end_date)  # ed: end date
    date_range = np.arange(sd, ed + np.timedelta64(1, 'D'), dtype='datetime64[D]')

    # 整合最终结果result
    result = []
    for date in date_range:
        pred_date = date.astype(str)
        for key, val in routes_dict.items():
            sub = {"pred_id": pred_id, "pred_date": pred_date, "depot_id": prob.depot_id, "veh_num": minVehNum,
                   "route_id": key}
            # sub = {"pred_id": pred_id, "pred_date": pred_date, "depot_id": prob.depot_id, "veh_num": minVehNum,
            #        "p_tank_num": minPTankNum, "route_id": key}       
            veh_id = getVehID(key, veh_sol_dict)  # 获取车辆id

            # 判断车辆是否在vehicle.xls中
            if veh_id in prob.avail_veh_ids:
                sub["veh_id"] = veh_id
            elif veh_id not in prob.avail_veh_ids:
                raise ValueError("车辆不足, 请在vehicle.xls中补充车辆后重新上传，并创建任务")

            # sub["p_tank_id"] = getPTankID(val, p_tank_sol_dict)  # 获取移动罐id

            # route_list 用于计算, r_route_list 用于result
            route_list = [1] + routes_dict[key] + [1]
            lat_lng_list = [prob.ncs[x] for x in route_list]
            veh_status_list = getVehStatusList(route_list, prob.pd_dict)
            timetable_list = getTimetableList(key, timetable_dict)
            r_route_list = [prob.cal_map_real[i] for i in route_list]

            if len(r_route_list[1:-1]) == 2 and r_route_list[1] == r_route_list[2]:
                r_route_list = r_route_list[:2] + r_route_list[3:]
                r_lat_lng_list = lat_lng_list[:2] + lat_lng_list[3:]
                r_veh_status_list = veh_status_list[:1] + [veh_status_list[1].split('-')[0]
                                                           + '-' + veh_status_list[2].split('-')[1]] \
                                                        + veh_status_list[3:]
                r_timetable_list = timetable_list[:1] + [timetable_list[1].split('-')[0]
                                                         + '-' + timetable_list[2].split('-')[1]] \
                                                      + timetable_list[3:]
            else:
                r_lat_lng_list = lat_lng_list
                r_veh_status_list = veh_status_list
                r_timetable_list = timetable_list

            sub["route"] = '-'.join([str(i) for i in r_route_list])
            sub["nodes_info"] = [{"node_id": r_route_list[x],
                                  "lng": str(r_lat_lng_list[x][1]),
                                  "lat": str(r_lat_lng_list[x][0]),
                                  "veh_status": r_veh_status_list[x],
                                  "timetable": r_timetable_list[x]} for x in range(len(r_route_list))]
            result.append(sub)
    return result


def saveResultJSON(result: list, pred_id: int, start_date: str, end_date: str):
    """Save result.json on the server."""
    sd = start_date.replace('-', '')
    ed = end_date.replace('-', '')
    result_file = f'./service/result/{pred_id}_{sd}_{ed}.json'
    with open(result_file, 'w+') as f:
        json.dump(result, f)
    result_json_path = os.getcwd() + f'/service/result/{pred_id}_{sd}_{ed}.json'
    print('result_json_path:', result_json_path)


def main(pred_id: int, start_date: str, end_date: str,
         depot_ids: list, demand_ids: list) -> list:
    """ Return result json."""
    prob = createLKHProblem(depot_ids, demand_ids)
    routes = solveLKHProblem(prob)
    result = outputResultJSON(routes, prob, pred_id, start_date, end_date)
    saveResultJSON(result, pred_id, start_date, end_date)
    return result
