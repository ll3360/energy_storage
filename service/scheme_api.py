import json
import math
import os
from copy import deepcopy
from datetime import datetime, timedelta
# from pprint import pprint
from typing import List, Tuple

import pandas as pd

from service import scheme_calculator
from service.pred_api import DEPOT_FILE, DEMAND_FILE

DATE_FORMAT = '%Y/%m/%d'
TIMESTAMP_FORMAT = '%Y/%m/%d %H:%M:%S'


def transferToExpectedResult(json_file, new_start_datetime):
    """Transfer existed result o expected result."""
    old_start_timestamp = json_file['start_timestamp']
    old_start_datetime = datetime.strptime(old_start_timestamp, TIMESTAMP_FORMAT)
    delta = new_start_datetime - old_start_datetime  # 计算时间间隔

    result = deepcopy(json_file)
    result['start_timestamp'] = new_start_datetime.strftime(TIMESTAMP_FORMAT)
    new_scheme = result['scheme']
    for idx in range(len(new_scheme)):
        new_nodes_info = new_scheme[idx]['nodes_info']
        new_scheme[idx]['scheme_date'] = (
                datetime.strptime(new_nodes_info[0]['timestamp'], TIMESTAMP_FORMAT) + delta).strftime(DATE_FORMAT)
        for i in range(len(new_nodes_info)):
            old_timestamp_str_list = new_nodes_info[i]['timestamp'].split('-')
            new_timestamp_str_list = [(datetime.strptime(x, TIMESTAMP_FORMAT) + delta).strftime(TIMESTAMP_FORMAT) for x
                                      in old_timestamp_str_list]
            new_nodes_info[i]['timestamp'] = '-'.join(new_timestamp_str_list)
    return result


def checkExistedResult(depot_ids: list, demand_ids: list, mode_ids: list, days: int, start_datetime: datetime):
    """Check whether the result had already been calculated. Return the result if it had
    been calculated, else return zero."""
    depot_ids_str, demand_ids_str, mode_ids_str = encodeDepotIdsDemandIdsModeIds(depot_ids, demand_ids, mode_ids)
    target_file = f'scheme_{depot_ids_str}_{demand_ids_str}_{mode_ids_str}_{days}.json'
    result_folder = os.path.join(os.getcwd(), 'service', 'result', 'scheme_api_result')
    existed_result_files = [f for f in os.listdir(result_folder) if f.startswith('scheme_') and f.endswith('.json')]

    # 判断是否计算过
    if target_file in existed_result_files:
        # 调用现有结果JSON, 再把开始日期变为输入的开始日期
        target_file_path = os.path.join(result_folder, target_file)
        print('已计算过', target_file)
        with open(target_file_path, 'r') as f:
            json_file = json.load(f)
        result = transferToExpectedResult(json_file, start_datetime)
        return result
    else:
        print('未计算过')
        return 0


def encodeDepotIdsDemandIdsModeIds(depot_ids, demand_ids, mode_ids) -> Tuple[str, str, str]:
    """Encode depot_ids and demand_ids and mode_ids in numeric order(ascending), return encoded string."""
    sorted_depot_ids = sorted(depot_ids, key=lambda x: int(x.lstrip('d')))
    depot_ids_str = '-'.join([str(x) for x in sorted_depot_ids])
    sorted_demand_ids_mode_ids = sorted(list(zip(demand_ids, mode_ids)), key=lambda x: int(x[0]))  # 元素转成整数排序
    demand_ids_str = '-'.join([str(x[0]) for x in sorted_demand_ids_mode_ids])  # 元素转回str,连接起来
    mode_ids_str = '-'.join([str(x[1]) for x in sorted_demand_ids_mode_ids])
    return depot_ids_str, demand_ids_str, mode_ids_str


def toTimestamp(start_date, minutes):
    return (start_date + timedelta(minutes=minutes)).strftime(TIMESTAMP_FORMAT)


def getVehStatusAndTimeStamp(start_datetime, start_time, end_time, start_node, end_node, scheme):
    """Get veh_status"""
    status1 = "1"  # 出发-满罐
    status2 = "2"  # 到达-满罐
    status3 = "3"  # 出发-车头
    status4 = "4"  # 到达-车头
    status5 = "5"  # 出发-冷罐
    status6 = "6"  # 到达-冷罐

    start_timestamp = toTimestamp(start_datetime, start_time)
    duration = end_time - start_time
    # 判断去程或返程
    if isinstance(start_node, str) and isinstance(end_node, int):
        demand_id = end_node
        model = scheme.models[demand_id]
        # 判断是否带罐
        if duration == model.none_p_tank_judge:  # 出发-满罐, 到达-满罐
            # 到达用户侧的时间(分钟) = 去程任务结束时间(分钟) - 服务时间(带罐去)(分钟)
            arr_time = end_time - model.service_time_with_p
            arr_timestamp = toTimestamp(start_datetime, arr_time)
            return start_timestamp, arr_timestamp, status1, status2
        elif duration < model.none_p_tank_judge:  # 出发-车头, 到达-车头
            # 到达用户侧的时间(分钟) = 去程任务结束时间(分钟) - 服务时间(空车头去)(分钟)
            arr_time = end_time - model.service_time_without_p
            arr_timestamp = toTimestamp(start_datetime, arr_time)
            return start_timestamp, arr_timestamp, status3, status4
        else:
            raise
    elif isinstance(start_node, int) and isinstance(end_node, str):
        demand_id = start_node
        model = scheme.models[demand_id]
        # 判断是否带罐
        if duration == model.none_p_tank_judge:  # 出发-冷罐, 到达-冷罐
            # 返回热源的时间(分钟) = 回程任务结束时间(分钟) - 甩挂时间(带罐回)(分钟)
            arr_time = end_time - model.drop_and_pull_time
            arr_timestamp = toTimestamp(start_datetime, arr_time)
            return start_timestamp, arr_timestamp, status5, status6
        elif duration < model.none_p_tank_judge:  # 出发-车头, 到达-车头
            # 返回热源的时间(分钟) = 回程任务结束时间(分钟) - 甩挂时间(空车头回)(分钟)
            arr_time = end_time - math.ceil(model.drop_and_pull_time / 2)
            arr_timestamp = toTimestamp(start_datetime, arr_time)
            return start_timestamp, arr_timestamp, status3, status4
        else:
            raise
    else:
        raise


def getWithPTank(start_time, start_node, end_node, sol_dict):
    for key, val in sol_dict.items():
        for i in val:
            if start_time == i[0] and start_node == i[2] and end_node == i[3]:
                val.remove(i)
                return key, sol_dict
    raise TypeError(f'没有从sol_dict: {sol_dict}中找到\n{start_time, start_node, end_node}')
    # return "NONE", sol_dict


def saveResultJSON(result, depot_ids: List[str], demand_ids: List[int], mode_ids: List[int], days: int):
    # """Save result.json on the server."""
    depot_ids_str, demand_ids_str, mode_ids_str = encodeDepotIdsDemandIdsModeIds(depot_ids, demand_ids, mode_ids)
    result_folder = os.path.join(os.getcwd(), 'service', 'result', 'scheme_api_result')
    result_filename = f'scheme_{depot_ids_str}_{demand_ids_str}_{mode_ids_str}_{days}.json'
    file_path = os.path.join(result_folder, result_filename)
    with open(file_path, 'w+') as f:
        json.dump(result, f)
    # print('saving...\nresult_json_path:', file_path)


def outputResultJSON(scheme, start_datetime) -> dict:
    # 按照第一条任务的id大小重排牵引车的解
    tractor_sol = sorted(scheme.tractor_sol, key=lambda x: x[0])
    veh_sol_dict = {}
    for i in range(len(tractor_sol)):
        veh_id = 'v' + str(i + 1)
        veh_tasks = [scheme.tasks_list[j - 1] for j in tractor_sol[i]]
        veh_sol_dict[veh_id] = veh_tasks

    trailer_sol = sorted(scheme.trailer_sol, key=lambda x: x[0])
    p_tank_sol_dict = {}
    for i in range(len(trailer_sol)):
        p_tank_id = 'p' + str(i + 1)
        p_tank_tasks = [scheme.tasks_list_trailer[j - 1] for j in trailer_sol[i]]
        p_tank_sol_dict[p_tank_id] = p_tank_tasks
    # print('p_tank_sol_dict:', p_tank_sol_dict)

    scheme_data = []
    task_id = 1
    dep_df = pd.read_excel(DEPOT_FILE, index_col='depot_id')
    dem_df = pd.read_excel(DEMAND_FILE, index_col='demand_id')
    for key, val in veh_sol_dict.items():
        veh_id = key  # 车辆id
        for dep_task, ret_task in zip(val[::2], val[1::2]):  # 两两合为一整个任务, 若总个数为奇数, 则不算val[-1]
            dep_start_time, dep_end_time, dep_start_node, dep_end_node = dep_task  # 去程任务
            ret_start_time, ret_end_time, ret_start_node, ret_end_node = ret_task  # 回程任务

            # 检查节点一致性
            if dep_start_node != ret_end_node:
                raise ValueError(f'去程开始节点{dep_start_node}与回程结束节点{ret_end_node}不一致')
            elif dep_end_node != ret_start_node:
                raise ValueError(f'去程结束节点{dep_end_node}与回程开始节点{ret_start_node}不一致')

            # 去程开始及到达时间戳, 去程开始及到达车辆状态
            dep_start_timestamp, dep_arr_timestamp, dep_start_veh_status, dep_arr_veh_status = getVehStatusAndTimeStamp(
                start_datetime, dep_start_time, dep_end_time, dep_start_node, dep_end_node, scheme)

            # 回程开始车辆状态, 回程到达车辆状态及时间
            ret_start_timestamp, ret_arr_timestamp, ret_start_veh_status, ret_arr_veh_status = getVehStatusAndTimeStamp(
                start_datetime, ret_start_time, ret_end_time, ret_start_node, ret_end_node, scheme)
            # print(dep_start_veh_status, dep_arr_veh_status, ret_start_veh_status, ret_arr_veh_status)

            appendable = True # 可以执行scheme_data.append(sub)
            try:
                if dep_start_veh_status == '3' and dep_arr_veh_status == '4':
                    # print(dep_start_veh_status, dep_arr_veh_status, ret_start_veh_status, ret_arr_veh_status)
                    dep_with_p_tank = 'NONE'
                    ret_with_p_tank, p_tank_sol_dict = getWithPTank(ret_start_time, ret_start_node, ret_end_node,
                                                                    p_tank_sol_dict)

                elif ret_start_veh_status == '3' and ret_arr_veh_status == '4':
                    # print(dep_start_veh_status, dep_arr_veh_status, ret_start_veh_status, ret_arr_veh_status)
                    dep_with_p_tank, p_tank_sol_dict = getWithPTank(dep_start_time, dep_start_node, dep_end_node,
                                                                    p_tank_sol_dict)
                    ret_with_p_tank = 'NONE'

                else:
                    # print(dep_task, dep_start_veh_status, dep_arr_veh_status)
                    # print(ret_task, ret_start_veh_status, ret_arr_veh_status)
                    dep_with_p_tank, p_tank_sol_dict = getWithPTank(dep_start_time, dep_start_node, dep_end_node,
                                                                    p_tank_sol_dict)
                    ret_with_p_tank, p_tank_sol_dict = getWithPTank(ret_start_time, ret_start_node, ret_end_node,
                                                                    p_tank_sol_dict)
                    # print(veh_id, '>>>dep:', dep_with_p_tank, 'ret:', ret_with_p_tank, '\n')
            except TypeError:
                appendable = False  # 不执行scheme_data.append(sub)
            
            if appendable:
                # 方案日期
                scheme_date = (start_datetime + timedelta(minutes=dep_start_time)).strftime(DATE_FORMAT)

                if dep_start_veh_status not in ['1', '3', '5'] or ret_start_veh_status not in ['1', '3', '5']:
                    raise ValueError('1')
                elif dep_arr_veh_status not in ['2', '4', '6'] or ret_arr_veh_status not in ['2', '4', '6']:
                    raise ValueError('2')
                elif dep_start_veh_status != '3' and dep_with_p_tank == 'NONE':
                    raise ValueError('3')
                elif dep_arr_veh_status != '4' and dep_with_p_tank == 'NONE':
                    raise ValueError('4')
                elif ret_start_veh_status != '3' and ret_with_p_tank == 'NONE':
                    raise ValueError('5')
                elif ret_arr_veh_status != '4' and ret_with_p_tank == 'NONE':
                    raise ValueError('6')

                # 路径
                route = '-'.join([str(dep_start_node), str(dep_end_node), str(ret_end_node)])

                # 节点信息
                nodes_info = [
                    {"node_id": dep_start_node,
                     "lng": str(dep_df.loc[dep_start_node, 'lng']),
                     "lat": str(dep_df.loc[dep_start_node, 'lat']),
                     "veh_status": dep_start_veh_status,
                     "timestamp": dep_start_timestamp,
                     "with_p_tank": dep_with_p_tank
                     },
                    {"node_id": dep_end_node,
                     "lng": str(dem_df.loc[dep_end_node, 'lng']),
                     "lat": str(dem_df.loc[dep_end_node, 'lat']),
                     "veh_status": '-'.join((dep_arr_veh_status, ret_start_veh_status)),
                     "timestamp": '-'.join((dep_arr_timestamp, ret_start_timestamp)),
                     "with_p_tank": '-'.join((dep_with_p_tank, ret_with_p_tank))
                     },
                    {"node_id": ret_end_node,
                     "lng": str(dep_df.loc[ret_end_node, 'lng']),
                     "lat": str(dep_df.loc[ret_end_node, 'lat']),
                     "veh_status": ret_arr_veh_status,
                     "timestamp": ret_arr_timestamp,
                     "with_p_tank": ret_with_p_tank
                     }]
                sub = {"scheme_date": scheme_date,
                       "task_id": task_id,
                       "veh_id": veh_id,
                       "route": route,
                       "nodes_info": nodes_info}
                task_id += 1
                scheme_data.append(sub)

    # print('p_tank_sol_dict:', p_tank_sol_dict)
    assert all(p_tank_sol_dict), str(p_tank_sol_dict)
    result = {"veh_num": scheme.tractor_num,
              "p_tank_num": scheme.trailer_num,
              "start_timestamp": toTimestamp(start_datetime, minutes=0),
              "scheme": scheme_data}
    return result


def main(start_date: str, end_date: str, depot_ids: List[str], demand_ids: List[int]) -> dict:
    """ Return result json."""

    # 得到运营方案的天数days
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    if start_datetime < end_datetime:
        days = (end_datetime - start_datetime).days + 1  # [开始日期, 结束日期] 前闭后闭
    elif start_datetime == end_datetime:
        days = 1
    else:
        raise ValueError('结束日期不能早于开始日期')

    # 模式0--仅储能车模式
    mode_ids = [0 for _ in range(len(demand_ids))]  # 目前只考虑模式0

    # 检查是否计算过, 如果计算过, 则直接调用存储在result_folder中的JSON
    result = checkExistedResult(depot_ids, demand_ids, mode_ids, days, start_datetime)
    if result == 0:
        scheme = scheme_calculator.calBestScheme(depot_ids, demand_ids, mode_ids, days)
        result = outputResultJSON(scheme, start_datetime)
        saveResultJSON(result, depot_ids, demand_ids, mode_ids, days)
    return result
