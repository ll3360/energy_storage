from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

import scheme_calculator


def getNodeName(start_node, end_node, dep_df, dem_df):
    if isinstance(start_node, str) and isinstance(end_node, int):
        return dep_df.loc[start_node, 'depot_name'], dem_df.loc[end_node, 'demand_name']
    elif isinstance(start_node, int) and isinstance(end_node, str):
        return dem_df.loc[start_node, 'demand_name'], dep_df.loc[end_node, 'depot_name']
    else:
        raise ValueError(f'Incorrect Node ID start_node = {start_node}, end_node = {end_node}')


def getTaskTypeAndVehStatus(start_node, end_node, duration, scheme):
    if isinstance(start_node, str) and isinstance(end_node, int):
        demand_id = end_node
        model = scheme.models[demand_id]
        if duration == model.none_p_tank_judge:
            return '去程', '满罐'
        elif duration < model.none_p_tank_judge:
            return '去程', '空车头'
    elif isinstance(start_node, int) and isinstance(end_node, str):
        demand_id = start_node
        model = scheme.models[demand_id]
        if duration == model.none_p_tank_judge:
            return '回程', '冷罐'
        elif duration < model.none_p_tank_judge:
            return '回程', '空车头'
    else:
        raise


def getTaskTypeAndPTankStatus(start_node, end_node):
    if isinstance(start_node, str) and isinstance(end_node, int):
        return '供热', '满罐-运输-供热-冷罐'
    elif isinstance(start_node, int) and isinstance(end_node, str):
        return '充热', '冷罐-运输-充热-满罐'
    else:
        raise


def numToDateAndTime(start_date: str, minutes: int):
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')

    # 计算结束时间的datetime对象
    end_datetime = start_datetime + timedelta(minutes=minutes)

    # 提取Date和Time对象
    end_date = end_datetime.date()
    end_time = end_datetime.time()
    return end_date, end_time


def getDfVehSchedule(scheme, dep_df, dem_df, start_date):
    columns = ['储能车编号', '起点', '终点', '出发日期', '出发时间', '到达日期', '到达时间', '任务类型', '车辆状态']
    sol = sorted(scheme.tractor_sol, key=lambda x: x[0])
    tasks_list = scheme.tasks_list
    data = []
    for i in range(len(sol)):
        veh_id = 'v' + str(i + 1)
        for task_id in sol[i]:
            start_t, end_t, start_node, end_node = tasks_list[task_id - 1]  # task_id - 1 为task_list中的索引
            start_node_name, end_node_name = getNodeName(start_node, end_node, dep_df, dem_df)
            start_datetime = numToDateAndTime(start_date, start_t)
            end_datetime = numToDateAndTime(start_date, end_t)
            duration = end_t - start_t
            task_type, veh_status = getTaskTypeAndVehStatus(start_node, end_node, duration, scheme)
            sub = [veh_id, start_node_name, end_node_name, start_datetime[0], start_datetime[1],
                   end_datetime[0], end_datetime[1], task_type, veh_status]
            data.append(sub)
    data = np.array(data)
    df = pd.DataFrame(data=data, columns=columns)
    return df


def getDfPTankSchedule(scheme, dep_df, dem_df, start_date):
    columns = ['移动罐编号', '起点', '终点', '开始日期', '开始时间', '结束日期', '结束时间', '任务类型', '移动罐状态']
    sol = sorted(scheme.trailer_sol, key=lambda x: x[0])
    tasks_list = scheme.tasks_list_trailer
    data = []
    for i in range(len(sol)):
        p_tank_id = 'p' + str(i + 1)
        for task_id in sol[i]:
            task = tasks_list[task_id - 1]  # task_id - 1 为task_list中的索引
            start_t, end_t, start_node, end_node = task
            start_node_name, end_node_name = getNodeName(start_node, end_node, dep_df, dem_df)
            start_datetime = numToDateAndTime(start_date, start_t)
            end_datetime = numToDateAndTime(start_date, end_t)
            task_type, p_tank_status = getTaskTypeAndPTankStatus(start_node, end_node)
            sub = [p_tank_id, start_node_name, end_node_name, start_datetime[0], start_datetime[1],
                   end_datetime[0], end_datetime[1], task_type, p_tank_status]
            data.append(sub)
    data = np.array(data)
    df = pd.DataFrame(data=data, columns=columns)
    return df


def writeUrlToCell(worksheet, col_name, df, df_sched_dict, url_format):
    """Write a hyperlink to a worksheet cell."""
    col_index = df.columns.get_loc(col_name)
    for sheetname in df_sched_dict.keys():
        row_index = df.loc[df[col_name] == sheetname].index[0]
        worksheet.write_url(row_index + 1, col_index, f'internal:{sheetname}!A1',
                            cell_format=url_format, string=sheetname)


def writeSchedWorksheet(writer, df_sched_dict, col_index1, col_name1, col_index2, col_name2, time_format):
    for sheetname, df_sched in df_sched_dict.items():
        df_sched.to_excel(writer, sheet_name=sheetname, index=False)
        sched_worksheet = writer.sheets[sheetname]

        sched_worksheet.write_column(col_index1, df_sched[col_name1], time_format)
        sched_worksheet.write_column(col_index2, df_sched[col_name2], time_format)
        sched_worksheet.autofit()


def getDataFrame(dem_df, dep_df, all_depot_ids, all_demand_ids, all_mode_ids, all_days, start_date):
    mode_names = {0: '仅储能车模式', 2: '两罐模式', 3: '三罐模式'}
    combos = [(depot_ids, demand_ids, [mode_id] * len(demand_ids), days)
              for depot_ids in all_depot_ids
              for demand_ids in all_demand_ids
              for mode_id in all_mode_ids
              for days in all_days]

    # 列名
    columns = ['热源(供热点)', '用户侧(用热点)', '运营模式', '运营天数', '固定罐数',
               '牵引车数', '移动罐数', '总成本(万元)', '总利润(万元)', '车辆调度', '移动罐调度']
    df = pd.DataFrame(columns=columns)
    df_veh_sched_dict = {}  # 车辆调度dict[str:DataFrame]
    df_p_tank_sched_dict = {}  # 移动罐调度dict[str:DataFrame]
    for combo in tqdm(combos):
        depot_ids = combo[0]
        demand_ids = combo[1]
        mode_ids = combo[2]
        days = combo[3]
        depot_name = dep_df.loc[depot_ids[0], 'depot_name']
        demand_name = '\n'.join([dem_df.loc[x, 'demand_name'] for x in demand_ids])
        mode_name = mode_names[mode_ids[0]]
        combo_str = '_'.join([str(j) for i in combo for j in (i if isinstance(i, list) else [i])])
        veh_sched_sheetname = 'v_' + combo_str
        p_tank_sched_sheetname = 'p_' + combo_str

        # 求解
        scheme = scheme_calculator.calBestScheme(depot_ids, demand_ids, mode_ids, days)
        new_row = pd.DataFrame([[depot_name, demand_name, mode_name, days, scheme.tank_num,
                                 scheme.tractor_num, scheme.trailer_num, round(scheme.total_cost, 6),
                                 round(scheme.total_profit, 6), veh_sched_sheetname, p_tank_sched_sheetname]],
                               columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

        # 车辆调度df
        df_veh_sched_dict[veh_sched_sheetname] = getDfVehSchedule(scheme, dep_df, dem_df, start_date)

        # 移动罐调度df
        df_p_tank_sched_dict[p_tank_sched_sheetname] = getDfPTankSchedule(scheme, dep_df, dem_df, start_date)
    return df, df_veh_sched_dict, df_p_tank_sched_dict


def writeExcel(file_name, df, df_veh_sched_dict, df_p_tank_sched_dict, all_demand_ids, all_mode_ids, all_days):
    row, col = df.shape
    merge_range = len(all_mode_ids) * len(all_days)  # 1 * 4
    days_range = len(all_days)  # 4
    
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='运营方案', index=False)

        # 获取工作簿
        workbook = writer.book

        # 获取工作表
        worksheet = writer.sheets['运营方案']

        # 背景色
        bg_blue = workbook.add_format({'bg_color': '#BDD6E9'})  # 浅蓝
        bg_green = workbook.add_format({'bg_color': '#C7DDB6'})  # 浅绿

        # 填充列名背景色(蓝色)
        worksheet.conditional_format('A1:K1', {'type': 'no_errors', 'format': bg_blue})

        # 合并格式
        merge_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True})

        for i in range(len(all_demand_ids)):
            # 合并单元格
            worksheet.merge_range('A{}:A{}'.format(merge_range * i + 2, merge_range * i + merge_range + 2 - 1),
                                  df.loc[i * merge_range, '热源(供热点)'], merge_format)
            worksheet.merge_range('B{}:B{}'.format(merge_range * i + 2, merge_range * i + merge_range + 2 - 1),
                                  df.loc[i * merge_range, '用户侧(用热点)'], merge_format)
            longest_day_cost = {}
            for j in range(len(all_mode_ids)):
                worksheet.merge_range(
                    'C{}:C{}'.format(merge_range * i + days_range * j + 2,
                                     merge_range * i + days_range * j + days_range + 2 - 1),
                    df.loc[merge_range * i + days_range * j, '运营模式'], merge_format)
                longest_day_cost[merge_range * i + days_range * j + days_range + 2 - 1] = df.loc[
                    merge_range * i + days_range * j + days_range - 1, '总成本(万元)']

            # 填充背景色(最长天数的总成本和总利润)
            worksheet.conditional_format('H{}:I{}'.format(*[1 + days_range * (i + 1)] * 2),
                                         {'type': 'no_errors', 'format': bg_green})

        # 自动换行
        wrap_format = workbook.add_format({'text_wrap': True})
        worksheet.set_column(0, col - 1, cell_format=wrap_format)

        # 添加全框线
        border_format = workbook.add_format({'border': 1})
        worksheet.conditional_format(0, 0, row, col - 1, {'type': 'no_errors', 'format': border_format})

        # 写超链接
        url_format = workbook.add_format({'color': 'blue', 'underline': 1})
        writeUrlToCell(worksheet, '车辆调度', df, df_veh_sched_dict, url_format)
        writeUrlToCell(worksheet, '移动罐调度', df, df_p_tank_sched_dict, url_format)

        # 时间格式
        time_format = workbook.add_format({'num_format': 'hh:mm'})

        # 写车辆调度表
        writeSchedWorksheet(writer, df_veh_sched_dict, 'E2', '出发时间', 'G2', '到达时间', time_format)

        # 写移动罐调度表
        writeSchedWorksheet(writer, df_p_tank_sched_dict, 'E2', '开始时间', 'G2', '结束时间', time_format)

        # 自动调整列宽
        worksheet.autofit()


def main():
    # 运营方案开始日期
    start_date = '2023-04-10'

    # 期望的组合(所有供热点, 所有用热点, 所有模式, 所有天数)
    all_depot_ids = [['d1']]
    # all_demand_ids = [[1],[2],[1,2],[1,2,3],[1,2,4],[1,2,3,4],[1,2,3,4,5]]
    all_demand_ids = [[1]]
    all_mode_ids = [0]  # 目前只优化了模式0
    all_days = [1, 7, 14, 30]

    depot_file_path = './xls/depot_new.xls'
    demand_file_path = './xls/demand_new.xls'
    dep_df = pd.read_excel(depot_file_path, index_col='depot_id')
    dem_df = pd.read_excel(demand_file_path, index_col='demand_id')
    df, df_veh_sched_dict, df_p_tank_sched_dict = getDataFrame(dem_df, dep_df, all_depot_ids, 
                                                               all_demand_ids, all_mode_ids, all_days, start_date)
    result_file_name = 'report.xls'
    writeExcel(result_file_name, df, df_veh_sched_dict, df_p_tank_sched_dict, 
               all_demand_ids, all_mode_ids, all_days)


if __name__ == '__main__':
    main()
