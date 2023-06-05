from collections import OrderedDict
from typing import List, Tuple, Union

from gurobipy import *


def getTasksDict(tasks_list: List[Tuple[int, int, Union[int, str], Union[int, str]]],
                 start_index: int = 1) -> OrderedDict[int, Tuple[str, int, int, Union[int, str], Union[int, str],
                                                                 float]]:
    """Return tasks_dict"""
    tasks_dict = OrderedDict()
    for i in range(len(tasks_list)):
        key = i + start_index
        start_t = tasks_list[i][0]
        end_t = tasks_list[i][1]
        start_node = tasks_list[i][2]
        end_node = tasks_list[i][3]
        hours = round((end_t - start_t) / 60, 2)
        if hours <= 0:
            raise ValueError(f'存在hours <= 0, hours: {hours}, start_t: {start_t}, end_t: {end_t}')
        tasks_dict[key] = 't' + str(key), start_t, end_t, start_node, end_node, hours
    return tasks_dict


def calMinVehNum(task: OrderedDict[int, Tuple[str, int, int, Union[int, str], Union[int, str], float]],
                 group_num: int, days: Union[int, float]) -> Tuple[int, List[List[int]]]:
    """Calculate the best solution of mixed integer programming problem.
    Return the objective and the best solution.
    """
    while True:
        try:
            xindex = {}  # xindex = {key:value}, where key = (i,j,k) and value = 工时(弧(i,j,k)的cost)
            yindex = {x: 0 for x in range(group_num)}  # 初始化
            for k in range(group_num):  # 预处理, 筛选符合约束(2)的xindex[i,j,k]
                xindex[0, len(task) + 1, k] = 0  # x[0,len(task+1),k] = 0 以0开头的弧(i,j,k)的工时为0
                for i in task.keys():
                    xindex[0, i, k] = 0
                    xindex[i, len(task) + 1, k] = task[i][5]  # 以len(task)+1结尾的弧(i,j,k)的工时
                    for j in task.keys():
                        if i != j and task[i][2] <= task[j][1] <= task[i][2] + 1440 * max(2.0, days / 4) \
                                and task[i][4] == task[j][3]:  # i不等于j,且任务i的结束时间早于任务j的开始时间
                            # if i != j and task[i][2] <= task[j][1] and task[i][4] == task[j][3]:  # 去掉<=最晚时间约束
                            xindex[i, j, k] = task[i][5]  # 工时

            # 建立运筹优化模型
            m = Model('model')
            x = m.addVars(xindex.keys(), vtype=GRB.BINARY, name='x')  # 添加决策变量x[i,j,k] = 0 or 1
            y = m.addVars(yindex.keys(), obj=100, vtype=GRB.BINARY, name='y')  # 添加决策变量y[k] = 0 or 1
            u = m.addVar(obj=1, lb=0.0, ub=24.0 * days, vtype=GRB.CONTINUOUS, name='u', column=None)  # 变量u
            v = m.addVar(obj=-1, lb=0.0, ub=24.0 * days, vtype=GRB.CONTINUOUS, name='v', column=None)  # 变量v

            # 设定优化目标函数 minimize Σy[k], where k ∈ W
            # m.setObjective(y.sum()+u-v, GRB.MINIMIZE)
            # m.setObjective(y.sum(), GRB.MINIMIZE)

            # 约束(1): 保证每一个生产块都被分配 ΣΣx[i,j,k] = 1
            m.addConstrs((x.sum(i, '*', '*') == 1 for i in task.keys()), name='c1')

            # 约束(2): 生产块是否连接, 通过预处理完成

            # 约束(3): 班组是否参加生产, 保证在y[i[2]] = 0(即y[k] = 0)时, x[i] = 0
            m.addConstrs((x[i] <= y[i[2]] for i in xindex.keys() if i[0] != 0 or i[1] != len(task) + 1), name='c3')

            # 约束(4): 生产块连接约束(流平衡)
            for k in range(group_num):
                m.addConstr((x.sum(0, '*', k) == 1), name='c4_1')
                m.addConstr((x.sum('*', len(task) + 1, k) == 1), name='c4_2')
                for i in task.keys():
                    m.addConstr((x.sum(i, '*', k) == x.sum('*', i, k)), name='c4_3')

            # # 约束(5): 最长工作时间(24H)约束
            # # 约束(6): 最短工作时间约束
            # m.addConstrs(24 >= x.prod(xindex, '*', '*', k) for k in range(groupNum))
            # m.addConstrs(days*24 >= x.prod(xindex, '*', '*', k) for k in range(groupNum))
            # M = 10 ** 6
            M = 24.0 * days
            for k in range(group_num):
                m.addConstr((u >= x.prod(xindex, '*', '*', k)), name='c5')
                m.addConstr((v <= (1 - y[k]) * M + x.prod(xindex, '*', '*', k)), name='c6')

            # 设置其他参数
            m.Params.OutputFlag = 0
            # m.setParam('TimeLimit',60)

            m.Params.MIPGap = 0.085
            m.Params.MIPFocus = 1
            m.Params.Method = 4
            m.Params.NodeMethod = 2

            m.optimize()
            m.getAttr("ObjVal")
            break
        except AttributeError:
            group_num += 1
            if group_num > len(task):
                raise ValueError(f'group_num > len(task)模型仍然不可解\ntask:{task}\ngroup_num:{group_num}')

    # 保存模型
    # m.write('mip.lp')

    # 优化结果
    bestX = {i: x[i].X for i in xindex.keys() if x[i].X > 0.9 and (i[0] != 0 or i[1] != len(task) + 1)}
    bestY = {k: y[k].X for k in range(group_num) if y[k].X > 0.9}

    # 临时变量tmp_sol
    tmp_sol = {j: [i[:-1] for i in bestX.keys() if i[2] == j] for j in bestY.keys()}

    # 临时变量tmp2_sol, tmp3_sol
    tmp2_sol = {}
    for key, val in tmp_sol.items():
        val_tl = tuplelist(val)
        start = 0
        end = len(task) + 1
        tmp3_sol = tuplelist([])
        while start != end:
            tmp = val_tl.select(start, '*')
            tmp3_sol += tmp
            start = tmp[0][1]
        tmp2_sol[key] = list(tmp3_sol)

    # 最优解obj, sol
    # obj = m.ObjVal
    sol = [[i[0] for i in v][1:] for v in tmp2_sol.values()]
    # print(f"obj:{obj}\nsol:{sol}")

    # 最小车辆数
    min_veh_num = len(sol)
    return min_veh_num, sol  # 返回最小车辆数, 最优解
