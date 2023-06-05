from flask import Flask, request, jsonify
from service import pred_api
from service import create_api
from service import scheme_api


# 创建APP对象
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# 接口1: 新建预测任务,上传四个Excel到指定路径
@app.route('/es/create', methods=['POST'])
def upload():
    if request.method != "POST":
        return "请求方法错误.", 400

    if not request.content_type.startswith('multipart/form-data'):
        return "Content-Type错误.", 400

    files = request.files.to_dict()

    # 检查上传文件数目
    if len(files) != 4:
        return "文件数目错误, 请上传且只上传四个文件.", 400

    # 检查上传文件名
    expected_names_list = ["depot.xls", "demand.xls", "s_tank.xls", "vehicle.xls"]
    expected_names = set(expected_names_list)
    uploaded_names = set(file.filename for name, file in files.items())
    if expected_names != uploaded_names:
        return f"文件名错误, 请检查文件名是否与{', '.join(expected_names_list)}一致.", 400

    # 上传到指定路径           
    try:
        dst_folder_path = '/home/dynamic/energy_storage/service/xls'
        for name, file in files.items():
            file.save(dst_folder_path + '/{}'.format(file.filename))
        pred_id = create_api.main()
        result = jsonify({"code": 200, "message": "success", "pred_id": pred_id})
    except Exception as e:
        result = jsonify({"code": 500, "message": f"failure! {e}"})
    return result


# 接口2: 预测接口
@app.route('/es/pred', methods=['POST'])
def cal_opt_sol():
    # 检查请求方法
    if request.method != "POST":
        return "请求方法错误.", 400

    # 检查 Content-Type
    if not request.content_type.startswith('application/json'):
        return "Content-Type错误.", 400

    # 获取参数
    pred_id = request.json.get("pred_id")
    start_date = request.json.get("start_date")
    end_date = request.json.get("end_date")
    depot_ids = request.json.get("depot_ids")
    demand_ids = request.json.get("demand_ids")

    # 调用算法程序
    try:
        data = pred_api.main(pred_id, start_date, end_date, depot_ids, demand_ids)
        result = jsonify({"code": 200, "message": "success", "data": data})
    except Exception as e:
        result = jsonify({"code": 500, "message": f"failure! {e}"})
    return result


# 接口3: 经济最优解的供热方案接口
@app.route('/es/scheme', methods=['POST'])
def cal_best_scheme():
    # 检查请求方法
    if request.method != "POST":
        return "请求方法错误.", 400

    # 检查Content-Type
    if not request.content_type.startswith('application/json'):
        return "Content-Type必须是application/json", 400

    # 检查必传参数和值类型
    required_params = ["start_date", "end_date", "depot_ids", "demand_ids"]
    params_types = [str, str, list, list]
    for param, param_type in zip(required_params, params_types):
        if param not in request.json:
            return f"{param}是必传参数", 400
        elif not isinstance(request.json[param], param_type):
            return f"{param}参数类型错误", 400

    # 获取参数
    start_date = request.json.get("start_date")
    end_date = request.json.get("end_date")
    depot_ids = request.json.get("depot_ids")
    demand_ids = request.json.get("demand_ids")   

    # 调用算法程序
    try:
        data = scheme_api.main(start_date, end_date, depot_ids, demand_ids)
        result = jsonify({"code": 200, "message": "success", "data": data})
    except Exception as e:
        result = jsonify({"code": 500, "message": f"failure! {e}"})
    return result


if __name__ == '__main__':
    # app.run(threaded=True, host='10.78.24.123', debug=True, port=9986, use_reloader=False)  # 服务器ip 后台运行
    # app.run(threaded=True, host='10.78.24.123', debug=True, port=9986) # 服务器ip 终端窗口运行
    app.run(threaded=True, host='127.0.0.1', debug=True, port=5001)  # 本地ip 终端窗口运行
