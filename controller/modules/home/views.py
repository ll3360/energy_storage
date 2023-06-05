from flask import session, render_template, redirect, url_for, Response
from controller.modules.home import home_blu

video_camera = None
global_frame = None


# 主页
@home_blu.route('/')
def index():
    # 模板渲染
    # username = session.get("username")
    # if not username:
    #     return redirect(url_for("user.login"))
    return render_template("index.html")
