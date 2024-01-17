import os
import sys
import json
import re
import torch
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from collections import namedtuple
from ultralytics import YOLO

default_project = os.path.join(os.path.dirname(__file__), f'../../../../local')
default_project_with_date = os.path.join(default_project, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default=f'{default_project_with_date}/faces.json')
parser.add_argument('-s', '--source', default=f'{default_project}/IMG_6782.mp4')
parser.add_argument('-r', '--rate', type=float, default=30)
parser.add_argument('-t', '--stride', type=int, default=2)
parser.add_argument('-z', '--imgsz', type=int, default=960)
parser.add_argument('--nosave', action='store_false')
parser.add_argument('-p', '--project', default=default_project_with_date)
parser.add_argument('-i', '--iou', type=float, default=0.7)
parser.add_argument('-c', '--conf', type=float, default=0.2)
parser.add_argument('-m', '--mask-ratio', type=float, default=1.0)
parser.add_argument('-e', '--min-size', type=int)
parser.add_argument('-x', '--margin', type=float, default=0)
parser.add_argument('-n', '--interpolation', default='spline:3')
parser.add_argument('-w', '--interpolation-for-width', default='polynomial:3')

args = parser.parse_args()
output = args.output
source = args.source
frameRate = max(args.rate, 1)
stride = args.stride
imgsz = args.imgsz
save = args.nosave  # store_falseなのでそのまま代入
project = args.project
iou = args.iou
conf = args.conf
mask_ratio = args.mask_ratio
min_size = args.min_size
margin = args.margin
interpolation = args.interpolation
interpolation_for_width = args.interpolation_for_width
trackingRate = frameRate / stride # frame数を秒数に直すための係数
model_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
tracker = os.path.join(os.path.dirname(__file__), 'boatsort.yaml')
is_still = source.lower().endswith(('.png', '.jpg', '.jpeg'))

FrameBox = namedtuple('FrameBox', ('frame', 'xywh', 'xyxyn'))
Face = namedtuple('Face', ('id', 'start', 'end', 'x', 'y', 'w'))

interpolation_name_pattern = re.compile(r'^(polynomial|spline|akima)(:(\d+))?$')

# YOLOv8モデルをロードしてトラッキング
model = YOLO(model_path)
if is_still:
    results = model.predict(source=source, classes=0, conf=conf, iou=iou, vid_stride=stride, imgsz=imgsz, verbose=True, save=save, project=project)
else:
    results = model.track(source=source, classes=0, tracker=tracker, conf=conf, iou=iou, vid_stride=stride, imgsz=imgsz, persist=True, stream=True, verbose=True, save=save, project=project)


# トラッキング結果をtrack_idごとにまとめる
no_tracking_id = 1
trackFrameBoxes = {}
orig_shape = None
for index, result in enumerate(results):
    orig_shape = result.orig_shape
    for box in result.boxes:
        if box.id:
            track_id = int(box.id[0])
        else: #単発のやつはtrack_idが振られない
            continue  # 単発のやつはほぼノイズなので除外する
            # track_id = f"N{no_tracking_id}"
            # no_tracking_id = no_tracking_id + 1
        frameBoxes = trackFrameBoxes[track_id] if track_id in trackFrameBoxes else []
        frameBoxes.append(FrameBox(index, box.xywh, box.xyxyn))
        trackFrameBoxes[track_id] = frameBoxes

def make_formulas(track_id, frameBoxes):
    seconds = np.array([frame / trackingRate for frame, _, _ in frameBoxes])
    xyxyns = np.array([xyxyn[0] for _, _, xyxyn in frameBoxes])
    xs, ys, ws, hs =  np.array([xywh[0] for _, xywh, _ in frameBoxes]).T # xywhではxyはバウンディングボックスの中心点
    ys = ys + ((ws - hs) / 2) # マスクがバウンディングボックスの上辺に接するよう調整
    xs, ys, ws = adjust_cut_off(xs, ys, ws, hs, xyxyns)
    ws = (ws * mask_ratio).clip(min_size or 0, None)

    x = make_formula(track_id, seconds, xs, interpolation, 'x')
    y = make_formula(track_id, seconds, ys, interpolation, 'y', show_plot=False)
    w = make_formula(track_id, seconds, ws, interpolation_for_width, 'w')

    return [x, y, w]

def make_formula(track_id, seconds, values, intp, axis, show_plot=False):
    match = interpolation_name_pattern.search(intp)
    if not match:
        raise Exception(f'サポートしていない補間方法（{intp}）です。')
    interpolation_name = match.group(1)
    degree = int(match.group(3) or 0)

    if interpolation_name == 'polynomial':
        formula = make_polynomial_fomula(track_id, seconds, values, degree, axis)
    elif interpolation_name in ['spline', 'akima']:
        formula = make_spline_formula(track_id, seconds, values, frameBoxes, interpolation_name, degree or 3, axis, show_plot=show_plot)

    if axis == 'w':
        formula = f'const w={formula};[w,w]'

    if interpolation_name in ['spline', 'akima']:
        formula = prepend_import(formula)

    return formula

# パラメータのExpressionを作成
def make_polynomial_fomula(track_id, seconds, values, degree, axis, show_plot=False):
    if len(seconds) >= 2:
        # 頭とおしりで数値の変化がバタつくので、ダミーデータを加えてなめらかにする
        values = np.array([values[0], *values, values[-1]])
        seconds = np.array([seconds[0] - margin, *seconds, seconds[-1] + margin])
    seconds =seconds[:, np.newaxis]
    values = values[:, np.newaxis]
    polynominal_features = PolynomialFeatures(degree=degree)
    seconds_poly = polynominal_features.fit_transform(seconds)
    regression = LinearRegression()
    regression.fit(seconds_poly, values)
    if show_plot:
        plot.scatter(seconds, values)
        plot.plot(seconds, regression.predict(seconds_poly), color='r')
        plot.title(f'{axis}{track_id}')
        plot.show()

    # 式を作成
    intercept = regression.intercept_[0]
    coef = regression.coef_[0, 1:]
    fomula = f'{intercept}'
    for i in range(degree):
        if i == 0:
            fomula = f'{fomula} + ({coef[i]} * time)'
        else:
            fomula = f'{fomula} + ({coef[i]} * Math.pow(time, {i + 1}))'
    return fomula

def make_spline_formula(track_id, seconds, values, frameBoxes, interpolation_name, degree, axis, show_plot=False):
    if len(seconds) == 1:
        return int(round(values[0]))

    if interpolation_name == 'akima':
        c = Akima1DInterpolator(seconds, values).c
    elif interpolation_name == 'spline':
        if degree == 3:
            c = CubicSpline(seconds, values).c
        elif degree == 1:  # 線形補間
            pairs = zip(seconds[:-1], values[:-1], seconds[1:], values[1:])
            c = np.array([[(v2 - v1) / (s2 - s1), v1] for s1, v1, s2, v2 in pairs]).T

    if not 'c' in locals():
        raise Exception(f'サポートしていない補間方法（{interpolation_name}:{degree}）です。')

    if show_plot:
        test_times = np.arange(start, end, 0.1)
        predicts = []
        for time in test_times:
            for i, x0 in reversed(list(enumerate(seconds[:-1]))): # akima/splineでは最後の1個はパラメータを作らないため[:-1]
                if x0 <= time:
                    range_index = i
                    range_x0 = x0 
                    break
            xdiff = time - range_x0
            predict = sum([c.item(i, range_index) * (xdiff ** (degree - i)) for i in range(degree + 1)])
            predicts.append(predict)
        plot.scatter(seconds, values)
        plot.plot(test_times, predicts, color='r')
        plot.title(f'{axis}{track_id}')
        plot.show()

    # 係数をフレームごとに分ける
    start_frame = frameBoxes[0].frame
    end_frame = frameBoxes[-1].frame
    coefficients = np.full(end_frame - start_frame, None)
    for i, frameBox in enumerate(frameBoxes[:-1]):
        coefficient = c[:, i].round().astype(np.int32).tolist()
        coefficients[frameBox.frame - start_frame] = coefficient

    params = {}
    params['axis'] = axis
    params['start'] = frameBoxes[0].frame
    params['end'] = frameBoxes[-1].frame
    params['frameRate'] = frameRate
    params['stride'] = stride
    params['coefficients'] = coefficients.tolist()

    return f'f.spline(time,{json.dumps(params)})'

# 人物が画面端（X軸方向）にいる場合見切れるため、認識した人物の幅が実際よりも小さくなってしまう。
# このため、人物が画面端で見切れていないときの縦横幅を基準とし、
# 人物が画面端にいるときの位置、幅を、そのときの高さから計算し直す。
def adjust_cut_off(xs, ys, ws, hs, xyxyns):
    # 人物がX軸方向に見切れていないときのx/y比の平均と、最大幅を求める
    aspects = []
    maxw = 0
    for i, xyxyn in enumerate(xyxyns):
        if xyxyn[0] != 0 or xyxyn[2] != 1:
            aspects.append(ws[i] / hs[i])
            maxw = max(maxw, ws[i])
    
    # 見切れていないタイミングがない場合はそのまま
    if len(aspects) == 0:
        return [xs, ws]
    
    aspect = np.mean(aspects)
    for i, xyxyn in enumerate(xyxyns):
        if xyxyn[0] == 0 or xyxyn[2] == 1:
            # 推定幅を求める。最大幅より小さくはならない。
            w = max(hs[i] * aspect, maxw)
            # 左上の座標を求める（xsは中心座標のため） & 推定幅を反映した中心座標を求める
            # y = ys[i] + ((w - ws[i]) / 2)
            # if xyxyn[0] == 0: #左側
            #     x = xs[i] + ((ws[i] - w) / 2)
            # else: #右側
            #     x = xs[i] + ((w - ws[i]) / 2)
            # xs[i] = x
            # ys[i] = y
            ws[i] = w
            
    return [xs, ys, ws]

def prepend_import(code):
    return f'const f=footage("nyankomaher-face-tracker-functions.jsx").sourceData;{code}'
    

# トラッキング結果をAfter Effectsに渡せる形にする
faces = []
for [track_id, frameBoxes] in trackFrameBoxes.items():
    if is_still:
        start = 0
        end = None
        xywh = frameBoxes[0].xywh # xywhではxyはバウンディングボックスの中心点
        w = xywh[0, 2].item()
        h = xywh[0, 3].item()
        x = xywh[0, 0].item()
        y = xywh[0, 1].item() + ((w - h) / 2) # マスクがバウンディングボックスの上辺に接するよう調整

    else:
        start = frameBoxes[0].frame / trackingRate
        end = frameBoxes[-1].frame / trackingRate
        x, y, w = make_formulas(track_id, frameBoxes)
            
    face = Face(track_id, start, end, x, y, w)
    faces.append(face)

# 結果出力
faces = [face._asdict() for face in faces]
f = open(output, 'w')
f.write(json.dumps(faces))
f.flush()
f.close()

sys.exit()

