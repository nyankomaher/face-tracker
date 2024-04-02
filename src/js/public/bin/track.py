import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # torchがインポートされるより先に設定する必要がある

import sys
import json
import re
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
parser.add_argument('-a', '--mask-size', type=int, default=100)
parser.add_argument('-e', '--min-size', type=int, default=50)
parser.add_argument('-x', '--margin', type=int, default=6)
parser.add_argument('-d', '--device', default='mps')
parser.add_argument('-n', '--interpolation', default='spline:3')
parser.add_argument('-w', '--interpolation-for-width', default='polynomial:3')
parser.add_argument('--scale', type=float, default=1)

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
mask_size = args.mask_size
min_size = args.min_size
margin = args.margin
device = args.device
interpolation = args.interpolation
interpolation_for_width = args.interpolation_for_width
scale = args.scale
trackingRate = frameRate / stride # frame数を秒数に直すための係数
stride_second = stride / frameRate # strideを秒数に直したもの
margin_frame_count = int(margin / stride) # marginを解析フレーム数に直したもの
model_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
tracker = os.path.join(os.path.dirname(__file__), 'boatsort.yaml')
is_still = source.lower().endswith(('.png', '.jpg', '.jpeg'))

FrameBox = namedtuple('FrameBox', ('frame', 'xywh', 'xyxyn'))
Face = namedtuple('Face', ('id', 'start', 'end', 'x', 'y', 'w'))

interpolation_name_pattern = re.compile(r'^(polynomial|spline|akima)(:(\d+))?$')
mask_offset_ratio = 0.3

# YOLOv8モデルをロードしてトラッキング
model = YOLO(model_path)
if is_still:
    results = model.predict(source=source, classes=0, conf=conf, iou=iou, vid_stride=stride, imgsz=imgsz, verbose=True, save=save, project=project)
else:
    results = model.track(source=source, classes=0, tracker=tracker, conf=conf, iou=iou, vid_stride=stride, imgsz=imgsz, persist=True, stream=True, verbose=True, save=save, project=project, device=device)


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
            if is_still:
                track_id = f"N{no_tracking_id}"
                no_tracking_id = no_tracking_id + 1
            else:
                continue  # 単発のやつはほぼノイズなので除外する
        frameBoxes = trackFrameBoxes[track_id] if track_id in trackFrameBoxes else []
        frameBoxes.append(FrameBox(index, box.xywh, box.xyxyn))
        trackFrameBoxes[track_id] = frameBoxes

def make_formulas(track_id, frameBoxes):
    seconds = np.array([frame / trackingRate for frame, _, _ in frameBoxes])
    xyxyns = np.array([xyxyn[0] for _, _, xyxyn in frameBoxes])
    xs, ys, ws, hs =  np.array([xywh[0] for _, xywh, _ in frameBoxes]).T # xywhではxyはバウンディングボックスの中心点
    xs, ys, ws = adjust_cut_off(xs, ys, ws, hs, xyxyns)
    ws = (ws * mask_ratio).clip(min_size, None)
    ys = ys + ((ws - hs) / 2) - (ws * mask_offset_ratio)  # マスクがバウンディングボックスの上辺からちょっと出るように調整
    wps = ws / mask_size * 100 # pxから%になおす

    x = make_formula(track_id, seconds, xs, interpolation, 'x')
    y = make_formula(track_id, seconds, ys, interpolation, 'y')
    w = make_formula(track_id, seconds, wps, interpolation_for_width, 'w')

    return [x, y, w]

def make_formula(track_id, seconds, values, intp, axis, show_plot=False):
    values = values * scale
    match = interpolation_name_pattern.search(intp)
    if not match:
        raise Exception(f'サポートしていない補間方法（{intp}）です。')

    if len(seconds) >= 2:
        # マスク時間拡張(margin)の分ダミーデータを追加する
        if axis == 'y' or axis == 'w':  # y位置と幅は変化させないほうが有利っぽい
            values = np.array([
                *np.full(margin_frame_count, values[0]),
                *values,
                *np.full(margin_frame_count, values[-1]),
            ])
        else:
            values = np.array([
                *reversed(values[0] + np.arange(1, margin_frame_count + 1) * (values[0] - values[1])),
                *values,
                *(values[-1] + np.arange(1, margin_frame_count + 1) * (values[-1] - values[-2]))
            ])
        seconds = np.array([
            *reversed(seconds[0] + np.arange(1, margin_frame_count + 1) * stride_second * -1),
            *seconds,
            *(seconds[-1] + np.arange(1, margin_frame_count + 1) * stride_second)
        ])

    interpolation_name = match.group(1)
    degree = int(match.group(3) or 0)

    if interpolation_name == 'polynomial':
        formula = make_polynomial_fomula(track_id, seconds, values, degree, axis)
    elif interpolation_name in ['spline', 'akima']:
        formula = make_spline_formula(track_id, seconds, values, frameBoxes, interpolation_name, degree or 3, axis, show_plot=show_plot)

    if axis == 'w':
        formula = f'const w=Math.max({formula},{min_size/mask_size*100});[w,w]'

    if interpolation_name in ['spline', 'akima']:
        formula = prepend_import(formula)

    return formula

# パラメータのExpressionを作成
def make_polynomial_fomula(track_id, seconds, values, degree, axis, show_plot=False):
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
        test_times = np.arange(seconds[0], seconds[-1], 0.1)
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
    start_frame = frameBoxes[0].frame - margin_frame_count
    end_frame = frameBoxes[-1].frame + margin_frame_count
    available_frames = [
        *np.arange(margin_frame_count) + start_frame,
        *[frameBox.frame for frameBox in frameBoxes],
        *np.arange(margin_frame_count) + frameBoxes[-1].frame,
    ]
    coefficients = np.full(end_frame - start_frame, None)
    for i, available_frame in enumerate(available_frames[:-1]):  # 点と点の間に関して補完をするので、最後の1個に対するcoefficientはない。そのため-1
        coefficient = c[:, i].round().astype(np.int32).tolist()
        coefficients[available_frame - start_frame] = coefficient

    params = {}
    params['axis'] = axis
    params['start'] = start_frame
    params['end'] = end_frame
    params['frameRate'] = frameRate
    params['stride'] = stride
    params['coefficients'] = coefficients.tolist()

    return f'f.spline(time,{json.dumps(params)})'

# 人物が画面端（X軸方向）にいる場合見切れるため、認識した人物の幅が実際よりも小さくなってしまう。
# このため、人物が画面端で見切れていないときの縦横幅を基準とし、
# 人物が画面端にいるときの位置、幅を、そのときの高さから計算し直す。
# また、人物が見切れていくときに高さの基準が頭から肩に移行するが、
# その際に急激にY位置が変化しマスクが外れることがある。
# Y位置の調整で対応したいが、良い方法が思い浮かばないので見切れ中はマスクサイズを大きくして対応する。
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
        return [xs, ys, ws]
    
    aspect = np.mean(aspects)
    for i, xyxyn in enumerate(xyxyns):
        if xyxyn[0] == 0 or xyxyn[2] == 1:
            # 推定幅を求める。最大幅より小さくはならない。また、見切れ中はマスクサイズを大きくする。
            w = max(hs[i] * aspect, maxw) * 1.2
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
        w = xywh[0, 2].item() * scale
        h = xywh[0, 3].item() * scale
        x = xywh[0, 0].item() * scale
        y = xywh[0, 1].item() * scale + ((w - h) / 2) - (w * mask_offset_ratio) # マスクがバウンディングボックスの上辺からちょっと出るように調整
        w = max(w, min_size) / mask_size * 100  # pxから%に変換

    else:
        start = (frameBoxes[0].frame - margin_frame_count) / trackingRate
        end = (frameBoxes[-1].frame + margin_frame_count) / trackingRate
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

