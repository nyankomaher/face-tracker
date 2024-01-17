export type Face = {
  id: number;
  start: number;
  end: number;
  x: string;
  y: string;
  w: string;
}

export type Mask = {
  id: number | null;
  name: string;
  path: string;
  isStill: boolean;
  thumbnail?: string;
}

export type Source = {
  id: number;
  path: string;
  frameRate: number;
}

export type Settings = {
  preset: number;
  venv: string;
  script: string;
  imgsz: number;
  stride: number;
  iou: number;
  conf: number;
  maskRatio: number;
  margin: number;
  minSize: number;
  interpolation: string;
  interpolationForWidth: string;
}

export type Preset = {
  id: number;
  name: string;
  updatable: boolean;
  settings: Settings;
}