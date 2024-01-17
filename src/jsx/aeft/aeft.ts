import {Face, Mask, Source, Settings} from '../../types'
import {getErrorMessage} from '../../js/helper/error-helper';
import { getActiveComp, isImage, isMovie } from './aeft-utils';

export const selectMaskImage = (): Mask | void => {
  const selection = app.project.selection;
  if (selection.length === 0) {
    alert('マスクに使用する画像が選択されていません。プロジェクトパネルで画像を選択してください。');
    return;
  } else if (selection.length > 1) {
    alert('複数の画像が選択されています。単一の画像を選択してください。');
    return;
  }

  const mask = selection[0];
  if (!isImage(mask)) {
    alert('画像ではないアイテムが選択されています。画像を選択してください。');
    return;
  }

  const mainSource = (mask as FootageItem).mainSource;

  return {
    id: mask.id,
    name: mask.name,
    path: File(mainSource.file!.absoluteURI).fsName, // ~をルートからのパスに変える
    isStill: mainSource.isStill,
  };
}

export const getSource = (): Source | void => {
  const activeViewer = app.activeViewer;
  const activeItem = app.project.activeItem;
  if ((activeViewer && activeViewer.type !== ViewerType.VIEWER_COMPOSITION) ||
      !activeItem ||
      !(activeItem instanceof CompItem)) {
    alert('タイムラインパネルがアクティブではありません。アクティブにしてください。');
    return;
  }

  const comp = activeItem as CompItem;
  if (comp.selectedLayers.length === 0) {
    alert('トラッキング対象の動画が選択されていません。タイムラインパネルで動画を選択してください。');
    return;
  } else if (comp.selectedLayers.length > 1) {
    alert('複数のレイヤーが選択されています。単一のレイヤーを選択してください。');
    return;
  }

  const selected = comp.selectedLayers[0];
  if (!(selected.hasVideo && isImage((selected as AVLayer).source))) {
    alert('選択されているレイヤーは動画ではありません。動画のレイヤーを選択してください。');
    return;
  }

  const source = (selected as AVLayer).source as FootageItem;

  return {
    id: source.id,
    path: File(source.file!.absoluteURI).fsName, // ~をルートからのパスに変える
    frameRate: source.frameRate,
  };
}

export const addFaceMasks = (faces: Face[], maskId: number | null, settings: Settings) => {
  try {
    importExpression();

    if (maskId === null) {
      alert('マスクに使用する画像が選択されていません。');
      return;
    }

    let mask: FootageItem | null = null;
    for (let i = 1; i <= app.project.items.length; i++) {
      const item = app.project.items[i];
      if (item.id === maskId) {
        mask = item as FootageItem;
        break;
      }
    }
    if (!mask || !isImage(mask)) {
      alert('マスクに指定した画像が見つかりませんでした。もう一度選択し直してください。');
      return;
    }

    const comp = getActiveComposition();
    for (let face of faces) {
      addFaceMask(face, mask, comp, settings);
    }

  } catch (e: unknown) {
    alert(getErrorMessage(e));
  }
}

const addFaceMask = (face: Face, mask: FootageItem, comp: CompItem, settings: Settings) => {
  const isStill = face.end === null;

  const layer = comp.layers.add(mask);
  const transform = layer.transform;
  const anchor = transform.anchorPoint as TwoDProperty;

  layer.name = `TRACK_ID: ${face.id}`;
  layer.startTime = Math.max(face.start - settings.margin, 0);
  if (!isStill) {
    layer.outPoint = Math.min(face.end + settings.margin, comp.duration);
  }
  anchor.setValue([mask.width / 2, mask.height / 2]);
  transform.position.dimensionsSeparated = true;
  if (isStill) {
    transform.xPosition.setValue(Number(face.x));
    transform.yPosition.setValue(Number(face.y));
    (transform.scale as TwoDProperty).setValue([Number(face.w), Number(face.w)]);
  } else {
    transform.xPosition.expression = face.x;
    transform.yPosition.expression = face.y;
    transform.scale.expression = face.w;
  }
}

const getItem = <T>(name: string, folder: FolderItem): T => {
  for (let i = 1; i <= folder.items.length; i++) {
    const item = folder.items[i];
    if (item.name === name) {
      return item as T;
    }
  }
  return null as T;
}

const NYANKOMAHER_FOLDER = 'nyankomaher-face-tracker';
const NYANKOMAHER_JSX = 'nyankomaher-face-tracker-functions.jsx';
const importExpression = () => {
  let folder: FolderItem | null = getItem(NYANKOMAHER_FOLDER, app.project.rootFolder)
  if (!folder) {
    folder = app.project.items.addFolder(NYANKOMAHER_FOLDER);
  }

  let functions: FootageItem | null = getItem(NYANKOMAHER_JSX, folder);
  if (!functions) {
    const x = new File($.fileName)
    const path = x.parent.parent.fsName + `/expression/${NYANKOMAHER_JSX}`;
    functions = app.project.importFile(new ImportOptions(new File(path))) as FootageItem;
    functions.parentFolder = folder;
  }
}

const getActiveComposition = () => {
  const comp = getActiveComp();
  if (comp) {
    return comp;
  } else {
    throw new Error('アクティブなコンポジションが取得できません。');
  }
}