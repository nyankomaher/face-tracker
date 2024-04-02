<script setup lang="ts">
import {Face, Mask, Preset, Settings} from '../../types'
import { onMounted, Ref, ref, watch } from "vue";
import { ChildProcessWithoutNullStreams } from "node:child_process";
import { os, path, fs, child_process } from "../lib/cep/node";
import { evalTS } from "../lib/utils/bolt";
import {fromImage, fromVideo} from 'imtool';
const kill = require('tree-kill');  // 普通にimportすると Module "child_process" has been externalized for browser compatibility. Cannot access "child_process.spawn" in client code. と言われる
import {getErrorMessage, FaceTrackerError} from '../helper/error-helper';
import "../index.scss";

const INTERPOLATION_METHODS = [
  {id: 'polynomial:3', name: '3次多項式補間'},
  {id: 'polynomial:5', name: '5次多項式補間'},
  {id: 'spline:1', name: '区分線形補間'},
  {id: 'spline:3', name: '3次スプライン補間'},
  {id: 'akima', name: '秋間補間'},
]

const DEFAULT_SETTINGS: Settings = {preset: 1, venv: '', script: '', imgsz: 960, stride: 2, iou: 0.70, conf: 0.20, maskRatio: 1.00, margin: 6, minSize: 100, interpolation: 'akima', interpolationForWidth: 'polynomial:3', gpu: false};
const DEFAULT_PRESET: Preset = {id: 1, name: 'デフォルト', updatable: false, settings: {...DEFAULT_SETTINGS}};

const useSettings = () => {
  const SETTINGS_KEY = 'nyankomaher-face-tracker-settings';

  let initSettings: Settings = {...DEFAULT_SETTINGS};
  const jsonSettings = localStorage.getItem(SETTINGS_KEY);
  if (jsonSettings) {
    initSettings = {...initSettings, ...JSON.parse(jsonSettings)};
  }
  const settings = ref(initSettings);
  const settingsOpen = ref(true);

  watch([settings], ([settings]) => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, {deep: true});

  return {settings, settingsOpen};
}
const {settings, settingsOpen} = useSettings();



const usePresets = (settings: Ref<Settings>) => {
  const PRESETS_KEY = 'nyankomaher-face-tracker-presets';

  const initPresets = [];
  const jsonPresets = localStorage.getItem(PRESETS_KEY);
  if (jsonPresets) {
    const storedPresets = JSON.parse(jsonPresets);
    for (let preset of storedPresets) {
      initPresets.push({...preset, settings: {...DEFAULT_SETTINGS, ...preset.settings}});
    }
  } else {
    initPresets.push(DEFAULT_PRESET);
  }
  const presets = ref(initPresets);

  const findPresetIndex = (id: number) => {
    return presets.value.findIndex(preset => preset.id === id);
  }

  const findPreset = (id: number) => {
    return presets.value.find(preset => preset.id === id);
  }

  const currentPreset = ref(findPreset(settings.value.preset));
  watch(() => settings.value.preset, (preset) => {
    currentPreset.value = findPreset(preset);
  });

  const applyPreset = () => {
    const id = settings.value.preset;
    let newSettings = findPreset(id).settings;
    if (id === DEFAULT_PRESET.id) {  // デフォルトを適用するときはvenvとscriptは必ず殻なので、現在の値を残す
      newSettings = {
        ...newSettings,
        venv: settings.value.venv,
        script: settings.value.script,
      }
    }
    settings.value = {...newSettings};
  }

  const addPreset = () => {
    const name = prompt('現在の状態を新規プリセットとして登録します。プリセット名を入力してください。');
    if (!name) return;
    const id = presets.value.slice(-1)[0].id + 1;
    const newPreset = {
      id,
      name,
      updatable: true,
      settings: {...settings.value, preset: id},
    }
    presets.value.push(newPreset);
    settings.value.preset = id;
  }

  const updatePreset = () => {
    if (confirm(`現在の値で${currentPreset.value.name}を更新します。よろしいですか？`)) {
      const index = findPresetIndex(settings.value.preset);
      const preset = presets.value[index];
      presets.value[index] = {
        ...preset,
        settings: {...settings.value},
      };
    }
  }

  const deletePreset = () => {
    if (confirm(`${currentPreset.value.name}を削除します。よろしいですか？`)) {
      const index = findPresetIndex(settings.value.preset);
      presets.value.splice(index, 1);
      settings.value.preset = 1;
    }
  }

  watch([presets], ([presets]) => {
    console.log('SAVE',presets)
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
  }, {deep: true});

  return {presets, currentPreset, applyPreset, addPreset, updatePreset, deletePreset};
}
const {presets, currentPreset, applyPreset, addPreset, updatePreset, deletePreset} = usePresets(settings);



const useProgress = () => {
  const progress = ref('');
  const progressEl = ref();
  const appendProgress = (newProgress: string) => {
    progress.value = progress.value + '\n' + newProgress.trim();
    setTimeout(() => progressEl.value.scrollTop = progressEl.value.scrollHeight);
  }
  const clearProgress = () => progress.value = '';
  return {progress, progressEl, appendProgress, clearProgress};
}
const {progress, progressEl, appendProgress, clearProgress} = useProgress();


const mask = ref<Mask>({id: null, name: '', path: '', isStill: true, width: 0});

const selectMaskItem = async () => {
  const selected = await evalTS("selectMaskImage");

  if (selected) {
    const blob = new Blob([fs.readFileSync(selected.path)]);

    if (selected.isStill) {
      const tool = await fromImage(blob);
      selected.thumbnail = await tool.thumbnail(60).toDataURL();
      mask.value = selected;

    } else {
      const url = URL.createObjectURL(blob)
      const video = document.createElement('video')
      video.src = url;
      video.currentTime = 1;
      video.load();
      video.addEventListener('canplay', async (e) => {  // readyState === 4を待たないとサムネが取得できないので、それまで待つ
        const tool = await fromVideo(video);
        selected.thumbnail = await tool.thumbnail(60).toDataURL();
        mask.value = selected;
        URL.revokeObjectURL(url);
      });
    }
  }
}

class AbortError extends Error {};
let tracking = ref(false);
let trackingProcess: ChildProcessWithoutNullStreams | null = null;  // processをrefに入れるとkillするときにIllegal invocationと言われるので、refは使用しない

const startTracking = async () => {
  let tmpdir = null;
  try {
    tracking.value = true;
    clearProgress();
    appendProgress('トラッキング開始');

    // 一時ディレクトリを作成
    tmpdir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'nyankomaher-face-tracker-'));
    const output = path.join(tmpdir, 'faces.json');

    // tracking対象ファイルを取得
    const source = await evalTS("getSource");
    if (!source) throw new Error('トラッキング対象の動画が取得できませんでした。タイムラインパネルでトラッキング対象の映像を正しく選択できているか確認してください。');
    if (source.id === mask.value.id) throw new Error('トラッキング対象の動画とマスク画像が同一です。タイムラインパネルでトラッキング対象の動画を正しく選択できているか確認してください。');

    // trackingスクリプト呼び出し

    await new Promise<void>((resolve, reject) => {
      const python = settings.value.venv || 'python';
      const script = settings.value.script || path.join(__dirname, 'bin', 'track.py');
      const args = [
        script, '--output', output, '--source', source.path, '--rate', source.frameRate, '--stride', settings.value.stride, 
        '--imgsz', settings.value.imgsz, '--nosave', '--iou', settings.value.iou, '--conf', settings.value.conf,
        '--mask-ratio', settings.value.maskRatio, '--mask-size', mask.value.width, '--min-size', settings.value.minSize,
        '--interpolation', settings.value.interpolation, '--margin', settings.value.margin,
        '--interpolation-for-width', settings.value.interpolationForWidth, '--scale', source.scale,
      ].map(o => String(o));
      if (settings.value.gpu) {
        args.push('-d', 'mps');
      }
      appendProgress(`COMMAND: ${python} ${args.join(' ')}`);
      const t = child_process.spawn(python, args);
      trackingProcess = t;
      t.stdout.on('data', chunk => {
        appendProgress(chunk.toString());
      });
      t.stderr.on('data', chunk => {
        console.error(chunk.toString());
        appendProgress(chunk.toString());
      })
      t.on('close', code => {
        appendProgress(`EXIT_CODE: ${code}`);
        if (code === 0) {
          resolve();
        } else if (code === null) {
          reject(new AbortError());
        } else {
          reject();
        }
      });
      t.on('error', e => {
        reject(e);
      });
    });

    // 結果読み込み
    const faces: Face[] = JSON.parse(fs.readFileSync(output, 'utf8'));
    // 画像を追加
    await evalTS("addFaceMasks", faces, mask.value.id, settings.value);

    appendProgress('トラッキングが完了しました。');

  } catch (e: unknown) {
    console.error(e);
    if (!(e instanceof AbortError)) {
      const message = getErrorMessage(e);
      if (e instanceof FaceTrackerError && e.needAlert) {
        alert(message);
      }
      appendProgress(message);
      appendProgress('トラッキングがエラーで中断されました。');
    }

  } finally {
    // 一時ディレクトリを削除
    if (tmpdir) fs.unlink(tmpdir, () => {});
    killProcess();
    tracking.value = false;
  }
}

const killProcess = () => {
  if (trackingProcess) {
    kill(trackingProcess.pid!);
    trackingProcess = null;
  }
}

const abortTracking = () => {
  killProcess();
  setTimeout(() => {
    tracking.value = false;
    appendProgress('トラッキングを中断しました。');
  }, 500);
}


const copyProgress = () => {
  // clipboard APIでコピーしようとするとWrite permission denied.などと言い出すので、execCommandを使用する
  const range = document.createRange();
  range.selectNodeContents(progressEl.value);
  const sel = window.getSelection();
  if (sel) {
    sel.removeAllRanges();
    sel.addRange(range);
    document.execCommand('copy');
  }
}
</script>

<template>
  <div class="app">
    <main class="app__main">
      <section class="app__tracking app__section">
        <div class="app__tracking__exec">
          <div class="app__tracking__exec__mask">
            <h4 class="app__tracking__exec__mask__title">マスク画像</h4>
            <div class="app__tracking__exec__mask__desc">
              <button @click="selectMaskItem" 
                :disabled="tracking"
                class="app__tracking__exec__mask__button">選択</button>
              <figure class="app__tracking__exec__mask__image">
                <img v-if="mask.thumbnail" :src="mask.thumbnail" alt="マスク画像のサムネイル" />
              </figure>
              <input :value="mask.name" readonly class="app__tracking__exec__mask__name" />
            </div>
          </div>
          <div class="app__tracking__exec__controls">
            <button v-if="!tracking"
              @click="startTracking"
              :disabled="tracking || !mask.id"
              class="app__tracking__exec__control">トラッキング開始</button>
            <button v-else
              @click="abortTracking"
              class="app__tracking__exec__control">トラッキング中断</button>
          </div>
        </div>

        <a @click="settingsOpen = !settingsOpen" class="app__tracking__toggle">▼設定</a>
        <dl v-if="settingsOpen" class="app__tracking__settings">
          <div class="app__tracking__preset app__tracking__setting">
            <dt class="app__tracking__preset__title app__tracking__setting__title">プリセット</dt>
            <dd class="app__tracking__preset__desc app__tracking__setting__desc">
              <select v-model="settings.preset"
                :disabled="tracking"
                class="app__tracking__preset__select">
                <option v-for="preset in presets"
                  :key="preset.id"
                  :value="preset.id">{{preset.name}}</option>
              </select>
              <div class="app__tracking__preset__buttons">
                <button @click="applyPreset"
                  :disabled="tracking"
                  class="app__tracking__preset__button">適用</button>
                <button @click="addPreset"
                  :disabled="tracking"
                  class="app__tracking__preset__button">新規</button>
                <button @click="updatePreset"
                  :disabled="!currentPreset.updatable || tracking"
                  class="app__tracking__preset__button">更新</button>
                <button @click="deletePreset"
                  :disabled="!currentPreset.updatable || tracking"
                  class="app__tracking__preset__button">削除</button>
              </div>
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">解析サイズ</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="1" max="9999" :disabled="tracking" v-model="settings.imgsz" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">解析間隔</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="1" max="99" :disabled="tracking" v-model="settings.stride" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">重複領域閾値</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="0" max="1" step="0.01" :disabled="tracking" v-model="settings.iou" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">確信度閾値</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="0" max="1" step="0.01" :disabled="tracking" v-model="settings.conf" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">マスク画像拡張</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="0" max="10" step="0.01" :disabled="tracking" v-model="settings.maskRatio" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">マスク時間拡張</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="0" max="99" step="1" :disabled="tracking" v-model="settings.margin" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">最小マスクサイズ</dt>
            <dd class="app__tracking__setting__desc">
              <input type="number" min="0" max="999" :disabled="tracking" v-model="settings.minSize" />
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">マスク位置補間</dt>
            <dd class="app__tracking__setting__desc">
              <select v-model="settings.interpolation"
                :disabled="tracking">
                <option v-for="method in INTERPOLATION_METHODS"
                  :key="method.id"
                  :disabled="tracking"
                  :value="method.id">{{method.name}}</option>
              </select>
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">マスクサイズ補間</dt>
            <dd class="app__tracking__setting__desc">
              <select v-model="settings.interpolationForWidth"
                :disabled="tracking">
                <option v-for="method in INTERPOLATION_METHODS"
                  :key="method.id"
                  :value="method.id">{{method.name}}</option>
              </select>
            </dd>
          </div>
          <div class="app__tracking__setting -short">
            <dt class="app__tracking__setting__title">GPU</dt>
            <dd class="app__tracking__setting__desc">
              <input type="checkbox" value="1" :disabled="tracking" v-model="settings.gpu" />
            </dd>
          </div>
          <div class="app__tracking__venv app__tracking__setting">
            <dt class="app__tracking__setting__title">Python</dt>
            <dd class="app__tracking__setting__desc">
              <input type="search" :disabled="tracking" v-model="settings.venv" />
            </dd>
          </div>
          <div class="app__tracking__venv app__tracking__setting">
            <dt class="app__tracking__setting__title">スクリプト</dt>
            <dd class="app__tracking__setting__desc">
              <input type="search" :disabled="tracking" v-model="settings.script" />
            </dd>
          </div>
        </dl>
      </section>
      <section class="app__log app__section">
        <header class="app__log__header">
          <h2 class="app__log__header__title">ログ</h2>
          <button @click="copyProgress"
            class="app__log__header__copy">コピー</button>
        </header>
        <p ref="progressEl" class="app__log__progress">{{progress}}</p>
      </section>
    </main>
  </div>
</template>
