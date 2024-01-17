(function() {
  function aaa(a, b) {
    return a + b;
  }

  return {
    test: (a, b) => {
      return aaa(a,b)
    },

    spline: (time, params) => {
      // このframeはYOLOv8の解析間隔で分割されたフレームのことで、動画のフレームレートのことではない
      let frame = Math.floor(time * params.frameRate / params.stride) - params.start;
      if (frame < 0) {
        frame = 0;
        // y軸は上下動するため（だと思う）startとendの範囲外ではけっこうな勢いで上か下にすっとぶことがある問題の対策
        if (params.axis == 'y') {
          time = params.start / (params.frameRate / params.stride);
        }
      } else if (frame > params.coefficients.length - 1) {
        frame = params.coefficients.length - 1;
        if (params.axis == 'y') {
          time = params.end / (params.frameRate / params.stride);
        }
      }

      // coefficientsはnullのことがあるので、nullでないframeまで遡る
      let c = null;
      let index = null;
      for (let i = frame; i >= 0; i--) {
        c = params.coefficients[i];
        if (c) {
          index = i;
          break;
        }
      }

      // timeに応じた値を計算
      let value = 0;
      const diff = time - (params.start + index) * params.stride / params.frameRate;
      for (let i = 0; i < c.length; i++) {
        value += c[i] * Math.pow(diff, (c.length - 1 - i));
      }

      return value;
    }
  }
})()