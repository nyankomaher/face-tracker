(function() {
  function aaa(a, b) {
    return a + b;
  }

  return {
    test: (a, b) => {
      return aaa(a,b)
    },

    spline: (time, params) => {
      let rawIndex = Math.floor(time * params.frameRate / params.stride) - params.start;
      if (rawIndex < 0) {
        rawIndex = 0;
      } else if (rawIndex > params.coefficients.length - 1) {
        rawIndex = params.coefficients.length - 1;
      }

      // coefficientsはnullのことがあるので、nullでないindexまで遡る
      let c = null;
      let index = null;
      for (let i = rawIndex; i >= 0; i--) {
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