// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

module.exports = function (config) {
  config.set({
    frameworks: ['mocha'],
    files: [
      { pattern: './node_modules/onnxruntime-web/dist/ort.js' },
      { pattern: './node_modules/onnxruntime-web/dist/**/*', included: false, nocache: true },
      { pattern: './browser-test-main.js' },
      { pattern: './model.onnx', included: false }
    ],
    proxies: {
      '/model.onnx': '/base/model.onnx',
    },
    client: { captureConsole: true, mocha: { expose: ['body'], timeout: 60000 } },
    reporters: ['mocha'],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    hostname: 'localhost'
  });
};
