// Provides the browser/Dash globals that app_src/assets/clientsideCallbacks.js
// expects, then loads it so tests can call the registered functions through
// window.dash_clientside.sleep_scoring.
//
// PatchStub mirrors the dash_clientside.Patch API the callbacks use: it
// records every assign() so tests can assert on the figure mutations a
// callback decided to make.

class PatchStub {
  constructor() {
    this.ops = [];
  }

  assign(path, value) {
    this.ops.push({ path, value });
  }

  build() {
    return this;
  }
}

globalThis.window = globalThis;
globalThis.dash_clientside = {
  no_update: Symbol("no_update"),
  Patch: PatchStub,
};

require("../../app_src/assets/clientsideCallbacks.js");
