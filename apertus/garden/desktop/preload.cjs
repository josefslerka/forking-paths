"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("apertusDesktop", {
  onStatus(callback) {
    if (typeof callback !== "function") return () => {};
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("desktop-status", handler);
    return () => ipcRenderer.removeListener("desktop-status", handler);
  },
  onStatusHistory(callback) {
    if (typeof callback !== "function") return () => {};
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.once("desktop-status-history", handler);
    return () => ipcRenderer.removeListener("desktop-status-history", handler);
  },
  chooseApertusRoot() {
    return ipcRenderer.invoke("desktop:choose-root");
  },
  chooseModelDirectory() {
    return ipcRenderer.invoke("desktop:choose-model-directory");
  },
  retry() {
    return ipcRenderer.invoke("desktop:retry");
  },
  openLog() {
    return ipcRenderer.invoke("desktop:open-log");
  },
});
