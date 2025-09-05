import { createModuleLogger } from "./utils/LoggerUtils.js";
const log = createModuleLogger('db');
const DB_NAME = 'CanvasNodeDB';
const STATE_STORE_NAME = 'CanvasState';
const IMAGE_STORE_NAME = 'CanvasImages';
const DB_VERSION = 3;
let db = null;
/**
 * Funkcja pomocnicza do tworzenia żądań IndexedDB z ujednoliconą obsługą błędów
 * @param {IDBObjectStore} store - Store IndexedDB
 * @param {DBRequestOperation} operation - Nazwa operacji (get, put, delete, clear)
 * @param {any} data - Dane dla operacji (opcjonalne)
 * @param {string} errorMessage - Wiadomość błędu
 * @returns {Promise<any>} Promise z wynikiem operacji
 */
function createDBRequest(store, operation, data, errorMessage) {
    return new Promise((resolve, reject) => {
        let request;
        switch (operation) {
            case 'get':
                request = store.get(data);
                break;
            case 'put':
                request = store.put(data);
                break;
            case 'delete':
                request = store.delete(data);
                break;
            case 'clear':
                request = store.clear();
                break;
            default:
                reject(new Error(`Unknown operation: ${operation}`));
                return;
        }
        request.onerror = (event) => {
            log.error(errorMessage, event.target.error);
            reject(errorMessage);
        };
        request.onsuccess = (event) => {
            resolve(event.target.result);
        };
    });
}
function openDB() {
    return new Promise((resolve, reject) => {
        if (db) {
            resolve(db);
            return;
        }
        log.info("Opening IndexedDB...");
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = (event) => {
            log.error("IndexedDB error:", event.target.error);
            reject("Error opening IndexedDB.");
        };
        request.onsuccess = (event) => {
            db = event.target.result;
            log.info("IndexedDB opened successfully.");
            resolve(db);
        };
        request.onupgradeneeded = (event) => {
            log.info("Upgrading IndexedDB...");
            const dbInstance = event.target.result;
            if (!dbInstance.objectStoreNames.contains(STATE_STORE_NAME)) {
                dbInstance.createObjectStore(STATE_STORE_NAME, { keyPath: 'id' });
                log.info("Object store created:", STATE_STORE_NAME);
            }
            if (!dbInstance.objectStoreNames.contains(IMAGE_STORE_NAME)) {
                dbInstance.createObjectStore(IMAGE_STORE_NAME, { keyPath: 'imageId' });
                log.info("Object store created:", IMAGE_STORE_NAME);
            }
        };
    });
}
export async function getCanvasState(id) {
    log.info(`Getting state for id: ${id}`);
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readonly');
    const store = transaction.objectStore(STATE_STORE_NAME);
    const result = await createDBRequest(store, 'get', id, "Error getting canvas state");
    log.debug(`Get success for id: ${id}`, result ? 'found' : 'not found');
    return result ? result.state : null;
}
export async function setCanvasState(id, state) {
    log.info(`Setting state for id: ${id}`);
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STATE_STORE_NAME);
    await createDBRequest(store, 'put', { id, state }, "Error setting canvas state");
    log.debug(`Set success for id: ${id}`);
}
export async function removeCanvasState(id) {
    log.info(`Removing state for id: ${id}`);
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STATE_STORE_NAME);
    await createDBRequest(store, 'delete', id, "Error removing canvas state");
    log.debug(`Remove success for id: ${id}`);
}
export async function saveImage(imageId, imageSrc) {
    log.info(`Saving image with id: ${imageId}`);
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    await createDBRequest(store, 'put', { imageId, imageSrc }, "Error saving image");
    log.debug(`Image saved successfully for id: ${imageId}`);
}
export async function getImage(imageId) {
    log.info(`Getting image with id: ${imageId}`);
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readonly');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    const result = await createDBRequest(store, 'get', imageId, "Error getting image");
    log.debug(`Get image success for id: ${imageId}`, result ? 'found' : 'not found');
    return result ? result.imageSrc : null;
}
export async function removeImage(imageId) {
    log.info(`Removing image with id: ${imageId}`);
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    await createDBRequest(store, 'delete', imageId, "Error removing image");
    log.debug(`Remove image success for id: ${imageId}`);
}
export async function getAllImageIds() {
    log.info("Getting all image IDs...");
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readonly');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    return new Promise((resolve, reject) => {
        const request = store.getAllKeys();
        request.onerror = (event) => {
            log.error("Error getting all image IDs:", event.target.error);
            reject("Error getting all image IDs");
        };
        request.onsuccess = (event) => {
            const imageIds = event.target.result;
            log.debug(`Found ${imageIds.length} image IDs in database`);
            resolve(imageIds);
        };
    });
}
export async function clearAllCanvasStates() {
    log.info("Clearing all canvas states...");
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STATE_STORE_NAME);
    await createDBRequest(store, 'clear', null, "Error clearing canvas states");
    log.info("All canvas states cleared successfully.");
}
