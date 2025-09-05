import {createModuleLogger} from "./LoggerUtils.js";
import {withErrorHandling, createValidationError} from "../ErrorHandler.js";
import { createCanvas } from "./CommonUtils.js";
import type { Tensor, ImageDataPixel } from '../types';

const log = createModuleLogger('ImageUtils');

export function validateImageData(data: any): boolean {
    log.debug("Validating data structure:", {
        hasData: !!data,
        type: typeof data,
        isArray: Array.isArray(data),
        keys: data ? Object.keys(data) : null,
        shape: data?.shape,
        dataType: data?.data ? data.data.constructor.name : null,
        fullData: data
    });

    if (!data) {
        log.info("Data is null or undefined");
        return false;
    }

    if (Array.isArray(data)) {
        log.debug("Data is array, getting first element");
        data = data[0];
    }

    if (!data || typeof data !== 'object') {
        log.info("Invalid data type");
        return false;
    }

    if (!data.data) {
        log.info("Missing data property");
        return false;
    }

    if (!(data.data instanceof Float32Array)) {
        try {
            data.data = new Float32Array(data.data);
        } catch (e) {
            log.error("Failed to convert data to Float32Array:", e);
            return false;
        }
    }

    return true;
}

export function convertImageData(data: any): ImageDataPixel {
    log.info("Converting image data:", data);

    if (Array.isArray(data)) {
        data = data[0];
    }

    const shape = data.shape;
    const height = shape[1];
    const width = shape[2];
    const channels = shape[3];
    const floatData = new Float32Array(data.data);

    log.debug("Processing dimensions:", {height, width, channels});

    const rgbaData = new Uint8ClampedArray(width * height * 4);

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const pixelIndex = (h * width + w) * 4;
            const tensorIndex = (h * width + w) * channels;

            for (let c = 0; c < channels; c++) {
                const value = floatData[tensorIndex + c];
                rgbaData[pixelIndex + c] = Math.max(0, Math.min(255, Math.round(value * 255)));
            }

            rgbaData[pixelIndex + 3] = 255;
        }
    }

    return {
        data: rgbaData,
        width: width,
        height: height
    };
}

export function applyMaskToImageData(imageData: ImageDataPixel, maskData: Tensor): ImageDataPixel {
    log.info("Applying mask to image data");

    const rgbaData = new Uint8ClampedArray(imageData.data);
    const width = imageData.width;
    const height = imageData.height;

    const maskShape = maskData.shape;
    const maskFloatData = new Float32Array(maskData.data);

    log.debug(`Applying mask of shape: ${maskShape}`);

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const pixelIndex = (h * width + w) * 4;
            const maskIndex = h * width + w;

            const alpha = maskFloatData[maskIndex];
            rgbaData[pixelIndex + 3] = Math.max(0, Math.min(255, Math.round(alpha * 255)));
        }
    }

    log.info("Mask application completed");

    return {
        data: rgbaData,
        width: width,
        height: height
    };
}

export const prepareImageForCanvas = withErrorHandling(function (inputImage: any): ImageDataPixel {
    log.info("Preparing image for canvas:", inputImage);

    if (Array.isArray(inputImage)) {
        inputImage = inputImage[0];
    }

    if (!inputImage || !inputImage.shape || !inputImage.data) {
        throw createValidationError("Invalid input image format", {inputImage});
    }

    const shape = inputImage.shape;
    const height = shape[1];
    const width = shape[2];
    const channels = shape[3];
    const floatData = new Float32Array(inputImage.data);

    log.debug("Image dimensions:", {height, width, channels});

    const rgbaData = new Uint8ClampedArray(width * height * 4);

    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const pixelIndex = (h * width + w) * 4;
            const tensorIndex = (h * width + w) * channels;

            for (let c = 0; c < channels; c++) {
                const value = floatData[tensorIndex + c];
                rgbaData[pixelIndex + c] = Math.max(0, Math.min(255, Math.round(value * 255)));
            }

            rgbaData[pixelIndex + 3] = 255;
        }
    }

    return {
        data: rgbaData,
        width: width,
        height: height
    };
}, 'prepareImageForCanvas');

export const imageToTensor = withErrorHandling(async function (image: HTMLImageElement | HTMLCanvasElement): Promise<Tensor> {
    if (!image) {
        throw createValidationError("Image is required");
    }

    const { canvas, ctx } = createCanvas(image.width, image.height, '2d', { willReadFrequently: true });

    if (ctx) {
        ctx.drawImage(image, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = new Float32Array(canvas.width * canvas.height * 3);

        for (let i = 0; i < imageData.data.length; i += 4) {
            const pixelIndex = i / 4;
            data[pixelIndex * 3] = imageData.data[i] / 255;
            data[pixelIndex * 3 + 1] = imageData.data[i + 1] / 255;
            data[pixelIndex * 3 + 2] = imageData.data[i + 2] / 255;
        }

        return {
            data: data,
            shape: [1, canvas.height, canvas.width, 3],
            width: canvas.width,
            height: canvas.height
        };
    }
    throw new Error("Canvas context not available");
}, 'imageToTensor');

export const tensorToImage = withErrorHandling(async function (tensor: Tensor): Promise<HTMLImageElement> {
    if (!tensor || !tensor.data || !tensor.shape) {
        throw createValidationError("Invalid tensor format", {tensor});
    }

    const [, height, width, channels] = tensor.shape;
    const { canvas, ctx } = createCanvas(width, height, '2d', { willReadFrequently: true });

    if (ctx) {
        const imageData = ctx.createImageData(width, height);
        const data = tensor.data;

        for (let i = 0; i < width * height; i++) {
            const pixelIndex = i * 4;
            const tensorIndex = i * channels;

            imageData.data[pixelIndex] = Math.round(data[tensorIndex] * 255);
            imageData.data[pixelIndex + 1] = Math.round(data[tensorIndex + 1] * 255);
            imageData.data[pixelIndex + 2] = Math.round(data[tensorIndex + 2] * 255);
            imageData.data[pixelIndex + 3] = 255;
        }

        ctx.putImageData(imageData, 0, 0);

        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = canvas.toDataURL();
        });
    }
    throw new Error("Canvas context not available");
}, 'tensorToImage');

export const resizeImage = withErrorHandling(async function (image: HTMLImageElement, maxWidth: number, maxHeight: number): Promise<HTMLImageElement> {
    if (!image) {
        throw createValidationError("Image is required");
    }

    const originalWidth = image.width;
    const originalHeight = image.height;
    const scale = Math.min(maxWidth / originalWidth, maxHeight / originalHeight);
    const newWidth = Math.round(originalWidth * scale);
    const newHeight = Math.round(originalHeight * scale);

    const { canvas, ctx } = createCanvas(newWidth, newHeight, '2d', { willReadFrequently: true });
    
    if (ctx) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(image, 0, 0, newWidth, newHeight);

        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = canvas.toDataURL();
        });
    }
    throw new Error("Canvas context not available");
}, 'resizeImage');

export const createThumbnail = withErrorHandling(async function (image: HTMLImageElement, size = 128): Promise<HTMLImageElement> {
    return resizeImage(image, size, size);
}, 'createThumbnail');

export const imageToBase64 = withErrorHandling(function (image: HTMLImageElement | HTMLCanvasElement, format = 'png', quality = 0.9): string {
    if (!image) {
        throw createValidationError("Image is required");
    }

    const width = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
    const height = image instanceof HTMLImageElement ? image.naturalHeight : image.height;
    const { canvas, ctx } = createCanvas(width, height, '2d', { willReadFrequently: true });

    if (ctx) {
        ctx.drawImage(image, 0, 0);
        const mimeType = `image/${format}`;
        return canvas.toDataURL(mimeType, quality);
    }
    throw new Error("Canvas context not available");
}, 'imageToBase64');

export const base64ToImage = withErrorHandling(function (base64: string): Promise<HTMLImageElement> {
    if (!base64) {
        throw createValidationError("Base64 string is required");
    }

    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("Failed to load image from base64"));
        img.src = base64;
    });
}, 'base64ToImage');

export function isValidImage(image: any): image is HTMLImageElement | HTMLCanvasElement {
    return image &&
        (image instanceof HTMLImageElement || image instanceof HTMLCanvasElement) &&
        image.width > 0 &&
        image.height > 0;
}

export function getImageInfo(image: HTMLImageElement | HTMLCanvasElement): {width: number, height: number, aspectRatio: number, area: number} | null {
    if (!isValidImage(image)) {
        return null;
    }

    const width = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
    const height = image instanceof HTMLImageElement ? image.naturalHeight : image.height;

    return {
        width,
        height,
        aspectRatio: width / height,
        area: width * height
    };
}

export function createImageFromSource(source: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = source;
    });
}

export const createEmptyImage = withErrorHandling(function (width: number, height: number, color = 'transparent'): Promise<HTMLImageElement> {
    const { canvas, ctx } = createCanvas(width, height, '2d', { willReadFrequently: true });

    if (ctx) {
        if (color !== 'transparent') {
            ctx.fillStyle = color;
            ctx.fillRect(0, 0, width, height);
        }

        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = canvas.toDataURL();
        });
    }
    throw new Error("Canvas context not available");
}, 'createEmptyImage');

/**
 * Converts a canvas or image to an Image element
 * Consolidated from MaskProcessingUtils.convertToImage()
 * @param source - Source canvas or image
 * @returns Promise with Image element
 */
export async function convertToImage(source: HTMLCanvasElement | HTMLImageElement): Promise<HTMLImageElement> {
    if (source instanceof HTMLImageElement) {
        return source; // Already an image
    }

    const image = new Image();
    image.src = source.toDataURL();
    
    await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = reject;
    });

    return image;
}

/**
 * Creates a mask from image source for use in mask editor
 * Consolidated from mask_utils.create_mask_from_image_src()
 * @param imageSrc - Image source (URL or data URL)
 * @returns Promise returning Image object
 */
export function createMaskFromImageSrc(imageSrc: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = imageSrc;
    });
}

/**
 * Converts canvas to Image for use as mask
 * Consolidated from mask_utils.canvas_to_mask_image()
 * @param canvas - Canvas to convert
 * @returns Promise returning Image object
 */
export function canvasToMaskImage(canvas: HTMLCanvasElement): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = canvas.toDataURL();
    });
}

/**
 * Scales an image to fit within specified bounds while maintaining aspect ratio
 * @param image - Image to scale
 * @param targetWidth - Target width to fit within
 * @param targetHeight - Target height to fit within
 * @returns Promise with scaled Image element
 */
export async function scaleImageToFit(image: HTMLImageElement, targetWidth: number, targetHeight: number): Promise<HTMLImageElement> {
    const scale = Math.min(targetWidth / image.width, targetHeight / image.height);
    const scaledWidth = Math.max(1, Math.round(image.width * scale));
    const scaledHeight = Math.max(1, Math.round(image.height * scale));
    
    const { canvas, ctx } = createCanvas(scaledWidth, scaledHeight, '2d', { willReadFrequently: true });
    if (!ctx) throw new Error("Could not create scaled image context");
    
    ctx.drawImage(image, 0, 0, scaledWidth, scaledHeight);
    
    return new Promise((resolve, reject) => {
        const scaledImg = new Image();
        scaledImg.onload = () => resolve(scaledImg);
        scaledImg.onerror = reject;
        scaledImg.src = canvas.toDataURL();
    });
}

/**
 * Unified tensor to image data conversion
 * Handles both RGB images and grayscale masks
 * @param tensor - Input tensor data
 * @param mode - 'rgb' for images or 'grayscale' for masks
 * @returns ImageData object
 */
export function tensorToImageData(tensor: any, mode: 'rgb' | 'grayscale' = 'rgb'): ImageData | null {
    try {
        const shape = tensor.shape;
        const height = shape[1];
        const width = shape[2];
        const channels = shape[3] || 1; // Default to 1 for masks

        log.debug("Converting tensor:", { shape, channels, mode });

        const imageData = new ImageData(width, height);
        const data = new Uint8ClampedArray(width * height * 4);

        const flatData = tensor.data;
        const pixelCount = width * height;

        const min = tensor.min_val ?? 0;
        const max = tensor.max_val ?? 1;
        const denom = (max - min) || 1;

        for (let i = 0; i < pixelCount; i++) {
            const pixelIndex = i * 4;
            const tensorIndex = i * channels;

            let lum: number;
            if (mode === 'grayscale' || channels === 1) {
                lum = flatData[tensorIndex];
            } else {
                // Compute luminance for RGB
                const r = flatData[tensorIndex + 0] ?? 0;
                const g = flatData[tensorIndex + 1] ?? 0;
                const b = flatData[tensorIndex + 2] ?? 0;
                lum = 0.299 * r + 0.587 * g + 0.114 * b;
            }

            let norm = (lum - min) / denom;
            if (!isFinite(norm)) norm = 0;
            norm = Math.max(0, Math.min(1, norm));
            const value = Math.round(norm * 255);

            if (mode === 'grayscale') {
                // For masks: RGB = value, A = 255 (MaskTool reads luminance)
                data[pixelIndex] = value;
                data[pixelIndex + 1] = value;
                data[pixelIndex + 2] = value;
                data[pixelIndex + 3] = 255;
            } else {
                // For images: RGB from channels, A = 255
                for (let c = 0; c < Math.min(3, channels); c++) {
                    const channelValue = flatData[tensorIndex + c];
                    const channelNorm = (channelValue - min) / denom;
                    data[pixelIndex + c] = Math.round(channelNorm * 255);
                }
                data[pixelIndex + 3] = 255;
            }
        }

        imageData.data.set(data);
        return imageData;
    } catch (error) {
        log.error("Error converting tensor:", error);
        return null;
    }
}

/**
 * Creates an HTMLImageElement from ImageData
 * @param imageData - Input ImageData
 * @returns Promise with HTMLImageElement
 */
export async function createImageFromImageData(imageData: ImageData): Promise<HTMLImageElement> {
    const { canvas, ctx } = createCanvas(imageData.width, imageData.height, '2d', { willReadFrequently: true });
    if (!ctx) throw new Error("Could not create canvas context");
    ctx.putImageData(imageData, 0, 0);
    return await createImageFromSource(canvas.toDataURL());
}
