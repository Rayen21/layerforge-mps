from PIL import Image, ImageOps
import hashlib
import torch
import numpy as np
import folder_paths
from server import PromptServer
from aiohttp import web
import asyncio
import threading
import os
from tqdm import tqdm
from torchvision import transforms
try:
    from transformers import AutoModelForImageSegmentation, PretrainedConfig
    from requests.exceptions import ConnectionError as RequestsConnectionError
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
import torch.nn.functional as F
import traceback
import uuid
import time
import base64
import io
import sys

# 设置 PyTorch 浮点矩阵乘法精度，对 GPU 和 MPS 设备有益
torch.set_float32_matmul_precision('high')

# --- 日志和实用函数 (保留原样，没有优化需求) ---
try:
    from python.logger import logger, LogLevel, debug, info, warn, error, exception
    from python.config import LOG_LEVEL
    logger.set_module_level('canvas_node', LogLevel[LOG_LEVEL])
    logger.configure({
        'log_to_file': True,
        'log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    })
    log_debug = lambda *args, **kwargs: debug('canvas_node', *args, **kwargs)
    log_info = lambda *args, **kwargs: info('canvas_node', *args, **kwargs)
    log_warn = lambda *args, **kwargs: warn('canvas_node', *args, **kwargs)
    log_error = lambda *args, **kwargs: error('canvas_node', *args, **kwargs)
    log_exception = lambda *args: exception('canvas_node', *args)
    log_info("Logger initialized for canvas_node")
except ImportError as e:
    print(f"Warning: Logger module not available: {e}")
    def log_debug(*args): print("[DEBUG]", *args)
    def log_info(*args): print("[INFO]", *args)
    def log_warn(*args): print("[WARN]", *args)
    def log_error(*args): print("[ERROR]", *args)
    def log_exception(*args):
        print("[ERROR]", *args)
        traceback.print_exc()

# --- BiRefNet 模型的配置和骨架 (保留原样，没有优化需求) ---
class BiRefNetConfig(PretrainedConfig):
    model_type = "BiRefNet"
    def __init__(self, bb_pretrained=False, **kwargs):
        self.bb_pretrained = bb_pretrained
        super().__init__(**kwargs)

class BiRefNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 1, kernel_size=1)
        )
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return [output]

# --- LayerForgeNode 类 (保留原样，主要用于数据流和缓存管理) ---
class LayerForgeNode:
    _canvas_data_storage = {}
    _storage_lock = threading.Lock()
    _canvas_cache = {
        'image': None,
        'mask': None,
        'data_flow_status': {},
        'persistent_cache': {},
        'last_execution_id': None
    }
    _websocket_data = {}
    _websocket_listeners = {}

    def __init__(self):
        super().__init__()
        self.flow_id = str(uuid.uuid4())
        self.node_id = None
        if self.__class__._canvas_cache['persistent_cache']:
            self.restore_cache()

    def restore_cache(self):
        try:
            persistent = self.__class__._canvas_cache['persistent_cache']
            current_execution = self.get_execution_id()
            if current_execution != self.__class__._canvas_cache['last_execution_id']:
                log_info(f"New execution detected: {current_execution}")
                self.__class__._canvas_cache['image'] = None
                self.__class__._canvas_cache['mask'] = None
                self.__class__._canvas_cache['last_execution_id'] = current_execution
            else:
                if persistent.get('image') is not None:
                    self.__class__._canvas_cache['image'] = persistent['image']
                    log_info("Restored image from persistent cache")
                if persistent.get('mask') is not None:
                    self.__class__._canvas_cache['mask'] = persistent['mask']
                    log_info("Restored mask from persistent cache")
        except Exception as e:
            log_error(f"Error restoring cache: {str(e)}")

    def get_execution_id(self):
        try:
            return str(int(time.time() * 1000))
        except Exception as e:
            log_error(f"Error getting execution ID: {str(e)}")
            return None

    def update_persistent_cache(self):
        try:
            self.__class__._canvas_cache['persistent_cache'] = {
                'image': self.__class__._canvas_cache['image'],
                'mask': self.__class__._canvas_cache['mask']
            }
            log_debug("Updated persistent cache")
        except Exception as e:
            log_error(f"Error updating persistent cache: {str(e)}")

    def track_data_flow(self, stage, status, data_info=None):
        flow_status = {
            'timestamp': time.time(),
            'stage': stage,
            'status': status,
            'data_info': data_info
        }
        log_debug(f"Data Flow [{self.flow_id}] - Stage: {stage}, Status: {status}")
        if data_info:
            log_debug(f"Data Info: {data_info}")
        self.__class__._canvas_cache['data_flow_status'][self.flow_id] = flow_status

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fit_on_add": ("BOOLEAN", {"default": False, "label_on": "Fit on Add/Paste", "label_off": "Default Behavior"}),
                "show_preview": ("BOOLEAN", {"default": False, "label_on": "Show Preview", "label_off": "Hide Preview"}),
                "auto_refresh_after_generation": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 99999999, "step": 1}),
                "node_id": ("STRING", {"default": "0"}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "input_mask": ("MASK",),
            },
            "hidden": {
                "prompt": ("PROMPT",),
                "unique_id": ("UNIQUE_ID",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_canvas_image"
    CATEGORY = "azNodes > LayerForge"

    def add_image_to_canvas(self, input_image):
        try:
            if not isinstance(input_image, torch.Tensor):
                raise ValueError("Input image must be a torch.Tensor")
            if input_image.dim() == 4:
                input_image = input_image.squeeze(0)
            if input_image.dim() == 3 and input_image.shape[0] in [1, 3]:
                input_image = input_image.permute(1, 2, 0)
            return input_image
        except Exception as e:
            log_error(f"Error in add_image_to_canvas: {str(e)}")
            return None

    def add_mask_to_canvas(self, input_mask, input_image):
        try:
            if not isinstance(input_mask, torch.Tensor):
                raise ValueError("Input mask must be a torch.Tensor")
            if input_mask.dim() == 4:
                input_mask = input_mask.squeeze(0)
            if input_mask.dim() == 3 and input_mask.shape[0] == 1:
                input_mask = input_mask.squeeze(0)
            if input_image is not None:
                expected_shape = input_image.shape[:2]
                if input_mask.shape != expected_shape:
                    input_mask = F.interpolate(
                        input_mask.unsqueeze(0).unsqueeze(0),
                        size=expected_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
            return input_mask
        except Exception as e:
            log_error(f"Error in add_mask_to_canvas: {str(e)}")
            return None

    _processing_lock = threading.Lock()

    def process_canvas_image(self, fit_on_add, show_preview, auto_refresh_after_generation, trigger, node_id, input_image=None, input_mask=None, prompt=None, unique_id=None):
        try:
            if not self.__class__._processing_lock.acquire(blocking=False):
                log_warn(f"Process already in progress for node {node_id}, skipping...")
                return self.get_cached_data()
            log_info(f"Lock acquired. Starting process_canvas_image for node_id: {node_id} (fallback unique_id: {unique_id})")
            log_info(f"Storing input data for node {node_id} - Image: {input_image is not None}, Mask: {input_mask is not None}")
            
            with self.__class__._storage_lock:
                input_data = {}
                if input_image is not None:
                    if isinstance(input_image, torch.Tensor):
                        if input_image.dim() == 3:
                            input_image = input_image.unsqueeze(0)
                        batch_size = input_image.shape[0]
                        log_info(f"Processing batch of {batch_size} image(s)")
                        
                        images_array = []
                        for i in range(batch_size):
                            # 将 Tensor 移动到 CPU 以进行 PIL 转换，确保高效
                            img_np = (input_image[i].cpu().numpy() * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_np, 'RGB')
                            buffered = io.BytesIO()
                            pil_img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            images_array.append({
                                'data': f"data:image/png;base64,{img_str}",
                                'width': pil_img.width,
                                'height': pil_img.height
                            })
                            log_debug(f"Stored batch image {i+1}/{batch_size}: {pil_img.width}x{pil_img.height}")
                        
                        if batch_size == 1:
                            input_data['input_image'] = images_array[0]['data']
                            input_data['input_image_width'] = images_array[0]['width']
                            input_data['input_image_height'] = images_array[0]['height']
                        else:
                            input_data['input_images_batch'] = images_array
                        log_info(f"Stored batch of {batch_size} images")
                
                if input_mask is not None:
                    if isinstance(input_mask, torch.Tensor):
                        if input_mask.dim() == 2:
                            input_mask = input_mask.unsqueeze(0)
                        if input_mask.dim() == 3 and input_mask.shape[0] == 1:
                            input_mask = input_mask.squeeze(0)
                        
                        # 将 Tensor 移动到 CPU 以进行 PIL 转换
                        mask_np = (input_mask.cpu().numpy() * 255).astype(np.uint8)
                        pil_mask = Image.fromarray(mask_np, 'L')
                        mask_buffered = io.BytesIO()
                        pil_mask.save(mask_buffered, format="PNG")
                        mask_str = base64.b64encode(mask_buffered.getvalue()).decode()
                        input_data['input_mask'] = f"data:image/png;base64,{mask_str}"
                        log_debug(f"Stored input mask: {pil_mask.width}x{pil_mask.height}")
                
                input_data['fit_on_add'] = fit_on_add
                self.__class__._canvas_data_storage[f"{node_id}_input"] = input_data

            storage_key = node_id
            processed_image = None
            processed_mask = None

            with self.__class__._storage_lock:
                canvas_data = self.__class__._canvas_data_storage.pop(storage_key, None)

            if canvas_data:
                log_info(f"Canvas data found for node {storage_key} from WebSocket")
                if canvas_data.get('image'):
                    image_data = canvas_data['image'].split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    # 直接创建 Tensor，默认在 CPU 上，如果需要可在下游节点手动转移到 GPU/MPS
                    processed_image = torch.from_numpy(image_array)[None,]
                    log_debug(f"Image loaded from WebSocket, shape: {processed_image.shape}")

                if canvas_data.get('mask'):
                    mask_data = canvas_data['mask'].split(',')[1]
                    mask_bytes = base64.b64decode(mask_data)
                    pil_mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
                    mask_array = np.array(pil_mask).astype(np.float32) / 255.0
                    processed_mask = torch.from_numpy(mask_array)[None,]
                    log_debug(f"Mask loaded from WebSocket, shape: {processed_mask.shape}")
            else:
                log_warn(f"No canvas data found for node {storage_key} in WebSocket cache.")

            # 确保返回非 None 的 Tensor
            if processed_image is None:
                log_warn(f"Processed image is still None, creating default blank image.")
                processed_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            if processed_mask is None:
                log_warn(f"Processed mask is still None, creating default blank mask.")
                processed_mask = torch.zeros((1, 512, 512), dtype=torch.float32)

            log_debug(f"About to return output - Image shape: {processed_image.shape}, Mask shape: {processed_mask.shape}")
            self.update_persistent_cache()
            log_info(f"Successfully returning processed image and mask")
            return (processed_image, processed_mask)

        except Exception as e:
            log_exception(f"Error in process_canvas_image: {str(e)}")
            return (None, None)
        finally:
            if self.__class__._processing_lock.locked():
                self.__class__._processing_lock.release()
                log_debug(f"Process completed for node {node_id}, lock released")

    def get_cached_data(self):
        return {
            'image': self.__class__._canvas_cache['image'],
            'mask': self.__class__._canvas_cache['mask']
        }

    # ... 其他路由和方法 (保留原样) ...
    @classmethod
    def api_get_data(cls, node_id):
        try:
            return {
                'success': True,
                'data': cls._canvas_cache
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    def get_latest_image(cls):
        output_dir = folder_paths.get_output_directory()
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if
                 os.path.isfile(os.path.join(output_dir, f))]

        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not image_files:
            return None

        latest_image_path = max(image_files, key=os.path.getctime)
        return latest_image_path

    @classmethod
    def get_latest_images(cls, since_timestamp=0):
        output_dir = folder_paths.get_output_directory()
        files = []
        for f_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f_name)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    mtime = os.path.getmtime(file_path)
                    if mtime > since_timestamp:
                        files.append((mtime, file_path))
                except OSError:
                    continue
        
        files.sort(key=lambda x: x[0])
        
        return [f[1] for f in files]

    @classmethod
    def get_flow_status(cls, flow_id=None):

        if flow_id:
            return cls._canvas_cache['data_flow_status'].get(flow_id)
        return cls._canvas_cache['data_flow_status']

    @classmethod
    def _cleanup_old_websocket_data(cls):
        """Clean up old WebSocket data from invalid nodes or data older than 5 minutes"""
        try:
            current_time = time.time()
            cleanup_threshold = 300  # 5 minutes
            
            nodes_to_remove = []
            for node_id, data in cls._websocket_data.items():

                if node_id < 0:
                    nodes_to_remove.append(node_id)
                    continue

                if current_time - data.get('timestamp', 0) > cleanup_threshold:
                    nodes_to_remove.append(node_id)
                    continue
            
            for node_id in nodes_to_remove:
                del cls._websocket_data[node_id]
                log_debug(f"Cleaned up old WebSocket data for node {node_id}")
            
            if nodes_to_remove:
                log_info(f"Cleaned up {len(nodes_to_remove)} old WebSocket entries")
                
        except Exception as e:
            log_error(f"Error during WebSocket cleanup: {str(e)}")

    @classmethod
    def setup_routes(cls):
        @PromptServer.instance.routes.get("/layerforge/canvas_ws")
        async def handle_canvas_websocket(request):
            ws = web.WebSocketResponse(max_msg_size=33554432)
            await ws.prepare(request)
            
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = msg.json()
                        node_id = data.get('nodeId')
                        if not node_id:
                            await ws.send_json({'status': 'error', 'message': 'nodeId is required'})
                            continue
                        
                        image_data = data.get('image')
                        mask_data = data.get('mask')
                        
                        with cls._storage_lock:
                            cls._canvas_data_storage[node_id] = {
                                'image': image_data,
                                'mask': mask_data,
                                'timestamp': time.time()
                            }
                        
                        log_info(f"Received canvas data for node {node_id} via WebSocket")

                        ack_payload = {
                            'type': 'ack',
                            'nodeId': node_id,
                            'status': 'success'
                        }
                        await ws.send_json(ack_payload)
                        log_debug(f"Sent ACK for node {node_id}")
                        
                    except Exception as e:
                        log_error(f"Error processing WebSocket message: {e}")
                        await ws.send_json({'status': 'error', 'message': str(e)})
                elif msg.type == web.WSMsgType.ERROR:
                    log_error(f"WebSocket connection closed with exception {ws.exception()}")

            log_info("WebSocket connection closed")
            return ws

        @PromptServer.instance.routes.get("/layerforge/get_input_data/{node_id}")
        async def get_input_data(request):
            try:
                node_id = request.match_info["node_id"]
                log_debug(f"Checking for input data for node: {node_id}")
                
                with cls._storage_lock:
                    input_key = f"{node_id}_input"
                    input_data = cls._canvas_data_storage.get(input_key, None)
                
                if input_data:
                    log_info(f"Input data found for node {node_id}, sending to frontend")
                    return web.json_response({
                        'success': True,
                        'has_input': True,
                        'data': input_data
                    })
                else:
                    log_debug(f"No input data found for node {node_id}")
                    return web.json_response({
                        'success': True,
                        'has_input': False
                    })
                    
            except Exception as e:
                log_error(f"Error in get_input_data: {str(e)}")
                return web.json_response({
                    'success': False,
                    'error': str(e)
                }, status=500)

        @PromptServer.instance.routes.post("/layerforge/clear_input_data/{node_id}")
        async def clear_input_data(request):
            try:
                node_id = request.match_info["node_id"]
                log_info(f"Clearing input data for node: {node_id}")
                
                with cls._storage_lock:
                    input_key = f"{node_id}_input"
                    if input_key in cls._canvas_data_storage:
                        del cls._canvas_data_storage[input_key]
                        log_info(f"Input data cleared for node {node_id}")
                    else:
                        log_debug(f"No input data to clear for node {node_id}")
                
                return web.json_response({
                    'success': True,
                    'message': f'Input data cleared for node {node_id}'
                })
                    
            except Exception as e:
                log_error(f"Error in clear_input_data: {str(e)}")
                return web.json_response({
                    'success': False,
                    'error': str(e)
                }, status=500)

        @PromptServer.instance.routes.get("/ycnode/get_canvas_data/{node_id}")
        async def get_canvas_data(request):
            try:
                node_id = request.match_info["node_id"]
                log_debug(f"Received request for node: {node_id}")

                cache_data = cls._canvas_cache
                log_debug(f"Cache content: {cache_data}")
                log_debug(f"Image in cache: {cache_data['image'] is not None}")

                response_data = {
                    'success': True,
                    'data': {
                        'image': None,
                        'mask': None
                    }
                }

                if cache_data['image'] is not None:
                    pil_image = cache_data['image']
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    response_data['data']['image'] = f"data:image/png;base64,{img_str}"

                if cache_data['mask'] is not None:
                    pil_mask = cache_data['mask']
                    mask_buffer = io.BytesIO()
                    pil_mask.save(mask_buffer, format="PNG")
                    mask_str = base64.b64encode(mask_buffer.getvalue()).decode()
                    response_data['data']['mask'] = f"data:image/png;base64,{mask_str}"

                return web.json_response(response_data)

            except Exception as e:
                log_error(f"Error in get_canvas_data: {str(e)}")
                return web.json_response({
                    'success': False,
                    'error': str(e)
                })

        @PromptServer.instance.routes.get("/layerforge/get-latest-images/{since}")
        async def get_latest_images_route(request):
            try:
                since_timestamp = float(request.match_info.get('since', 0))
                # JS Timestamps are in milliseconds, Python's are in seconds
                latest_image_paths = cls.get_latest_images(since_timestamp / 1000.0)

                images_data = []
                for image_path in latest_image_paths:
                    with open(image_path, "rb") as f:
                        encoded_string = base64.b64encode(f.read()).decode('utf-8')
                        images_data.append(f"data:image/png;base64,{encoded_string}")
                
                return web.json_response({
                    'success': True,
                    'images': images_data
                })
            except Exception as e:
                log_error(f"Error in get_latest_images_route: {str(e)}")
                return web.json_response({
                    'success': False,
                    'error': str(e)
                }, status=500)

        @PromptServer.instance.routes.get("/ycnode/get_latest_image")
        async def get_latest_image_route(request):
            try:
                latest_image_path = cls.get_latest_image()
                if latest_image_path:
                    with open(latest_image_path, "rb") as f:
                        encoded_string = base64.b64encode(f.read()).decode('utf-8')
                    return web.json_response({
                        'success': True,
                        'image_data': f"data:image/png;base64,{encoded_string}"
                    })
                else:
                    return web.json_response({
                        'success': False,
                        'error': 'No images found in output directory.'
                    }, status=404)
            except Exception as e:
                return web.json_response({
                    'success': False,
                    'error': str(e)
                }, status=500)

        @PromptServer.instance.routes.post("/ycnode/load_image_from_path")
        async def load_image_from_path_route(request):
            try:
                data = await request.json()
                file_path = data.get('file_path')
                
                if not file_path:
                    return web.json_response({
                        'success': False,
                        'error': 'file_path is required'
                    }, status=400)
                
                log_info(f"Attempting to load image from path: {file_path}")
                
                # Check if file exists and is accessible
                if not os.path.exists(file_path):
                    log_warn(f"File not found: {file_path}")
                    return web.json_response({
                        'success': False,
                        'error': f'File not found: {file_path}'
                    }, status=404)
                
                # Check if it's an image file
                valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.ico', '.avif')
                if not file_path.lower().endswith(valid_extensions):
                    return web.json_response({
                        'success': False,
                        'error': f'Invalid image file extension. Supported: {valid_extensions}'
                    }, status=400)
                
                # Try to load and convert the image
                try:
                    with Image.open(file_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        log_info(f"Successfully loaded image from path: {file_path}")
                        return web.json_response({
                            'success': True,
                            'image_data': f"data:image/png;base64,{img_str}",
                            'width': img.width,
                            'height': img.height
                        })
                        
                except Exception as img_error:
                    log_error(f"Error processing image file {file_path}: {str(img_error)}")
                    return web.json_response({
                        'success': False,
                        'error': f'Error processing image file: {str(img_error)}'
                    }, status=500)
                    
            except Exception as e:
                log_error(f"Error in load_image_from_path_route: {str(e)}")
                return web.json_response({
                    'success': False,
                    'error': str(e)
                }, status=500)
        

    def store_image(self, image_data):
      
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            self.cached_image = Image.open(io.BytesIO(image_bytes))
        else:
            self.cached_image = image_data

    def get_cached_image(self):
        
        if self.cached_image:
            buffered = io.BytesIO()
            self.cached_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        return None

class BiRefNetMatting:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_cache = {}
        # 设置模型基准路径
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        self.device = self._get_best_device()
        print(f"BiRefNetMatting initialized. Determined device: {self.device}")

    def _get_best_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def load_model(self, model_path):
        """
        加载本地BiRefNet模型。
        
        这个版本强制从本地路径加载模型，如果文件不存在，将直接报错。
        """
        try:
            full_model_path = os.path.join(self.base_path, "BiRefNet")
            # 确保模型目录存在，但不再从网络下载
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(
                    f"模型目录不存在。请确保模型文件位于：{full_model_path}"
                )

            # 检查缓存，如果模型已加载则直接使用
            if full_model_path in self.model_cache:
                self.model = self.model_cache[full_model_path]
                if self.model.device != self.device:
                    self.model = self.model.to(self.device)
                print(f"Using cached model. Model is on device: {self.model.device}")
                return
            
            # 从本地目录加载模型
            print(f"Loading BiRefNet model from local path: {full_model_path}...")
            
            # 使用 transformers 的 from_pretrained 方法从本地目录加载模型
            # 这要求模型文件（如 model.safetensors 或 pytorch_model.bin）和配置文件 config.json 必须存在于该目录
            self.model = AutoModelForImageSegmentation.from_pretrained(
                full_model_path,
                trust_remote_code=True
            )
            
            # 确保模型加载后立即转移到目标设备
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_cache[full_model_path] = self.model
            print(f"Model loaded successfully. Model is on device: {self.model.device}")

        except Exception as e:
            # 统一处理加载错误，提示用户检查本地文件
            print(f"Error loading local model from {full_model_path}: {e}")
            log_exception("Local model loading failed")
            raise RuntimeError(
                f"Failed to load the local matting model. "
                f"Please ensure the model files (e.g., model.safetensors and config.json) are correctly placed "
                f"in the directory: {full_model_path}. "
                f"Original error: {str(e)}"
            ) from e

    def preprocess_image(self, image):
        try:
            # 优化: 检查输入 Tensor 是否已经在正确的设备上，避免不必要的传输
            if isinstance(image, torch.Tensor) and image.device != self.device:
                image = image.to(self.device)
            elif not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            
            # 转换过程在设备上完成
            transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = transform_image(image)
            print(f"Preprocessed image tensor on device: {image_tensor.device}")
            return image_tensor
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def execute(self, image, model_path, threshold=0.5, refinement=1):
        try:
            PromptServer.instance.send_sync("matting_status", {"status": "processing"})
            self.load_model(model_path)
            
            # 确保输入图像 Tensor 在正确的设备上
            if isinstance(image, torch.Tensor):
                original_size = image.shape[-2:]
                image = image.to(self.device)
            else:
                original_size = image.size[::-1]
                image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

            print(f"Input image tensor on device after conversion: {image.device}")

            with torch.no_grad():
                processed_image = self.preprocess_image(image)
                if processed_image is None:
                    raise Exception("Failed to preprocess image")
                
                print(f"Processed image tensor before model inference on device: {processed_image.device}")
                
                outputs = self.model(processed_image)
                
                print(f"Model output tensor on device: {outputs[-1].device}")
                
                result = outputs[-1].sigmoid()
                if result.dim() == 3:
                    result = result.unsqueeze(1)
                elif result.dim() == 2:
                    result = result.unsqueeze(0).unsqueeze(0)

                print(f"Reshaped result shape: {result.shape}, device: {result.device}")
                # 确保插值操作也在设备上进行
                result = F.interpolate(
                    result,
                    size=(original_size[0], original_size[1]),
                    mode='bilinear',
                    align_corners=True
                )
                print(f"Resized result shape: {result.shape}, device: {result.device}")
                result = result.squeeze()
                ma = torch.max(result)
                mi = torch.min(result)
                result = (result - mi) / (ma - mi)
                if threshold > 0:
                    result = (result > threshold).float()

                alpha_mask = result.unsqueeze(0).unsqueeze(0)
                # 确保乘法操作在同一设备上
                image_for_mask = image
                alpha_mask_for_mult = alpha_mask
                masked_image = image_for_mask * alpha_mask_for_mult

                PromptServer.instance.send_sync("matting_status", {"status": "completed"})

                # 优化: 返回前将结果移动到 CPU，以兼容下游非 GPU 节点
                return (masked_image.cpu(), alpha_mask.cpu())

        except Exception as e:
            PromptServer.instance.send_sync("matting_status", {"status": "error"})
            raise e

    @classmethod
    def IS_CHANGED(cls, image, model_path, threshold, refinement):
        m = hashlib.md5()
        # 优化: 更好的哈希方法，对 Tensor 状态进行哈希
        if isinstance(image, torch.Tensor):
            m.update(image.numpy().tobytes())
        else:
            m.update(str(image).encode())
        m.update(str(model_path).encode())
        m.update(str(threshold).encode())
        m.update(str(refinement).encode())
        return m.hexdigest()

_matting_lock = None

@PromptServer.instance.routes.post("/matting")
async def matting(request):
    global _matting_lock
    if not TRANSFORMERS_AVAILABLE:
        print("Matting request failed: 'transformers' library is not installed.")
        return web.json_response({
            "error": "Dependency Not Found",
            "details": "The 'transformers' library is required for the matting feature. Please install it by running: pip install transformers"
        }, status=400)
    if _matting_lock is not None:
        print("Matting already in progress, rejecting request")
        return web.json_response({
            "error": "Another matting operation is in progress",
            "details": "Please wait for the current operation to complete"
        }, status=429)
    _matting_lock = True
    try:
        print("Received matting request")
        data = await request.json()
        matting_instance = BiRefNetMatting()
        print(f"Using device: {matting_instance.device}")
        
        # 优化: 将设备信息传递给转换函数
        image_tensor, original_alpha = convert_base64_to_tensor(data["image"], matting_instance.device)
        print(f"Input image tensor on device after conversion: {image_tensor.device}")
        
        matted_image, alpha_mask = matting_instance.execute(
            image_tensor,
            "BiRefNet/model.safetensors",
            threshold=data.get("threshold", 0.5),
            refinement=data.get("refinement", 1)
        )
        # 优化: 返回前将结果转换回 base64 格式
        result_image = convert_tensor_to_base64(matted_image, alpha_mask, original_alpha)
        result_mask = convert_tensor_to_base64(alpha_mask)
        return web.json_response({
            "matted_image": result_image,
            "alpha_mask": result_mask
        })
    except RuntimeError as e:
        print(f"Runtime error during matting: {e}")
        return web.json_response({
            "error": "Matting Model Error",
            "details": str(e)
        }, status=500)
    except Exception as e:
        print(f"Error in matting endpoint: {e}")
        return web.json_response({
            "error": "An unexpected error occurred",
            "details": traceback.format_exc()
        }, status=500)
    finally:
        _matting_lock = None
        print("Matting lock released")

def convert_base64_to_tensor(base64_str, device):
    import base64
    import io
    try:
        img_data = base64.b64decode(base64_str.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        has_alpha = img.mode == 'RGBA'
        alpha = None
        if has_alpha:
            alpha = img.split()[3]
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=alpha)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(device) # 确保 Tensor 在正确的设备上
        print(f"Image tensor after base64 conversion is on device: {img_tensor.device}")
        if has_alpha:
            alpha_tensor = transforms.ToTensor()(alpha).unsqueeze(0).to(device)
            return img_tensor, alpha_tensor
        return img_tensor, None
    except Exception as e:
        print(f"Error in convert_base64_to_tensor: {str(e)}")
        raise

def convert_tensor_to_base64(tensor, alpha_mask=None, original_alpha=None):
    import base64
    import io
    try:
        # 优化: 在操作前将所有 Tensor 移动到 CPU
        tensor = tensor.cpu()
        if alpha_mask is not None:
            alpha_mask = alpha_mask.cpu()
        if original_alpha is not None:
            original_alpha = original_alpha.cpu()
        
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
        img_array = (tensor.numpy() * 255).astype(np.uint8)
        
        if alpha_mask is not None and original_alpha is not None:
            alpha_mask = alpha_mask.squeeze().numpy()
            alpha_mask = (alpha_mask * 255).astype(np.uint8)
            original_alpha = original_alpha.squeeze().numpy()
            original_alpha = (original_alpha * 255).astype(np.uint8)
            combined_alpha = np.minimum(alpha_mask, original_alpha)
            img = Image.fromarray(img_array, mode='RGB')
            alpha_img = Image.fromarray(combined_alpha, mode='L')
            img.putalpha(alpha_img)
        else:
            if img_array.shape[-1] == 1:
                img_array = img_array.squeeze(-1)
                img = Image.fromarray(img_array, mode='L')
            else:
                img = Image.fromarray(img_array, mode='RGB')
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error in convert_tensor_to_base64: {str(e)}")
        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        raise

# 注册路由
if __name__ == '__main__':
    LayerForgeNode.setup_routes()
