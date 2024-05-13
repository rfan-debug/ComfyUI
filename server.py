import os
import shutil
import sys
import asyncio
import tempfile
import traceback

import websocket
import nodes
import folder_paths
import execution
import uuid
import urllib
import urllib.parse
import urllib.request
import json
import glob
import struct
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
from google.cloud import storage

import aiohttp
from aiohttp import web
import logging

import mimetypes
from comfy.cli_args import args
import comfy.utils
import comfy.model_management

from app.user_manager import UserManager
from conversion_utils import lora_convert_and_save
from gcs_utils import download_gcs_file


class BinaryEventTypes:
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2


async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
        logging.warning("send error: {}".format(err))


@web.middleware
async def cache_control(request: web.Request, handler):
    response: web.Response = await handler(request)
    if request.path.endswith('.js') or request.path.endswith('.css'):
        response.headers.setdefault('Cache-Control', 'no-cache')
    return response


def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    return cors_middleware


class PromptServer():
    def __init__(self, loop):
        PromptServer.instance = self

        mimetypes.init()
        mimetypes.types_map['.js'] = 'application/javascript; charset=utf-8'

        self.user_manager = UserManager()
        self.supports = ["custom_nodes_from_web"]
        self.prompt_queue = None
        self.loop = loop
        self.messages = asyncio.Queue()
        self.number = 0

        middlewares = [cache_control]
        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))

        max_upload_size = round(args.max_upload_size * 1024 * 1024)
        self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)
        self.sockets = dict()
        self.web_root = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "web")
        routes = web.RouteTableDef()
        self.routes = routes
        self.last_node_id = None
        self.client_id = None

        self.on_prompt_handlers = []
        self.gcs_client = storage.Client()
        self.weight_type2path = {
            "lora": "./models/loras",
            "base": "./models/checkpoints",
            "controlnet": "./models/controlnet",
        }

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            self.sockets[sid] = ws

            try:
                # Send initial state to the new client
                await self.send("status", {"status": self.get_queue_info(), 'sid': sid}, sid)
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:
                    await self.send("executing", {"node": self.last_node_id}, sid)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        logging.warning('ws connection closed with exception %s' % ws.exception())
            finally:
                self.sockets.pop(sid, None)
            return ws

        @routes.get("/")
        async def get_root(request):
            return web.FileResponse(os.path.join(self.web_root, "index.html"))

        @routes.get("/embeddings")
        def get_embeddings(self):
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/extensions")
        async def get_extensions(request):
            files = glob.glob(os.path.join(
                glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)

            extensions = list(map(lambda f: "/" + os.path.relpath(f, self.web_root).replace("\\", "/"), files))

            for name, dir in nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(
                    name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type):
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        def image_upload(post, image_save_function=None):
            image = post.get("image")
            overwrite = post.get("overwrite")

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400)

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if image_save_function is not None:
                    image_save_function(image, post, filepath)
                else:
                    with open(filepath, "wb") as f:
                        f.write(image.file.read())

                return web.json_response({"name": filename, "subfolder": subfolder, "type": image_upload_type})
            else:
                return web.Response(status=400)

        def _image_upload_all(image_files,
                              overwrite: bool,
                              image_upload_type: str | None,
                              subfolder: str = "") -> dict:
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)
            responses = []
            for filename, image_file_data in image_files.items():

                if not filename:
                    raise RuntimeError("Filename not found in image upload.")

                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    raise RuntimeError("Invalid common path in image upload.")

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                with open(filepath, "wb") as f:
                    f.write(image_file_data.read())

                responses.append({"name": filename, "subfolder": subfolder, "type": image_upload_type})

            return {"image_uploads": responses}

        def _fetch_weight(weight_url: str,
                          weight_type: str,
                          local_file_name: str,
                          convert_weight: bool,
                          ) -> (dict, bool):
            if weight_type in self.weight_type2path:
                target_path = self.weight_type2path[weight_type]
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_save_path = os.path.join(temp_dir, local_file_name)
                        final_save_path = os.path.join(target_path, local_file_name)
                        logging.info(f"Downloading file to {temp_save_path}.")
                        logging.info(f"Final save path to {final_save_path}")
                        download_gcs_file(
                            gcs_client=self.gcs_client,
                            uri=weight_url,
                            target_file_path=temp_save_path,
                        )
                        if weight_type == "lora" and convert_weight:
                            lora_convert_and_save(
                                input_lora=temp_save_path,
                                output_lora=final_save_path,
                            )
                        else:
                            shutil.move(
                                src=temp_save_path,
                                dst=final_save_path,
                            )
                        return {
                            "weight_type": weight_type,
                            "weight_url": weight_url,
                            "target_path": target_path,
                            "convert_weight": convert_weight,
                            "download_status": "success",
                        }, True
                except Exception as e:
                    return {
                        "weight_type": weight_type,
                        "weight_url": weight_url,
                        "target_path": target_path,
                        "convert_weight": convert_weight,
                        "download_status": f"Failed with exception: {e}",
                    }, False
            else:
                return {"download_status": "Invalid weight_type: {weight_type}."}, False

        # Start: for collecting results from prompting
        def _get_single_image(filename, subfolder, folder_type):
            data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
            url_values = urllib.parse.urlencode(data)
            with urllib.request.urlopen(f"http://localhost:{self.port}/view?{url_values}") as response:
                return response.read()

        def _get_history(prompt_id):
            with urllib.request.urlopen(f"http://localhost:{self.port}/history/{prompt_id}") as response:
                return json.loads(response.read())

        def _stream_output_images_from_socket(ws, prompt_id):
            output_images = {}
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message["type"] == "executing":
                        data = message["data"]
                        if data["node"] is None and data["prompt_id"] == prompt_id:
                            break  # Execution is done
                else:
                    continue  # previews are binary data

            history = _get_history(prompt_id)[prompt_id]
            for o in history["outputs"]:
                for node_id in history["outputs"]:
                    node_output = history["outputs"][node_id]
                    images_output = []
                    if "images" in node_output:
                        for image in node_output["images"]:
                            image_data = _get_single_image(image["filename"], image["subfolder"], image["type"])
                            images_output.append(image_data)
                    output_images[node_id] = images_output

            return output_images

        def _queue_prompt_for_socket(prompt):
            p = {"prompt": prompt, "client_id": self.client_id}
            data = json.dumps(p).encode("utf-8")
            req = urllib.request.Request(f"http://localhost:{self.port}/prompt", data=data)
            return json.loads(urllib.request.urlopen(req).read())

        def _get_images_from_socket(ws, prompt):
            prompt_id = _queue_prompt_for_socket(prompt)["prompt_id"]
            output_images = _stream_output_images_from_socket(ws, prompt_id)
            return output_images

        def _process_prompt(prompt, client_id) -> dict:
            ws = websocket.WebSocket()
            try:
                ws.connect(
                    f"ws://localhost:{self.port}/ws?clientId={client_id}",
                    timeout=1000,  # Fixed timeouts for local connections
                )
                images = _get_images_from_socket(ws, prompt)
                return images
            except Exception as e:
                raise e
            finally:
                ws.close()

        def _api_inference(post):
            data = await post.json()

            # Extracting data from the request
            weights = data.get('weights', [])
            prompt = data.get('prompt', {})
            client_id = data.get('client_id', '')

            # Upload images
            _image_upload_all(
                post.files,
                overwrite=data.get["image_overwrite"],
                image_upload_type=data.get("image_upload_type", None),
            )
            # Fetch weights
            for weight_info in weights:
                result_dict, is_success = _fetch_weight(
                    weight_url=weight_info["weight_url"],
                    weight_type=weight_info["weight_type"],
                    local_file_name=weight_info["local_file_name"],
                    convert_weight=weight_info["convert_weight"],
                )
                if not is_success:
                    return web.json_response(status=400, text="Unsuccessful weight upload!")

            # process comfy_prompt to get images.
            generated_images = _process_prompt(prompt=prompt, client_id=client_id)
            return generated_images

        # End: for collecting results from prompting

        @routes.post("/api/inference")
        async def inference(request):
            post = await request.post()
            return _api_inference(post)

        @routes.get("/health")
        async def health(request):
            return web.Response(status=200, text="Healthy")

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            return image_upload(post)

        @routes.post("/upload/mask")
        async def upload_mask(request):
            post = await request.post()

            def image_save_function(image, post, filepath):
                original_ref = json.loads(post.get("original_ref"))
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil, 'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')
                        mask_pil = Image.open(image.file).convert('RGBA')

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')
                        original_pil.putalpha(new_alpha)
                        original_pil.save(filepath, compress_level=4, pnginfo=metadata)

            return image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]
                filename, output_dir = folder_paths.annotated_filepath(filename)

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = request.rel_url.query.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if "subfolder" in request.rel_url.query:
                    full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                filename = os.path.basename(filename)
                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    if 'preview' in request.rel_url.query:
                        with Image.open(file) as img:
                            preview_info = request.rel_url.query['preview'].split(';')
                            image_format = preview_info[0]
                            if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                                image_format = 'webp'

                            quality = 90
                            if preview_info[-1].isdigit():
                                quality = int(preview_info[-1])

                            buffer = BytesIO()
                            if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                                img = img.convert("RGB")
                            img.save(buffer, format=image_format, quality=quality)
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    if 'channel' not in request.rel_url.query:
                        channel = 'rgba'
                    else:
                        channel = request.rel_url.query["channel"]

                    if channel == 'rgb':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                r, g, b, a = img.split()
                                new_img = Image.merge('RGB', (r, g, b))
                            else:
                                new_img = img.convert("RGB")

                            buffer = BytesIO()
                            new_img.save(buffer, format='PNG')
                            buffer.seek(0)

                            return web.Response(body=buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})

                    elif channel == 'a':
                        with Image.open(file) as img:
                            if img.mode == "RGBA":
                                _, _, _, a = img.split()
                            else:
                                a = Image.new('L', img.size, 255)

                            # alpha img
                            alpha_img = Image.new('RGBA', img.size)
                            alpha_img.putalpha(a)
                            alpha_buffer = BytesIO()
                            alpha_img.save(alpha_buffer, format='PNG')
                            alpha_buffer.seek(0)

                            return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                                headers={"Content-Disposition": f"filename=\"{filename}\""})
                    else:
                        return web.FileResponse(file, headers={"Content-Disposition": f"filename=\"{filename}\""})

            return web.Response(status=404)

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404)
            if not "filename" in request.rel_url.query:
                return web.Response(status=404)

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404)

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(status=404)
            out = comfy.utils.safetensors_header(safetensors_path, max_size=1024 * 1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if not "__metadata__" in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/view_file/{folder_name}")
        async def view_file(request):
            folder_name = request.match_info.get("folder_name", None)
            if folder_name is None:
                return web.Response(status=404, text="Request pass in a valid folder name")
            if not "filename" in request.rel_url.query:
                return web.Response(status=404, text="Require filename in params")

            filename = request.rel_url.query["filename"]
            if not filename.endswith(".safetensors"):
                return web.Response(status=404, text="target file isn't a .safetensors")

            safetensors_path = folder_paths.get_full_path(folder_name, filename)
            if safetensors_path is None:
                return web.Response(
                    status=404,
                    text="File not found")
            else:
                return web.Response(
                    status=200,
                    text=f"File Exists: {folder_name}/{filename}"
                )

        @routes.post("/fetch_weight")
        async def fetch_weight(request):
            post = await request.post()
            result_dict, is_success = _fetch_weight(
                weight_url=post.get("weight_url"),
                weight_type=post.get("weight_type"),
                local_file_name=post.get("local_file_name"),
                convert_weight=post.get("convert_weight", True),
            )

            if is_success:
                return web.json_response(data=result_dict, status=200)
            else:
                return web.json_response(data=result_dict, status=400)

        @routes.get("/system_stats")
        async def get_queue(request):
            device = comfy.model_management.get_torch_device()
            device_name = comfy.model_management.get_torch_device_name(device)
            vram_total, torch_vram_total = comfy.model_management.get_total_memory(device, torch_total_too=True)
            vram_free, torch_vram_free = comfy.model_management.get_free_memory(device, torch_free_too=True)
            system_stats = {
                "system": {
                    "os": os.name,
                    "python_version": sys.version,
                    "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded"
                },
                "devices": [
                    {
                        "name": device_name,
                        "type": device.type,
                        "index": device.index,
                        "vram_total": vram_total,
                        "vram_free": vram_free,
                        "torch_vram_total": torch_vram_total,
                        "torch_vram_free": torch_vram_free,
                    }
                ]
            }
            return web.json_response(system_stats)

        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())

        def node_info(node_class):
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [
                                                                                                               False] * len(
                obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[
                node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
            info['description'] = obj_class.DESCRIPTION if hasattr(obj_class, 'DESCRIPTION') else ''
            info['category'] = 'sd'
            if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
                info['output_node'] = True
            else:
                info['output_node'] = False

            if hasattr(obj_class, 'CATEGORY'):
                info['category'] = obj_class.CATEGORY
            return info

        @routes.get("/object_info")
        async def get_object_info(request):
            out = {}
            for x in nodes.NODE_CLASS_MAPPINGS:
                try:
                    out[x] = node_info(x)
                except Exception as e:
                    logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                    logging.error(traceback.format_exc())
            return web.json_response(out)

        @routes.get("/available_models")
        async def get_available_models(request):
            out = {}

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)
            return web.json_response(self.prompt_queue.get_history(max_items=max_items))

        @routes.get("/history/{prompt_id}")
        async def get_history(request):
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

        @routes.get("/queue")
        async def get_queue(request):
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue()
            queue_info['queue_running'] = current_queue[0]
            queue_info['queue_pending'] = current_queue[1]
            return web.json_response(queue_info)

        @routes.post("/prompt")
        async def post_prompt(request):
            logging.info("got prompt")
            resp_code = 200
            out_string = ""
            json_data = await request.json()
            json_data = self.trigger_on_prompt(json_data)

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                valid = execution.validate_prompt(prompt)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                if valid[0]:
                    prompt_id = str(uuid.uuid4())
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
                    response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                    return web.json_response(response)
                else:
                    logging.warning("invalid prompt: {}".format(valid[1]))
                    return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
            else:
                return web.json_response({"error": "no prompt", "node_errors": []}, status=400)

        @routes.post("/queue")
        async def post_queue(request):
            json_data = await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):
            nodes.interrupt_processing()
            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):
            json_data = await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)

    def add_routes(self):
        self.user_manager.add_routes(self.routes)
        self.app.add_routes(self.routes)

        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([
                web.static('/extensions/' + urllib.parse.quote(name), dir),
            ])

        self.app.add_routes([
            web.static('/', self.web_root),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))

    def queue_updated(self):
        self.send_sync("status", {"status": self.get_queue_info()})

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port, verbose=True, call_on_start=None):
        self.port = port
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, address, port)
        await site.start()

        if verbose:
            logging.info("Starting server\n")
            logging.info("To see the GUI go to: http://{}:{}".format(address, port))
        if call_on_start is not None:
            call_on_start(address, port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception as e:
                logging.warning(f"[ERROR] An error occurred during the on_prompt_handler processing")
                logging.warning(traceback.format_exc())

        return json_data
