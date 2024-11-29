import argparse
import cv2
import glob
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer, PreloadTorchRealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import numpy as np
from fastapi import FastAPI, File, UploadFile
import base64
import uvicorn
import gc
import sys
from multiprocessing import Process, Queue

upsampler = None
model_name = 'RealESRGAN_x4plus'
model_path = None
tile = 0 # Tile size, 0 for no tile during testing
tile_pad = 10 # Tile padding
pre_pad = 0 # Pre padding size at each border
gpu_id = None # gpu device to use (default=None) can be 0,1,2 for multi-gpu
fp32 = False # Use fp32 precision during inference. 
if gpu_id == "cpu":
    fp32 = True
outscale = 1

def init_upsampler(model_path, gpu_id = None):
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    else:
        raise NotImplementedError

    if model_path is None:
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
                
    if isinstance(model_path, list):
        raise NotImplementedError
        # # dni
        # assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
        # loadnet = self.dni(model_path[0], model_path[1], dni_weight)
    else:
        # if the model_path starts with https, it will first download models to the folder: weights
        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))

    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    # initialize model
    if gpu_id == "cpu":
        device = torch.device('cpu')
    elif gpu_id:
        device = torch.device(
            f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    upsampler = PreloadTorchRealESRGANer(
        scale=netscale,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        device=device)
    
    return upsampler

def general_restoration(restoration_model, bgr_img):
    output, _ = restoration_model.enhance(bgr_img, outscale=outscale)
    return output

class ProcessManager:
    def __init__(self, target, args = ()):
        self.result_queue = Queue()
        self.target = target
        self.args = args
        self.p = None

    def worker(self, args):
        results = self.target(args)
        self.result_queue.put(results)

    def run(self):
        self.p = Process(target=self.worker, args=self.args)
        self.p.start()
        self.p.join()

        results = "-1"
        while not self.result_queue.empty():
            results = self.result_queue.get()

        self.p.terminate()
        return results


def process_general_restoration(bgr_img):
    upsampler = init_upsampler(model_path)
    output_face = general_restoration(upsampler, bgr_img)
    hq_img_bgr = cv2.cvtColor(output_face, cv2.COLOR_RGB2BGR)
    res, im_png = cv2.imencode(".png", hq_img_bgr)
    jpg_as_text = base64.b64encode(im_png)
    return jpg_as_text


app = FastAPI()

@app.post("/api/general_image_restoration")
async def general_image_restoration(file: UploadFile = File(...)):
    '''
        Perform face super-resolution
    '''
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        pm = ProcessManager(target=process_general_restoration, args=(bgr_img,))
        jpg_as_text = pm.run()

        response = {
            "is_success": True,
            "msg": "Success",
            "results": 'data:image/jpg;base64,' + jpg_as_text.decode('utf-8')
        }

        print(len(response["results"]))

    except Exception as e:
        response = {
            "is_success": False,
            "msg": "Server error",
            "results": str(e)
        }
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=17799)