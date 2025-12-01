from comfy_api.latest import io
import time
import importlib
import logging
import nodes
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import json
import torch
from controlnet_aux.open_pose import util, draw_poses, PoseResult
from controlnet_aux.open_pose.body import Keypoint, BodyResult
import numpy as np
import os

import paste_image_node
importlib.reload(paste_image_node)
from paste_image_node import PasteImageNode as PasteImageNode
from paste_image_node import SimplePasteImageNode as SimplePasteImageNode

import color_match
importlib.reload(color_match)

_executor = ThreadPoolExecutor(max_workers=1)

class ReloadNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSomReload",
            display_name="CSom Reload Node",
            category="",
            inputs=[
            ],
            outputs=[
                # io.Int.Output()
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls): #, image, string_field, int_field, float_field, print_to_screen) -> io.NodeOutput:
        logging.info("#KES6#")
        try:
            _executor.submit(lambda: asyncio.run(nodes.load_custom_node("/workspace/ComfyUI/custom_nodes/csom"))).result()
        except Exception as e:
            logging.error(f"load_custom_node failed")
            logging.error(f"{e}")
        return io.NodeOutput()

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # optional method to control when the node is re executed.
    @classmethod
    def fingerprint_inputs(cls): #s, image, string_field, int_field, float_field, print_to_screen):
        logging.info("#KES6#")
        return random.randint(0, 1E9) #int(time.time())

class DWPoseKeysNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSomDWPose",
            display_name="CSom DWPoseKeys Node",
            category="",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("kps0"),
                io.Custom("POSE_KEYPOINT").Input("kps1"),
                
                io.Int.Input("length", default=49, min=1),
            ],
            outputs=[
                io.Image.Output(),
                io.Custom("CSOM_PASTE_DATA").Output("paste_data")
            ],
        )

    @classmethod
    def execute(cls, kps0, kps1, length) -> io.NodeOutput:
        js = [kps0[0], kps1[0]]
        ds = [(j["canvas_height"], j["canvas_width"]) for j in js]
        ks = [np.asarray(j["people"][0]["pose_keypoints_2d"]).reshape(-1, 3) for j in js]

        h, w = ds[0]

        def bb(a):
            a = a[a[:, 2] > 0, :2]
            xymin = a.min(axis=0)
            xymax = a.max(axis=0)
            return xymin, xymax

        # NOSE, NECK, EYE_L< E
        sel = [0,1,2,5,14,15,16,17]
        bbs = [bb(k[sel][:]) for k in ks]

        # cts = [((bbs[i][1] + bbs[i][0])) * 0.5 for i in range(2)]
        # mds = [np.max(bbs[i][1] - bbs[i][0])for i in range(2) ]
        cts = [(ks[i][14,:2] + ks[i][15,:2])/2 for i in range(2)]
        mds = [np.linalg.norm(ks[i][14,:2] - ks[i][15,:2]) for i in range(2) ]

        logging.info(f"{ks[0][sel,:2]}")
        logging.info(f"{ks[1][sel,:2]}")
        logging.info(f"{cts}")
        logging.info(f"{mds}")

        for i in range(2):
            ks[i][~np.isin(np.arange(18), sel), 2] = 0
            sc = mds[0] / mds[i] / (1.0 + i * 0.0)
            tr = cts[0] - cts[i] * sc
            logging.info(f"{sc} {tr} {tr + ks[i][sel,:2] * sc}")
            ks[i][:,:2] = (tr + ks[i][:,:2] * sc) / (w, h)
            if i == 1:
                paste_data = (((0, 0) - tr) / sc).tolist() + (((w, h) - tr) / sc).tolist()


        logging.info(f"{ks[0][sel,:2]}")
        logging.info(f"{ks[1][sel,:2]}")
        # h = h // 2 // 16 * 16
        # w = w // 2 // 16 * 16
        # h, w = 720, 720
        # print(h, w)
        # ks[0][~np.isin(np.arange(18), [0,14,15,16,17]), 2] = 0
        # ks[1][5:,2] = 0
        images = []
        for i, s in enumerate(np.linspace(0, 1.0, length)):
            t = np.clip(s ** 1.0, 0.0, 1.0)
            body_xyc = (ks[0] * (1-t) + ks[1] * t)
            if t != 0 and t != 1:
                body_xyc[:,2] *= ks[0][:,2] * ks[1][:,2]
            keypoints: List[Keypoint] = []
            for idx, (x, y, s) in enumerate(body_xyc):
                if s <= 0.0:
                    keypoints.append(None)
                else:
                    keypoints.append(Keypoint(x=float(x), y=float(y), score=float(s), id=idx))
            body_result = BodyResult(
                keypoints=keypoints,
                total_score=float(body_xyc[:, 2].sum()),
                total_parts=int((body_xyc[:, 2] > 0).sum()),
            )

            pose = PoseResult(
                body=body_result,
                left_hand=None,
                right_hand=None,
                face=None,
            )

            # This uses util.draw_bodypose / draw_handpose / draw_facepose internally
            canvas = draw_poses([pose], h, w, draw_body=True, draw_hand=False, draw_face=False)
            images.append(torch.from_numpy(canvas).unsqueeze(0))
            
        return io.NodeOutput(
            torch.cat(images, dim=0),
            paste_data,
        )

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    @classmethod
    def fingerprint_inputs(*args, **kwargs):
        v = int(os.path.getmtime(__file__))
        return f"{v}"



class CalibrationFrameNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSomCalibrationFrame",
            display_name="CSom Calibration Frame",
            category="",
            inputs=[
                io.Image.Input("images"),
            ],
            outputs=[
                io.Image.Output(display_name="framed"),
                io.Image.Output(display_name="frame"),
            ],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        l = 16 * 6
        frame = torch.zeros((1, images.shape[1] + 2 * l, images.shape[2] + 2 * l, 4), dtype=images.dtype, device=images.device)

        def col(h, x):
            h = (h % 1) * 6
            # b = 0.4, 0.8, 0.4
            # a: 0. 0.1 0.3
            # c. = b[0]/2 + x * (1-b[1]/2-b[0]/2)
            # a = c - (b[1]-b[0])/2
            # a = b[0]/2 + x * (1-b[1]/2-b[0]/2) - b[x]/2
            bx = 0.3
            b = bx + x * (1 - x) * 4.0 * (0.6 - bx)
            a = bx*0.5 + x * (1-bx) - b * 0.5
            return tuple(
                max(0, min(1, abs(((h + 3 - n*2) % 6) - 3) - 1)) * b + a
                for n in range(3)
            ) + (1,)

        cc = torch.tensor([0.91, 0.37, 0.66, 1.0])
        h, w= frame.shape[1:3]
        m = 2*w+h-6*l
        for i in range(h-2*l):
            f = i/(h-2*l-1)
            frame[:,l+i,:l,:] = torch.tensor([f,f,f,1.0])
            frame[:,l+i,-l:,:] = torch.tensor([col((i + w - 2 * l) / m, (j//16*16) / (l-16)) for j in range(l)]).reshape(1, l, 4)
        for i in range(w-2*l):
            frame[:,:l,l+i,:] = torch.tensor([col(i / m, (j//16*16) / (l-16))  for j in range(l)]).reshape(1, l, 4)
            frame[:,-l:,l+i,:] = torch.tensor([col((w - 2 * l - 1 - i + w - 2 * l + h - 2 * l) / m, (j//16*16) / (l-16))  for j in range(l)]).reshape(1, l, 4)

        frame[:,:l,:l,:] = frame[:,-l:,:l,:] = frame[:,:l,-l:,:] = frame[:,-l:,-l:,:] = cc #torch.tensor((0.5, 0.5, 0.5, 1.0))

        out = frame.repeat(images.shape[0], 1, 1, 1)

        out[:, l:-l, l:-l, :] = images

        return io.NodeOutput(
            out,
            frame,
        )

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    @classmethod
    def fingerprint_inputs(*args, **kwargs):
        v = int(os.path.getmtime(__file__))
        logging.info(f"#KES3# fingerprint_inputs {v}")
        return "" #v


class FixColorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSomFixColor",
            display_name="CSom Fix Color",
            category="",
            inputs=[
                io.Image.Input("images"),
                io.Image.Input("frame"),
            ],
            outputs=[
                io.Image.Output(display_name="fixed"),
            ],
        )

    @classmethod
    def execute(cls, images, frame) -> io.NodeOutput:
        l = 16 * 3

        images = images.detach().cpu().numpy()
        frame = frame.detach().cpu().numpy()

        # out = images[:,l:-l,l:-l,:].detach().cpu().numpy()
        # out = (images.unsqueeze(-1)).detach().cpu().numpy()

        out = np.concatenate([images, np.repeat(1.0 - frame[:,:,:,3:], images.shape[0], axis=0)], axis=-1)
        # frame = frame.detach().cpu().numpy()

        g = []
        for i, img in enumerate(images):
            src = np.concatenate([img[None, ...], frame[:,:,:,3:]], axis=-1)
            # params = color_match.build_params(src, frame)
            # print("#KKKKEKESSS#", params)
            # out[i:i+1,:,:,:] = color_match.apply_params(out[i:i+1], params)



            mm1 = np.where(frame[...,3].reshape(-1) > 0.5)[0]
            print(mm1[:10])
            ff = frame[...,:3].reshape(-1, 3)[mm1,:3]
            cc = src[...,:3].reshape(-1, 3)[mm1,:3]
            print("Luuttt....")
            mm = 8
            nn = mm * mm
            # lut = color_match.build_lut(ff, cc, n=nn)
            lut = color_match.build_lut3d_from_pairs_laplacian(ff, cc, n=nn)
            
            print("Apply....")
            out = color_match.apply_lut(out, lut)

            # g.append(torch.tensor(lut.reshape(mm, mm, nn, nn, 3).transpose(0, 2, 1, 3, 4).reshape(1, mm * nn, mm * nn, 3)))

        return io.NodeOutput(
            # torch.cat(g, dim=0)
            torch.tensor(out[:,:,:,:3]) # * out[:,:,:,3:]),
        )

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    @classmethod
    def fingerprint_inputs(*args, **kwargs):
        v = int(os.path.getmtime(__file__))
        return f"{v}-{color_match.modiftime()}"
