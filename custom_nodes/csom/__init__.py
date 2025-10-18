from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import time
import logging
import nodes
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random
import importlib
import importlib

import comfy.ldm.wan.model as wan_model
importlib.reload(wan_model)

import os, sys
path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

from . import implem

class ReloadNode(io.ComfyNode):
    @classmethod
    def define_schema(cls, *args, **kwargs) -> io.Schema:
        importlib.reload(implem)
        return implem.ReloadNode.define_schema(*args, **kwargs)

    @classmethod
    def execute(cls, *args, **kwargs):
        importlib.reload(implem)
        return implem.ReloadNode.execute(*args, **kwargs                           )

    # optional method to control when the node is re executed.
    @classmethod
    def fingerprint_inputs(cls, *args, **kwargs): #s, image, string_field, int_field, float_field, print_to_screen):
        importlib.reload(implem)
        return implem.ReloadNode.fingerprint_inputs(*args, **kwargs)

import importlib
import implem

def make_node_proxy(class_name: str, with_fingerprint: bool = False):
    def _target():
        importlib.reload(implem)
        return getattr(implem, class_name)

    class _Proxy(io.ComfyNode):
        @classmethod
        def define_schema(cls, *args, **kwargs):
            return _target().define_schema(*args, **kwargs)

        @classmethod
        def execute(cls, *args, **kwargs):
            return _target().execute(*args, **kwargs)

    # Add fingerprint_inputs only if requested
    if with_fingerprint:
        @classmethod
        def fingerprint_inputs(cls, *args, **kwargs):
            return _target().fingerprint_inputs(*args, **kwargs)
        _Proxy.fingerprint_inputs = fingerprint_inputs

    _Proxy.__name__ = class_name
    return _Proxy

class CSomExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        importlib.reload(implem)
        logging.info("#KES# get_node_list")
        return [
            ReloadNode,
            make_node_proxy("DWPoseKeysNode", True),
            make_node_proxy("PasteImageNode", True),
            make_node_proxy("CalibrationFrameNode", True),
            make_node_proxy("FixColorNode", True),
        ]

async def comfy_entrypoint() -> CSomExtension:  # ComfyUI calls this to load your extension and its nodes.
    logging.info("#KES# comfy_entrypoint")
    importlib.reload(implem)
    return CSomExtension()

logging.info("#KES1# Version 4.1")