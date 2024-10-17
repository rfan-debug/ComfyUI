"""
@author: eesahe
@title: eesahe's Nodes
@nickname: eesahesNodes
@description: InstantX's Flux union ControlNet loader and implementation
"""

from .nodes import InstantXFluxUnionControlNetLoader
from .nodes import ControlNetApplyAdvancedMasked

NODE_CLASS_MAPPINGS = { 
    "InstantX Flux Union ControlNet Loader": InstantXFluxUnionControlNetLoader,
    "ControlNetApplyAdvancedMasked": ControlNetApplyAdvancedMasked
}
