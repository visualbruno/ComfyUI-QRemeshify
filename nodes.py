from PIL import Image
from torch.utils.data import Dataset
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
from torchvision import transforms
import os
import time
import re
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from .lib import Quadwild, QWException
#from .util import bisect, exporter, importer
import trimesh as Trimesh

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_dir = os.path.join(comfy_path, "models", "shadow_r")

def get_filename_list(folder_name: str):
    files = [f for f in os.listdir(folder_name)]
    return files
    
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)    

def export_sharp_features(mesh, sharp_filepath: str, sharp_angle_degrees: float = 35):
    #mesh = Trimesh.load(mesh_filepath, process=False) # Avoid merging vertices initially if precise indices matter

    sharp_angle_rad = np.radians(sharp_angle_degrees)
    sharp_features = []

    # face_adjacency stores pairs of face indices that share an edge
    # face_adjacency_edges stores the vertex indices (start, end) of that shared edge
    # face_adjacency_angles stores the dihedral angle between the pair of faces
    adj_faces = mesh.face_adjacency
    adj_edges_vertices = mesh.face_adjacency_edges
    adj_angles = mesh.face_adjacency_angles

    for i, angle in enumerate(adj_angles):
        # Check if the edge is sharp based on the angle
        # Note: Blender's 'smooth' is often the inverse - smooth below an angle.
        # We adapt the concept: sharp *above* an angle, or perhaps *not smooth* below it.
        # A common definition of sharp is when the angle between normals is large,
        # meaning the dihedral angle deviates significantly from pi (180 deg).
        # Let's assume 'sharp' means the dihedral angle is LESS than (180 - sharp_angle)
        # or MORE than (180 + sharp_angle). More simply, the angle between normals > sharp_angle.
        # The angle between normals is pi - dihedral_angle.
        angle_between_normals = np.pi - angle
        if angle_between_normals > sharp_angle_rad:
             # Edge is considered sharp

             convexity = 1 if angle < np.pi else 0 # Dihedral angle < 180 deg is convex

             face_index = adj_faces[i][0] # Pick the first face like in the bmesh script
             edge_vertices = adj_edges_vertices[i] # Get the vertex indices for this edge

             # Find the local index of the edge within the chosen face
             face_verts = mesh.faces[face_index]
             edge_index = -1
             # Check pairs of vertices in the face definition
             if tuple(sorted(edge_vertices)) == tuple(sorted((face_verts[0], face_verts[1]))):
                 edge_index = 0
             elif tuple(sorted(edge_vertices)) == tuple(sorted((face_verts[1], face_verts[2]))):
                 edge_index = 1
             elif tuple(sorted(edge_vertices)) == tuple(sorted((face_verts[2], face_verts[0]))):
                 edge_index = 2
             else:
                 # This shouldn't happen if topology is correct
                 print(f"Warning: Edge {edge_vertices} not found in face {face_index} ({face_verts})")
                 continue # Skip if edge not found in face


             sharp_features.append(f"{convexity},{face_index},{edge_index}")

    # Write to file
    num_sharp_features = len(sharp_features)
    with open(sharp_filepath, 'w') as f:
        f.write(f"{num_sharp_features}\n")
        for feature in sharp_features:
            f.write(f"{feature}\n")

    return num_sharp_features

# Example usage:
# mesh_file = 'path/to/your/mesh.obj'
# output_file = 'path/to/sharp_features.txt'
# export_sharp_features_trimesh(mesh_file, output_file, sharp_angle_degrees=35)

class QRemeshify:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "enableSharp": ("BOOLEAN", {"default": False}),
                "sharpAngle": ("FLOAT", {"default":35.0}),
                "quad": ("BOOLEAN", {"default": False}),
                "enableSmoothing": ("BOOLEAN", {"default":False}),
                "scaleFactor" : ("FLOAT", {"default":1.0,"min":0.01,"max":10.0, "tooltip":"Values > 1 for larger quads, < 1 to preserve more detail"}),
                "fixedChartClusters": ("INT", {"default":0}),
                "alpha": ("FLOAT", {"default":0.005,"min":0.0,"max":0.999,"step":0.005,"tooltip":"Blends between isometry (alpha) and regularity (1-alpha)"}),
                "ilpMethod": (["LEASTSQUARES","ABS"],{"default":"LEASTSQUARES", "tooltip":"ILP method for solving the ILP problem : Least Squares or Absolute"}),
                "isometry": ("BOOLEAN",{"default":True}),
                "regularityQuadrilaterals": ("BOOLEAN", {"default":True}),
                "regularityNonQuadrilaterals": ("BOOLEAN", {"default":True}),
                "regularityNonQuadrilateralsWeight": ("FLOAT",{"default":0.9, "min":0.0,"max":100.0}),
                "alignSingularities": ("BOOLEAN",{"default":True}),
                "alignSingularitiesWeight": ("FLOAT",{"default":0.1,"min":0.0,"max":100.0}),
                "repeatLosingConstraintsIterations": ("BOOLEAN",{"default":True}),
                "repeatLosingConstraintsQuads": ("BOOLEAN",{"default":False}),
                "repeatLosingConstraintsNonQuads": ("BOOLEAN", {"default":False}),
                "repeatLosingConstraintsAlign": ("BOOLEAN", {"default":True}),
                "hardParityConstraint": ("BOOLEAN", {"default":True}),
                "flowConfig": (["SIMPLE","HALF"], {"default":"SIMPLE"}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "remesh"
    CATEGORY = "QRemeshifyWrapper"
    OUTPUT_NODE = True
    
    def remesh(self, mesh, enableSharp, sharpAngle, quad, enableSmoothing, scaleFactor, fixedChartClusters, alpha, ilpMethod, isometry, regularityQuadrilaterals, regularityNonQuadrilaterals, regularityNonQuadrilateralsWeight, alignSingularities, alignSingularitiesWeight,repeatLosingConstraintsIterations, repeatLosingConstraintsQuads, repeatLosingConstraintsNonQuads, repeatLosingConstraintsAlign, hardParityConstraint, flowConfig):
        device = mm.get_torch_device()
        
        temp_dir = folder_paths.get_temp_directory()
        mesh_filepath = os.path.join(temp_dir,'qremesh_temp.obj')
        mesh.export(mesh_filepath, file_type='obj')
        
        qw = Quadwild(mesh_filepath) 
        
        if enableSharp:
            num_sharp_features = export_sharp_features(mesh, qw.sharp_path, sharpAngle)
            print(f"Found {num_sharp_features} sharp edges")        

        enableRemesh = True
        qw.remeshAndField(remesh=enableRemesh, enableSharp=enableSharp, sharpAngle=sharpAngle)        
        
        if quad == True:
            qw.trace()
            timeLimit = 200
            gapLimit = 0.0
            minimumGap = 0.4
            satsumaConfig = 'DEFAULT'
            callbackTimeLimit = [3.00, 5.000, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0]
            callbackGapLimit = [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
            
            qw.quadrangulate(
                enableSmoothing,
                scaleFactor,
                fixedChartClusters,
                alpha,
                
                ilpMethod,
                timeLimit,
                gapLimit,
                minimumGap,
                isometry,
                regularityQuadrilaterals,
                regularityNonQuadrilaterals,
                regularityNonQuadrilateralsWeight,
                alignSingularities,
                alignSingularitiesWeight,
                repeatLosingConstraintsIterations,
                repeatLosingConstraintsQuads,
                repeatLosingConstraintsNonQuads,
                repeatLosingConstraintsAlign,
                hardParityConstraint,

                flowConfig,
                satsumaConfig,

                callbackTimeLimit,
                callbackGapLimit,
            )
            
            output_path = qw.output_smoothed_path if enableSmoothing else qw.output_path
        else:            
            output_path = qw.remeshed_path
        
        #output_path = qw.remeshed_path
        output_mesh = Trimesh.load(output_path, force="mesh")
        
        return (output_mesh,)

NODE_CLASS_MAPPINGS = {
    "QRemeshify": QRemeshify
    }
    
NODE_DISPLAY_NAME_MAPPINGS = {
    "QRemeshify": "QRemeshify"
    }