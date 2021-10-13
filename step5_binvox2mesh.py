# -*- coding: utf-8 -*-
import numpy as np
import trimesh # for converting voxel grids to meshes (to import objects into simulators)
import time # to know how long it takes for the code to run
import os # to walk through directories, to rename files
import sys
import glob

import binvox_rw # to manipulate binvox files
#import subdividing
#import dualcontour

# Parses a file of type BINVOX
# Returns a voxel grid, generated using the binvox_rw.py package
def parse_BINVOX_file_into_voxel_grid(filename):
    filereader = open(filename, 'rb')
    binvox_model = binvox_rw.read_as_3d_array(filereader)
    voxelgrid = binvox_model.data
    dim = binvox_model.dims
    scale = binvox_model.scale
    print(dim)
    print(scale)
    return voxelgrid, dim, scale

#def subdividing(vertices, faces):
#    
#    
num = str(11)

def main():
    
#    filename = sys.argv[1]
    filenames = glob.glob('../../result/run' + num + '/*.binvox')

    for filename in filenames:
        midname = filename[filename.rindex("\\")+1:filename.rindex(".")]
        print(midname)
        # Load the voxelgrid from file
        voxelgrid, dim, scale = parse_BINVOX_file_into_voxel_grid(filename)
               
       # Generate a folder to store the images
#       print("Generating a folder to save the mesh")
        directory = "../../result/run" + num + "/" + midname + '_' + str(time.time())[:10]
        if not os.path.exists(directory):
            os.makedirs(directory)

    
######### smoothing / marching cubes
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(matrix=voxelgrid, pitch=1.0)
#            origin=(0,0,0))


######### subdividing       
#        mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)


######### smoothing
#        mesh = trimesh.smoothing.filter_laplacian(mesh,lamb = 0.25, iterations=10)
        mesh = trimesh.smoothing.filter_humphrey(mesh,iterations=10)
#        mesh = trimesh.smoothing.filter_taubin(mesh) # 잘 안 됨??


        print("Merging vertices closer than a pre-set constant...")
        mesh.merge_vertices()
        print("Removing duplicate faces...")
        mesh.remove_duplicate_faces()

        print("Scaling...")
        mesh.apply_scale(scaling=scale/dim[0]) # scale/dim -> fit to original dimension,,
#        mesh.apply_scale(scaling=1.0)
        
        print("Making the mesh watertight...")
        trimesh.repair.fill_holes(mesh)
        
        print("Fixing inversion and winding...")
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
    
        print("Generating the STL mesh file")
        trimesh.exchange.export.export_mesh(
            mesh=mesh,
            file_obj=directory + "/" + midname + ".stl",
            file_type="stl"
        )


###############################################################################
if __name__ == "__main__":
    main()