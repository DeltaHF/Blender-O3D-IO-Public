# ==============================================================================
#  Copyright (c) 2022-2023 Thomas Mathieson.
# ==============================================================================

import os
import time

import numpy as np

import bmesh
import bpy
from mathutils import Matrix
from . import o3d_cfg_parser, o3dconvert

if not (bpy.app.version[0] < 3 and bpy.app.version[1] < 80):
    from bpy_extras import node_shader_utils


def log(*args):
    print("[O3D_Export]", *args)


def extract_mesh_data(context, blender_obj, mesh, materials, export_custom_normals):
    """
    Extracts mesh data from a Blender object into O3D-compatible arrays.
    :param context: blender context
    :param blender_obj: the blender object
    :param mesh: the mesh data to extract
    :param materials: the material slots of the object
    :param export_custom_normals: whether to export custom split normals
    :return: (verts, tris, o3d_mats, bones)
    """
    has_uvs = len(mesh.uv_layers) > 0
    if has_uvs:
        uv_layer = mesh.uv_layers.active.data[:]
    else:
        uv_layer = None

    # Extract mesh data
    tris = []
    verts = []  # Array of (xp, yp, zp, xn, yn, zn, u, v)
    vert_map = {}
    vert_count = 0

    if bpy.app.version < (2, 80):
        mesh.calc_normals_split()
        mesh.calc_tessface()
        uv_layer = mesh.tessface_uv_textures.active
        for face in mesh.tessfaces:
            face_inds = []

            face_len = len(face.vertices)
            for i in range(face_len):
                v_co = mesh.vertices[face.vertices[i]].co[:]
                v_nrm = face.split_normals[i][:]
                if uv_layer is not None:
                    v_uv = uv_layer.data[face.index].uv[i][:]
                else:
                    v_uv = (0, 0)

                if (v_co, v_nrm, v_uv) in vert_map:
                    face_inds.append(vert_map[(v_co, v_nrm, v_uv)])
                else:
                    vert_map[(v_co, v_nrm, v_uv)] = vert_count
                    verts.append(
                        [v_co[0], v_co[1], v_co[2],
                         v_nrm[0], v_nrm[1], v_nrm[2],
                         v_uv[0], 1 - v_uv[1]])

                    face_inds.append(vert_count)

                    vert_count += 1

            # Create the triangle
            if face_len >= 3:
                tris.append((face_inds[0], face_inds[1], face_inds[2], face.material_index))

            # Sometimes we have to deal with quads...
            # 2---3
            # | \ |
            # 0---1
            if face_len >= 4:
                tris.append((face_inds[1], face_inds[3], face_inds[2], face.material_index))
    else:
        mesh.calc_loop_triangles()
        if export_custom_normals and mesh.has_custom_normals:
            # mesh.polygons.foreach_set("use_smooth", [False] * len(mesh.polygons))
            mesh.use_auto_smooth = True
        else:
            mesh.free_normals_split()
        mesh.calc_normals_split()
        for tri_loop in mesh.loop_triangles:
            tri = []
            tris.append(tri)

            for tri_vert, loop, normal in zip(tri_loop.vertices, tri_loop.loops, tri_loop.split_normals):
                vert = mesh.vertices[tri_vert]
                v_co = vert.co[:]
                v_nrm = mesh.loops[loop].normal[:]
                if uv_layer is not None:
                    v_uv = uv_layer[loop].uv[:2]
                else:
                    v_uv = (0, 0)

                if (v_co, v_nrm, v_uv) in vert_map:
                    tri.append(vert_map[(v_co, v_nrm, v_uv)])
                else:
                    vert_map[(v_co, v_nrm, v_uv)] = vert_count
                    verts.append(
                        [v_co[0], v_co[1], v_co[2],
                         -v_nrm[0], -v_nrm[1], -v_nrm[2],
                         v_uv[0], 1 - v_uv[1]])
                    tri.append(vert_count)
                    vert_count += 1

            tri.append(tri_loop.material_index)

    # Construct embedded material array
    o3d_mats = []
    tex_found_count = 0
    tex_missing_count = 0
    for mat in materials:
        # O3D mat structure:
        # (diffuse_r, diffuse_g, diffuse_b, diffuse_a, specular_r, specular_g, specular_b, emission_r, emission_g,
        #  emission_b, specular_power, texture_name)
        o3d_mat = []
        o3d_mats.append(o3d_mat)
        mat_name = mat.name if hasattr(mat, 'name') else "?"
        if bpy.app.version < (2, 80):
            mat = mat.material
            o3d_mat.extend(mat.diffuse_color)
            o3d_mat.append(mat.alpha)
            o3d_mat.extend(np.array(mat.specular_color) * mat.specular_intensity)
            o3d_mat.extend(np.array(mat.diffuse_color) * mat.emit)
            o3d_mat.append(mat.specular_hardness)
            texture_data = ""
            for i, texture in reversed(list(enumerate(mat.texture_slots))):
                if texture is None or texture.texture_coords != 'UV' or not texture.use_map_color_diffuse:
                    continue
                texture_data = bpy.data.textures[texture.name]
                if texture_data.type != 'IMAGE' or texture_data.image is None or not mat.use_textures[i]:
                    continue
                texture_data = texture_data.image.name

            o3d_mat.append(texture_data)
            if texture_data:
                tex_found_count += 1
                log("  Material [{0}]: texture '{1}'".format(mat_name, texture_data))
            else:
                tex_missing_count += 1
                log("  Material [{0}]: no texture found".format(mat_name))
        else:
            mat = node_shader_utils.PrincipledBSDFWrapper(mat.material, is_readonly=True)
            o3d_mat.extend(mat.base_color[:3])
            o3d_mat.append(mat.alpha)
            o3d_mat.extend([mat.specular, mat.specular, mat.specular])
            o3d_mat.extend(mat.emission_color[:3])
            o3d_mat.append(1 - mat.roughness)
            if mat.base_color_texture is not None and mat.base_color_texture.image is not None:
                tex_path = os.path.basename(mat.base_color_texture.image.filepath)
                o3d_mat.append(tex_path)
                tex_found_count += 1
                log("  Material [{0}]: texture '{1}'".format(mat_name, tex_path))
            else:
                o3d_mat.append("")
                tex_missing_count += 1
                log("  Material [{0}]: no texture found".format(mat_name))

    # Construct bones
    bones = []
    for v_group in blender_obj.vertex_groups:
        bone = (v_group.name, [])
        bones.append(bone)
        for index in range(len(verts)):
            try:
                bone[1].append((index, v_group.weight(index)))
            except Exception as e:
                pass

    log("  Extracted: {0} verts, {1} tris, {2} mats ({3} with tex, {4} without), {5} bones".format(
        len(verts), len(tris), len(o3d_mats), tex_found_count, tex_missing_count, len(bones)))

    return verts, tris, o3d_mats, bones


def export_mesh(filepath, context, blender_obj, mesh, transform_matrix, materials, o3d_version, export_custom_normals):
    # Create o3d file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        verts, tris, o3d_mats, bones = extract_mesh_data(
            context, blender_obj, mesh, materials, export_custom_normals)

        o3dconvert.export_o3d(f, verts, tris, o3d_mats, bones, transform_matrix,
                              version=o3d_version,
                              encrypted=False, encryption_key=0x0,
                              long_triangle_indices=False,
                              alt_encryption_seed=True,
                              invert_triangle_winding=True)

    log("  => Saved: {0} ({1} verts, {2} tris)".format(filepath, len(verts), len(tris)))


def merge_mesh_data(mesh_data_list):
    """
    Merges multiple sets of extracted mesh data into a single set.
    Handles vertex index offsetting, material index offsetting, and bone merging.

    :param mesh_data_list: list of (verts, tris, o3d_mats, bones) tuples
    :return: (merged_verts, merged_tris, merged_mats, merged_bones)
    """
    merged_verts = []
    merged_tris = []
    merged_mats = []
    merged_bones_dict = {}  # bone_name -> [(vertex_index, weight), ...]

    vertex_offset = 0
    material_offset = 0

    for verts, tris, mats, bones in mesh_data_list:
        # Append vertices directly
        merged_verts.extend(verts)

        # Offset triangle vertex indices and material indices
        for tri in tris:
            merged_tris.append((
                tri[0] + vertex_offset,
                tri[1] + vertex_offset,
                tri[2] + vertex_offset,
                tri[3] + material_offset
            ))

        # Append materials
        merged_mats.extend(mats)

        # Merge bones: combine weight lists for same-named bones
        for bone_name, weights in bones:
            adjusted_weights = [(idx + vertex_offset, w) for idx, w in weights]
            if bone_name in merged_bones_dict:
                merged_bones_dict[bone_name].extend(adjusted_weights)
            else:
                merged_bones_dict[bone_name] = adjusted_weights

        vertex_offset += len(verts)
        material_offset += len(mats)

    # Convert bones dict back to list of tuples
    merged_bones = [(name, weights) for name, weights in merged_bones_dict.items()]

    return merged_verts, merged_tris, merged_mats, merged_bones


def do_export(filepath, context, global_matrix, use_selection, o3d_version, export_custom_normals=True, merge_export=False):
    """
    Exports the selected CFG/SCO/O3D file
    :param o3d_version: O3D version to export the file as
    :param use_selection: export only the selected objects
    :param global_matrix: transformation matrix to apply before export
    :param filepath: the path to the file to import
    :param context: blender context
    :param merge_export: merge all objects into a single .o3d file
    :return: success message
    """
    obj_root = os.path.dirname(filepath)
    start_time = time.time()

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    if use_selection:
        obs = context.selected_objects
    else:
        obs = context.scene.objects

    if bpy.app.version < (2, 80):
        deps_graph = None
    else:
        deps_graph = context.evaluated_depsgraph_get()

    bpy.context.window_manager.progress_begin(0, len(obs))

    single_o3d = False
    if filepath[-3:] == "o3d":
        single_o3d = True

    log("========================================")
    log("Export started: {0}".format(filepath))
    log("  Objects: {0}, Mode: {1}".format(len(obs), "MERGE" if merge_export and single_o3d else "INDIVIDUAL"))
    log("  O3D Version: {0}, Custom Normals: {1}".format(o3d_version, export_custom_normals))
    log("========================================")

    # Collect mesh data for merge export
    mesh_data_list = []

    index = 0
    exported_paths = set()
    for ob in obs:
        if "skip_export" in ob:
            log("Skipping {0}...".format(ob.name))
            index += 1
            continue

        log("Exporting " + ob.name + "...")
        bpy.context.window_manager.progress_update(index)
        if bpy.app.version < (2, 80):
            ob_eval = ob

            try:
                me = ob_eval.to_mesh(context.scene, True, 'PREVIEW')
            except RuntimeError:
                continue
        else:
            ob_eval = ob.evaluated_get(deps_graph)

            try:
                me = ob_eval.to_mesh()
            except RuntimeError:
                continue

        axis_conversion_matrix = Matrix((
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        ))

        if bpy.app.version < (2, 80):
            o3d_matrix = axis_conversion_matrix * ob.matrix_world * axis_conversion_matrix
        else:
            o3d_matrix = axis_conversion_matrix @ ob.matrix_world @ axis_conversion_matrix
        o3d_matrix.transpose()
        me.transform(ob.matrix_world)
        me.transform(axis_conversion_matrix)
        if ob.matrix_world.is_negative:
            me.flip_normals()

        log("Exported matrix: \n{0}".format(o3d_matrix))

        bm = bmesh.new()
        bm.from_mesh(me)

        if global_matrix is not None:
            bm.transform(global_matrix)

        if bpy.app.version[0] < 3 and bpy.app.version[1] < 80:
            bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)
        else:
            bmesh.ops.triangulate(bm, faces=bm.faces)

        bm.to_mesh(me)
        bm.free()

        me.calc_normals_split()

        if merge_export and single_o3d:
            # Merge mode: extract data and collect for later merging
            data = extract_mesh_data(context, ob_eval, me, ob_eval.material_slots, export_custom_normals)
            mesh_data_list.append(data)
        else:
            # Export individual model
            if "export_path" in ob:
                path = os.path.join(obj_root, ob["export_path"])
            else:
                path = os.path.join(obj_root, ob.name + ".o3d")

            if single_o3d:
                if len(obs) == 1:
                    path = filepath
                else:
                    path = os.path.join(obj_root, os.path.basename(filepath)[:-4] + "-" + ob.name + ".o3d")

            # Export the mesh if it hasn't already been exported
            if path not in exported_paths:
                exported_paths.add(path)
                export_mesh(path, context, ob_eval, me, [x for y in o3d_matrix for x in y], ob_eval.material_slots,
                            o3d_version, export_custom_normals)

        index += 1

    # Write merged .o3d file
    if merge_export and single_o3d and len(mesh_data_list) > 0:
        merged_verts, merged_tris, merged_mats, merged_bones = merge_mesh_data(mesh_data_list)
        long_indices = len(merged_verts) > 65535

        tex_with = sum(1 for m in merged_mats if m[-1])
        tex_without = sum(1 for m in merged_mats if not m[-1])

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        identity_transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        with open(filepath, "wb") as f:
            o3dconvert.export_o3d(f, merged_verts, merged_tris, merged_mats, merged_bones, identity_transform,
                                  version=o3d_version,
                                  encrypted=False, encryption_key=0x0,
                                  long_triangle_indices=long_indices,
                                  alt_encryption_seed=True,
                                  invert_triangle_winding=True)
        log("----------------------------------------")
        log("MERGE RESULT:")
        log("  Merged {0} objects => {1}".format(len(mesh_data_list), filepath))
        log("  Vertices: {0}, Triangles: {1}".format(len(merged_verts), len(merged_tris)))
        log("  Materials: {0} ({1} with tex, {2} without)".format(
            len(merged_mats), tex_with, tex_without))
        if tex_with > 0:
            for i, m in enumerate(merged_mats):
                if m[-1]:
                    log("    Mat[{0}]: '{1}'".format(i, m[-1]))
        log("  Bones: {0}, Long indices: {1}".format(len(merged_bones), long_indices))
        log("----------------------------------------")

    if not single_o3d:
        cfg_materials = o3d_cfg_parser.write_cfg(filepath, obs, context, use_selection)

    bpy.context.window_manager.progress_end()
    log("Export finished: {0} objects processed in {1:.2f}s".format(len(obs), time.time() - start_time))
