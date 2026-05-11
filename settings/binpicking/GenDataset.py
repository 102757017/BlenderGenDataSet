import blenderproc as bproc  # <--- 依然保持在第一行
import sys
import os

import glob

import math
import bpy
import argparse
import os.path as osp
import numpy as np
from typing import List
from loguru import logger
from functools import partial
from omegaconf import OmegaConf

def load_object(object_cfg):
    object = bproc.loader.load_obj(object_cfg.FILE)[0]
    unit = object_cfg.UNIT
    if unit == 'auto':
        bound_box = object.get_bound_box(local_coords=True)
        diag_length = np.linalg.norm(np.max(bound_box, axis=0) - np.min(bound_box, axis=0))
        unit = 'm' if diag_length < 2 else 'mm'
        logger.info('Object diagonal length: {:.4f} {}', diag_length, unit)

    if unit == 'mm':
        object.set_scale([0.001, 0.001, 0.001])
        
    if object_cfg.DECIMATION > 0:
        num_polygons = len(object.get_mesh().polygons.values())
        if object_cfg.DECIMATION < num_polygons:
            object.edit_mode()
            bpy.ops.mesh.decimate(ratio=object_cfg.DECIMATION / num_polygons)
            bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
            object.object_mode()

    if object_cfg.MOVE_TO_MASS_CENTER:
        object.set_origin(mode='CENTER_OF_MASS')
        object.set_location([0,0,0])

    if object_cfg.SHADING_MODE in['auto', 'flat', 'smooth']:
        object.set_shading_mode(object_cfg.SHADING_MODE)
    return object

def compute_tote_size(object: bproc.types.MeshObject, tote_cfg):
    object_bbox = object.get_bound_box()
    object_length = float(np.linalg.norm(np.max(object_bbox, axis=0) - np.min(object_bbox, axis=0)))
    if tote_cfg.WIDTH == 'auto': tote_cfg.WIDTH = 4 * object_length
    if tote_cfg.LENGTH == 'auto': tote_cfg.LENGTH = 4 * object_length
    if tote_cfg.HEIGHT == 'auto': tote_cfg.HEIGHT = 4 * object_length
    
    tote_xy_size = np.array([tote_cfg.WIDTH, tote_cfg.LENGTH])
    if np.linalg.norm(tote_xy_size) < 2 * object_length:
        tote_xy_size *= 2 * object_length / np.linalg.norm(tote_xy_size)
        tote_cfg.WIDTH, tote_cfg.LENGTH = float(tote_xy_size[0]), float(tote_xy_size[1])

def create_tote_planes(tote_cfg):
    width, length, height = tote_cfg.WIDTH, tote_cfg.LENGTH, tote_cfg.HEIGHT
    bottom_thickness = 0.01  # 1cm厚度，避免穿透
    
    # 使用CUBE替代PLANE作为箱底，提供明确的碰撞边界
    tote_bottom = bproc.object.create_primitive('CUBE',
                                                scale=[width*2, length*2, bottom_thickness],
                                                location=[0, 0, -bottom_thickness/2])
    
    # 箱壁保持PLANE
    tote_walls = [
        bproc.object.create_primitive('PLANE', scale=[width/2, height/2, 1], location=[0, -length/2, height/2], rotation=[-math.pi/2, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[width/2, height/2, 1], location=[0, length/2, height/2], rotation=[math.pi/2, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[height/2, length/2, 1], location=[-width/2, 0, height/2], rotation=[0, math.pi/2, 0]),
        bproc.object.create_primitive('PLANE', scale=[height/2, length/2, 1], location=[width/2, 0, height/2], rotation=[0, -math.pi/2, 0]),
    ]
    
    tote_planes = [tote_bottom] + tote_walls
    
    for i, plane in enumerate(tote_planes):
        if i == 0:
            # 箱底：静态刚体(mass=0)，高摩擦力防止滑动
            plane.enable_rigidbody(False, collision_shape='BOX', mass=0.0, 
                                  friction=150.0, linear_damping=1.0, angular_damping=1.0)
        else:
            plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, 
                                  friction=50.0, linear_damping=0.99, angular_damping=0.99)
        if i != 0 and not tote_cfg.WALL_VISIBLE: 
            plane.hide()

    funnel_angle = math.pi/4
    funnel_plane_coords = [[0, -length/2, height],[0, length/2, height], [-width/2, 0, height],[width/2, 0, height]]
    funnel_plane_rotation = [[-funnel_angle, 0, 0], [funnel_angle, 0, 0], [0, funnel_angle, 0],[0, -funnel_angle, 0]]
    
    funnel_planes =[bproc.object.create_primitive('PLANE', scale=[10, 10, 1], location=c, rotation=r) for c, r in zip(funnel_plane_coords, funnel_plane_rotation)]
    for i, plane in enumerate(funnel_planes):
        plane.edit_mode()
        bpy.ops.mesh.bisect(plane_co=funnel_plane_coords[i], plane_no=[0,0,1], clear_inner=True)
        plane.object_mode()
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=30.0, linear_damping=0.5, angular_damping=0.5)
        plane.hide()
    return tote_planes, funnel_planes

def sample_objects(base_obj, objects_to_check_collisions, sampler_cfg, tote_cfg):
    # 1. 设置下落高度范围（适当拉高 Z 轴，让物体呈柱状分布，减少重叠压力）
    drop_height_min = tote_cfg.HEIGHT + 0.05
    drop_height_max = drop_height_min + 0.8  # 40个部品建议给 80cm 的垂直空间
    
    # 2. 计算水平采样范围
    safe_margin = 0.03
    x_min, x_max = -tote_cfg.WIDTH/2 + safe_margin, tote_cfg.WIDTH/2 - safe_margin
    y_min, y_max = -tote_cfg.LENGTH/2 + safe_margin, tote_cfg.LENGTH/2 - safe_margin

    objects = []
    num_objects = np.random.randint(sampler_cfg.MIN_NUM, sampler_cfg.MAX_NUM + 1)
    
    logger.info(f"Directly scattering {num_objects} objects without collision check...")

    for i in range(num_objects):
        obj = base_obj.duplicate()
        obj.hide(False)
        
        # 启用刚体
        obj.enable_rigidbody(True, mass=1.0, friction=0.8, 
                            linear_damping=0.5, angular_damping=0.5,
                            collision_margin=0.001,
                            collision_shape='CONVEX_HULL')
        
        # --- 直接手动设置位置和角度，跳过 BlenderProc 的 sample_poses 检查 ---
        random_loc = [
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(drop_height_min, drop_height_max)
        ]
        obj.set_location(random_loc)
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        # ------------------------------------------------------------------
        
        objects.append(obj)

    # 执行物理模拟
    if sampler_cfg.PHYSICS.ENABLE:
        logger.info("Starting physics simulation to resolve overlaps...")
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3.0,
            max_simulation_time=6.0,
            substeps_per_frame=20, 
            solver_iters=25,
            check_object_interval=1
        )
    
    # 检查是否有物体飞出箱子（可选）
    final_objects = []
    for obj in objects:
        loc = obj.get_location()
        # 如果物体掉落后高度太低（穿透箱底）或者水平飞出太远，则剔除
        if loc[2] > -0.05 and abs(loc[0]) < tote_cfg.WIDTH and abs(loc[1]) < tote_cfg.LENGTH:
            final_objects.append(obj)
        else:
            obj.delete()
            
    return final_objects

def sample_objects_material(objects, sampler_cfg):
    for obj in objects:        
        mat = obj.get_materials()[0]
        grey = np.random.uniform(sampler_cfg.MIN_GRAY, sampler_cfg.MAX_GRAY)
        mat.set_principled_shader_value("Base Color", [grey, grey, grey, 1])      
        mat.set_principled_shader_value("Roughness", np.random.uniform(sampler_cfg.MIN_ROUGHNESS, sampler_cfg.MAX_ROUGHNESS))
        try:
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(sampler_cfg.MIN_SPECULAR, sampler_cfg.MAX_SPECULAR))
        except KeyError:
            mat.set_principled_shader_value("Specular", np.random.uniform(sampler_cfg.MIN_SPECULAR, sampler_cfg.MAX_SPECULAR))
        mat.set_principled_shader_value("Metallic", np.random.uniform(sampler_cfg.MIN_METALLIC, sampler_cfg.MAX_METALLIC))

def sample_lights(light_cfg) -> List[bproc.types.Light]:
    lights =[]
    num_lights = np.random.randint(light_cfg.MIN_NUM, light_cfg.MAX_NUM + 1)
    for _ in range(num_lights):
        light = bproc.types.Light(light_type="POINT")
        light.set_energy(np.random.uniform(light_cfg.MIN_ENERGY, light_cfg.MAX_ENERGY))
        location = bproc.sampler.shell(center=[0,0,0], radius_min=light_cfg.MIN_HEIGHT, radius_max=light_cfg.MAX_HEIGHT,
                                       elevation_min=light_cfg.MIN_ELEVATION, elevation_max=light_cfg.MAX_ELEVATION)
        light.set_color(np.random.uniform([0.6, 0.6, 0.65],[0.8, 0.8, 0.85]))
        light.set_location(location)
        lights.append(light)
    return lights

def set_camera_intrinsics(camera_cfg):
    bproc.camera.set_intrinsics_from_K_matrix([[camera_cfg.FX, 0, camera_cfg.PPX], [0, camera_cfg.FY, camera_cfg.PPY],[0, 0, 1]], 
                                              camera_cfg.WIDTH, camera_cfg.HEIGHT, 0.05, 10)

# 注意：增加了 frame 参数
def sample_camera(objects, sampler_cfg, tote_cfg, frame=0):
    tote_center =[0, 0, tote_cfg.HEIGHT/2]
    location = bproc.sampler.shell(
        center=tote_center, radius_min=sampler_cfg.MIN_HEIGHT, radius_max=sampler_cfg.MAX_HEIGHT,
        elevation_min=sampler_cfg.MIN_ELEVATION, elevation_max=sampler_cfg.MAX_ELEVATION,
        azimuth_min=sampler_cfg.MIN_AZIMUTH, azimuth_max=sampler_cfg.MAX_AZIMUTH  
    )
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array(tote_center) - location, inplane_rot=0)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    # 将相机注册到正确的帧序列
    bproc.camera.add_camera_pose(cam2world_matrix, frame=frame)
    return cam2world_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for object detection and segmentation.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    bproc.init()
    bpy.context.scene.render.engine = 'CYCLES'
    #bpy.context.scene.cycles.device = 'GPU'
    bproc.renderer.set_max_amount_of_samples(10)
    bproc.renderer.set_output_format(file_format="PNG", color_depth=8)
    
    if cfg.ENABLE_DEPTH:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        
    # [必须添加] 启用 BOP 需要的分割图输出依据
    bproc.renderer.enable_segmentation_output(map_by=["class", "instance", "name"])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 预先在循环外加载一次材质库，避免变量报错和重复加载性能损耗
    materials = None
    if cfg.TOTE.RANDOM_TEXTURE and osp.exists(cfg.CC_TEXTURES_DIR):
        materials = bproc.loader.load_ccmaterials(cfg.CC_TEXTURES_DIR)

    for scene_id in range(cfg.NUM_SCENES):
        logger.info('Generating scene {}', scene_id)
        
        try:
            obj_id = int(os.path.basename(cfg.OBJECT.FILE).split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            obj_id = 1
        
        base_obj = load_object(cfg.OBJECT)
        compute_tote_size(base_obj, cfg.TOTE)
        tote_planes, funnel_planes = create_tote_planes(cfg.TOTE)
        objects = sample_objects(base_obj, tote_planes + funnel_planes, cfg.OBJECT.POSE_SAMPLER, cfg.TOTE)

        # 推荐使用 bproc 官方 set_cp 写入 BOP 相关属性
        for i, obj in enumerate(objects):
            obj.set_cp("category_id", obj_id)

        lights = sample_lights(cfg.LIGHT)
        set_camera_intrinsics(cfg.CAMERA.INTRINSICS)
        
        # 将材质和颜色赋予移出相机循环（现实物理中场景建好后是不变的，只有相机在动）
        if materials is not None:
            mat = np.random.choice(materials)
            for plane in tote_planes:
                plane.replace_materials(mat)
        sample_objects_material(objects, cfg.OBJECT.MATERIAL_SAMPLER)

        # --- 核心修复 1：在循环中仅注册相机位姿序列 ---
        for camera_id in range(cfg.NUM_CAMERAS_PER_SCENE):
            logger.info('Scene {} / Camera {}', scene_id, camera_id)
            sample_camera(objects, cfg.CAMERA.POSE_SAMPLER, cfg.TOTE, frame=camera_id)

        # --- 核心修复 2：移出循环，一次性执行批渲染 ---
        data = bproc.renderer.render()

        logger.info('Writing BOP format dataset...')
        
        # --- 核心修复 3：append_to_existing_output 必须置 True ---
        bproc.writer.write_bop(
            output_dir=cfg.OUTPUT_DIR,
            target_objects=objects,
            depths=data.get("depth", None) if cfg.ENABLE_DEPTH else None,
            colors=data.get("colors", None),
            color_file_format="PNG",
            #数据集名称必须是固定的，不能在循环里加 scene_id,BOP writer 会在这个目录下自动生成 000000, 000001 的场景子文件夹
            dataset="deep_tote_dataset", 
            depth_scale=cfg.DEPTH_SCALE,
            append_to_existing_output=True,
            calc_mask_info_coco=False,  #开启掩码及可见性计算，生成 mask 与 mask_visib
        )


        
        # 彻底的清理，防止污染下一轮 scene
        bproc.clean_up(clean_up_camera=True)
        
        logger.info('BOP dataset scene_{:06d} written successfully!', scene_id)
