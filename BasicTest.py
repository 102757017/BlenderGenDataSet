import blenderproc as bproc
import numpy as np
import os

if __name__ == "__main__":
    bproc.init()

    # 创建物体并设置 category_id
    obj = bproc.object.create_primitive("MONKEY")
    obj.set_cp("category_id", 1)

    # 添加光源
    light = bproc.types.Light()
    light.set_location([2, -2, 0])
    light.set_energy(300)

    # 设置相机位姿
    cam_pose = bproc.math.build_transformation_mat([0, -5, 0],[np.pi / 2, 0, 0])
    bproc.camera.add_camera_pose(cam_pose)

    # 1. 仅启用深度输出 (删除 enable_segmentation_output)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # 2. 第一步渲染：渲染 RGB 和 Depth
    data = bproc.renderer.render()

    # 3. 第二步渲染：单独渲染 Segmap (自带完整功能，生成 .npy 格式数据)
    seg_data = bproc.renderer.render_segmap(map_by=["class", "instance", "name"])
    data.update(seg_data)

    # 4. 写入 BOP 基础姿态和深度数据
    bproc.writer.write_bop(
        output_dir="output/",
        target_objects=[obj],
        depths=data["depth"],          
        colors=data["colors"],
        color_file_format="PNG",
        dataset="BOP",
        annotation_unit="m",
        calc_mask_info_coco=False      # 屏蔽导致 Windows 崩溃的 BOP 多进程 Pool
    )
