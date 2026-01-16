#!/usr/bin/env python3
"""
Diagnostic script to visualize what EasyHEC is seeing during optimization.
This will save images showing:
1. The actual segmentation mask (from SAM2)
2. The rendered robot mask (from initial extrinsic guess)
3. The overlay/difference
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, "simple-easyhec")

from urchin import URDF
from easyhec import ROBOT_DEFINITIONS_DIR
from easyhec.utils.utils_3d import merge_meshes
from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
from easyhec.utils import utils_3d
from transforms3d.euler import euler2mat
from easyhec.utils.camera_conversions import ros2opencv
import trimesh

def main():
    # Load data
    results_dir = Path("simple-easyhec/results/so100/stone_home")
    base_camera_dir = results_dir / "base_camera"

    link_poses = np.load(results_dir / "link_poses_dataset.npy")
    masks = np.load(base_camera_dir / "mask.npy")
    intrinsic = np.load(base_camera_dir / "camera_intrinsic.npy")
    images = np.load(results_dir / "image_dataset.npy", allow_pickle=True).reshape(-1)[0]
    
    camera_width = images['base_camera'].shape[2]
    camera_height = images['base_camera'].shape[1]
    
    print(f"Image size: {camera_width}x{camera_height}")
    print(f"Number of frames: {masks.shape[0]}")
    print(f"Link poses shape: {link_poses.shape}")

    # Load meshes
    robot_def_path = ROBOT_DEFINITIONS_DIR / "so100"
    robot_urdf = URDF.load(str(robot_def_path / "so100.urdf"))
    meshes = []
    for link in robot_urdf.links:
        link_meshes = []
        for visual in link.visuals:
            link_meshes += visual.geometry.mesh.meshes
        meshes.append(merge_meshes(link_meshes))

    # Initial extrinsic guess
    initial_extrinsic_guess = np.eye(4)
    initial_extrinsic_guess[:3, :3] = euler2mat(0, np.pi / 4, -np.pi / 5)
    initial_extrinsic_guess[:3, 3] = np.array([-0.4, 0.1, 0.5])
    initial_extrinsic_guess = ros2opencv(initial_extrinsic_guess)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup renderer
    renderer = NVDiffrastRenderer(camera_height, camera_width)
    intrinsic_t = torch.from_numpy(intrinsic).float().to(device)
    Tc_c2b = torch.from_numpy(initial_extrinsic_guess).float().to(device)

    # Prepare mesh data
    mesh_data = []
    for link_idx, mesh in enumerate(meshes):
        if mesh is not None:
            vertices = torch.from_numpy(mesh.vertices).float().to(device)
            faces = torch.from_numpy(mesh.faces).int().to(device)
            mesh_data.append((vertices, faces))
        else:
            mesh_data.append(None)

    link_poses_t = torch.from_numpy(link_poses).float().to(device)

    # Create figure for each frame
    for frame_idx in range(masks.shape[0]):
        print(f"\n--- Frame {frame_idx} ---")
        
        # Get actual mask
        actual_mask = masks[frame_idx]
        
        # Render robot mask
        all_link_si = []
        for link_idx in range(len(mesh_data)):
            if mesh_data[link_idx] is None:
                continue
            verts, faces = mesh_data[link_idx]
            Tc_c2l = Tc_c2b @ link_poses_t[frame_idx, link_idx]
            si = renderer.render_mask(verts, faces, intrinsic_t, Tc_c2l)
            all_link_si.append(si)
        
        if len(all_link_si) > 0:
            rendered_mask = torch.stack(all_link_si).sum(0).clamp(max=1)
        else:
            rendered_mask = torch.zeros(camera_height, camera_width, device=device)
        
        rendered_mask_np = rendered_mask.cpu().numpy()
        
        # Compute loss for this frame
        loss = np.sum((rendered_mask_np - actual_mask) ** 2)
        print(f"  Loss: {loss:.0f}")
        print(f"  Actual mask coverage: {actual_mask.sum() / actual_mask.size * 100:.1f}%")
        print(f"  Rendered mask coverage: {rendered_mask_np.sum() / rendered_mask_np.size * 100:.1f}%")
        
        # Get the real image
        real_image = images['base_camera'][frame_idx]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Masks
        axes[0, 0].imshow(actual_mask, cmap='gray')
        axes[0, 0].set_title(f'Actual Mask (SAM2)\nCoverage: {actual_mask.sum() / actual_mask.size * 100:.1f}%')
        
        axes[0, 1].imshow(rendered_mask_np, cmap='gray')
        axes[0, 1].set_title(f'Rendered Mask (Initial Guess)\nCoverage: {rendered_mask_np.sum() / rendered_mask_np.size * 100:.1f}%')
        
        # Difference map
        diff = np.abs(rendered_mask_np - actual_mask)
        axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title(f'Difference (Loss: {loss:.0f})')
        
        # Row 2: Real image with overlays
        axes[1, 0].imshow(real_image)
        axes[1, 0].set_title('Real Image')
        
        # Real image with actual mask overlay
        overlay1 = real_image.copy()
        overlay1[actual_mask > 0.5] = overlay1[actual_mask > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[1, 1].imshow(overlay1.astype(np.uint8))
        axes[1, 1].set_title('Real Image + Actual Mask (green)')
        
        # Real image with rendered mask overlay
        overlay2 = real_image.copy()
        overlay2[rendered_mask_np > 0.5] = overlay2[rendered_mask_np > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[1, 2].imshow(overlay2.astype(np.uint8))
        axes[1, 2].set_title('Real Image + Rendered Mask (red)')
        
        plt.tight_layout()
        output_path = f"calibration_diagnostic_frame_{frame_idx}.png"
        plt.savefig(output_path, dpi=150)
        print(f"  Saved: {output_path}")
        plt.close()

    print("\n=== DIAGNOSIS ===")
    print("If rendered mask (red) is in a COMPLETELY DIFFERENT location than actual mask (green):")
    print("  -> Initial extrinsic guess is too far off")
    print("  -> Or link poses are wrong (CALIBRATION_OFFSET issue)")
    print("")
    print("If rendered mask has similar location but wrong POSE/SHAPE:")
    print("  -> Link poses (joint angles) are wrong")
    print("  -> CALIBRATION_OFFSET needs tuning")
    print("")
    print("If masks look similar but still high loss:")
    print("  -> Try more iterations or different learning rate")

if __name__ == "__main__":
    main()
