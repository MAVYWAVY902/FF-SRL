#!/usr/bin/env python3
"""
Simplified version of testInteractiveControl.py for profiling
Runs 100 frames with rendering every 2 frames
"""

import os
import sys
import time
import warp as wp
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FF_SRL as dk
from FF_SRL.tool_controller import ToolController

class ProfileSimulation:
    def __init__(self):
        print("Initializing simulation for profiling...")
        wp.init()
        wp.config.verify_cuda = False
        
        # USD Stage
        from pxr import Usd
        self.stage = Usd.Stage.Open("../scenes/liverRetractionTexture.usd")
        
        # Create model
        self.sim_model = dk.SimModelDO(
            self.stage,
            numEnvs=1,
            device="cuda:0",
            globalKsDrag=1.0,
            simFrameRate=60,
            simConstraintsSteps=1,
            simSubsteps=10,
            globalLaparoscopeDragLookupRadius=1.0,
            globalKsDistance=1.0,
            globalKsVolume=1.0
        )
        
        # BVH
        self.sim_bvh = dk.SimBVH(
            "cuda:0", self.sim_model,
            binsNumber=8, load=True, save=False,
            path="../scenes/liverRetractionTexture.npz"
        )
        
        # Renderer
        self.renderer = dk.render.WarpRaycastRendererDO(
            device="cuda:0",
            simModel=self.sim_model,
            resolution=512,
            cameraPos=[2.0, 22.0, 40.0],
            cameraRot=[0.0, 0.0, 0.0],
            lightPos=[0.0, 15.0, 80.0],
            mode="headless"
        )
        
        # Tool controller
        self.tool_controller = ToolController(self.sim_model, device="cuda:0")
        
        # Integrator (needed for physics stepping)
        self.sim_integrator = dk.SimIntegratorDO("cuda:0")
        
        # CUDA Graphs
        self.sim_graph = None
        self._render_counter = 0
        
        print("✅ Initialization complete\n")
    
    def step(self):
        """Physics step with tool control"""
        # Physics + Tool control in same Graph
        if self.sim_graph is None:
            wp.capture_begin()
            self.tool_controller.apply_actions_graph(num_envs=1)
            self.sim_integrator.stepModelReduce(self.sim_model)
            self.sim_graph = wp.capture_end()
        else:
            wp.capture_launch(self.sim_graph)
        
        # Reset actions (must be outside Graph)
        self.tool_controller.cartesian_actions.zero_()
    
    def render(self):
        """Render every 2 frames"""
        self._render_counter += 1
        if self._render_counter % 2 == 0:
            self.sim_bvh.refitBVH(useGraph=True)
            self.renderer.renderNew(self.sim_bvh, self.sim_model)
    
    def run_profile(self, num_frames=100):
        """Run fixed number of frames for profiling"""
        print(f"🚀 Starting profiling run ({num_frames} frames)...")
        print("=" * 60)
        
        start_time = time.time()
        
        for i in range(num_frames):
            frame_start = time.time()
            
            # Simulation step
            self.step()
            
            # Render (every 2 frames)
            self.render()
            
            # Print progress every 20 frames
            if (i + 1) % 20 == 0:
                elapsed = time.time() - frame_start
                print(f"Frame {i+1:3d}/{num_frames} | {elapsed*1000:.1f}ms")
        
        total_time = time.time() - start_time
        avg_fps = num_frames / total_time
        avg_ms = (total_time / num_frames) * 1000
        
        print("=" * 60)
        print(f"✅ Profiling complete!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average frame time: {avg_ms:.1f}ms")

if __name__ == "__main__":
    sim = ProfileSimulation()
    sim.run_profile(num_frames=100)
    print("\nProfile data saved. Analyze with: nsys stats report.nsys-rep")
