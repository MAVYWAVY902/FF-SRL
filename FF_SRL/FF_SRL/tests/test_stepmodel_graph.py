#!/usr/bin/env python3
"""
Test if stepModelReduce can be captured in CUDA Graph
"""

import os
import sys
import warp as wp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import FF_SRL as dk
from pxr import Usd

print("Initializing...")
wp.init()
wp.config.verify_cuda = False

# Load model
stage = Usd.Stage.Open("../scenes/liverRetractionTexture.usd")
sim_model = dk.SimModelDO(
    stage,
    numEnvs=1,
    device="cuda:0",
    globalKsDrag=1.0,
    simFrameRate=60,
    simConstraintsSteps=1,
    simSubsteps=10
)

sim_integrator = dk.SimIntegratorDO("cuda:0")

print("\nTesting Graph capture of stepModelReduce...")
try:
    print("Attempting wp.capture_begin()...")
    wp.capture_begin()
    
    print("Calling stepModelReduce...")
    sim_integrator.stepModelReduce(sim_model)
    
    print("Attempting wp.capture_end()...")
    graph = wp.capture_end()
    
    print(f"✅ SUCCESS! Graph captured: {graph}")
    
    # Try to launch it
    print("\nTesting graph launch...")
    wp.capture_launch(graph)
    wp.synchronize()
    print("✅ Graph launch successful!")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
