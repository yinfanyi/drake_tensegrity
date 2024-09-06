# Import some basic libraries and functions for this tutorial.
import numpy as np
import os

from pydrake.common import temp_directory
from pydrake.geometry import SceneGraphConfig, StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

import pydrake.geometry
import pydrake.math
import pydrake.multibody.plant
import pydrake.multibody.tree
import pydrake.systems.framework
import pydrake.systems.primitives
from pydrake.math import RigidTransform as RigidTransform, RollPitchYaw as RollPitchYaw
from typing import Any, Callable, ClassVar, overload

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ContactVisualizer,
    DiagramBuilder,
    ExternallyAppliedSpatialForce,
    LeafSystem,
    List,
    MeshcatVisualizer,
    VectorLogSink,
    ModelVisualizer,
    Parser,
    Simulator,
    SpatialForce,
    StartMeshcat,
    Value,
)

from manipulation import ConfigureParser, FindResource, running_as_notebook

# Define a simple cylinder model.

def create_bar_sdf(mass=1.0,
                   pose=[0, 0, 0, 0, 0, 0],
                   inertia=[0.005833, 0.0, 0.0, 0.005833, 0.0, 0.005], 
                   radius=0.1, 
                   length=0.2, 
                   color=[1.0, 1.0, 1.0, 1.0],
                   name="bar_link"):
  cylinder_sdf = f"""<?xml version="1.0"?>
  <sdf version="1.7">
    <model name="cylinder">
      <pose>{' '.join(map(str, pose))}</pose>
      <link name="{name}">
        <inertial>
          <mass>{mass}</mass>
          <inertia>
            <ixx>{inertia[0]}</ixx>
            <ixy>{inertia[1]}</ixy>
            <ixz>{inertia[2]}</ixz>
            <iyy>{inertia[3]}</iyy>
            <iyz>{inertia[4]}</iyz>
            <izz>{inertia[5]}</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>{radius}</radius>
              <length>{length}</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>{radius}</radius>
              <length>{length}</length>
            </cylinder>
          </geometry>
          <material>
            <diffuse>{' '.join(map(str, color))}</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </sdf>
  """
  return cylinder_sdf

def create_floor_sdf():
    floor_sdf = """<?xml version="1.0"?>
    <sdf version="1.7">
    <model name="table_top">
        <link name="table_top_link">
        <visual name="visual">
            <pose>0 0 0.445 0 0 0</pose>
            <geometry>
            <box>
                <size>0.55 1.1 0.05</size>
            </box>
            </geometry>
            <material>
            <diffuse>0.9 0.8 0.7 1.0</diffuse>
            </material>
        </visual>
        <collision name="collision">
            <pose>0 0 0.445  0 0 0</pose>
            <geometry>
            <box>
                <size>0.55 1.1 0.05</size>
            </box>
            </geometry>
        </collision>
        </link>
        <frame name="table_top_center">
        <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
        </frame>
    </model>
    </sdf>
    """
    return floor_sdf

def initialize_simulation(diagram):
    simulator = initialize_simulation(diagram)
    simulator.set_target_realtime_rate(1.0)
    return simulator

def run_simulation(diagram, meshcat, finish_time=5):
    simulator = initialize_simulation(diagram)
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    meshcat.StartRecording()
    simulator.AdvanceTo(finish_time)
    meshcat.PublishRecording()

    return simulator