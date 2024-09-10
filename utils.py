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

class Spring(LeafSystem):
    def __init__(self, plant, stiffness=100, damping=10, original_length=0.5):
        LeafSystem.__init__(self)
        forces_cls = Value[List[ExternallyAppliedSpatialForce]]
        print(ExternallyAppliedSpatialForce)
        self.DeclareAbstractOutputPort(
            "applied_force", lambda: forces_cls(), self.CalcOutput
        )
        self.DeclareVectorInputPort(name="bar_1_state", size=13)
        self.DeclareVectorInputPort(name="bar_2_state", size=13)
        self.plant = plant
        self.stiffness = stiffness
        self.damping = damping
        self.original_length = original_length

    def CalcOutput(self, context, output):
        bar_1_state = self.get_input_port(0).Eval(context)
        bar_2_state = self.get_input_port(1).Eval(context)

        distance = np.linalg.norm(bar_1_state[4:7] - bar_2_state[4:7])
        u = (bar_1_state[4:7] - bar_2_state[4:7])/distance

        delta_length = self.original_length - distance
        delta_velocity = np.array([bar_1_state[10]- bar_2_state[10], bar_1_state[11]- bar_2_state[11], bar_1_state[12]- bar_2_state[12]])
        
        F = self.stiffness * delta_length - self.damping * delta_velocity

        forces = []

        force = ExternallyAppliedSpatialForce()
        force.body_index = self.plant.GetBodyByName(f"bar_1_link").index()
        force.p_BoBq_B = np.array([0,0,0.1])  # world 0, 0, 0
        # force.p_BoBq_B = cylinder.CalcCenterOfMassInBodyFrame(plant_context)    # pos in body frame 0 0 0
        # print(force.p_BoBq_B)
        force.F_Bq_W = SpatialForce(
            tau=np.array([0,0,1]),
            f=F*u,
        )
        forces.append(force)

        force2 = ExternallyAppliedSpatialForce()
        force2.body_index = self.plant.GetBodyByName(f"bar_2_link").index()
        force2.p_BoBq_B = np.array([0,0,0])  # world 0, 0, 0
        # force.p_BoBq_B = cylinder.CalcCenterOfMassInBodyFrame(plant_context)    # pos in body frame 0 0 0
        # print(force.p_BoBq_B)

        force2.F_Bq_W = SpatialForce(
            tau=np.array([0,0,0]),
            f=-F*u
        )
        forces.append(force2)

        output.set_value(forces)

def create_bar_sdf1(mass=1.0,
                   pose=[0, 0, 0, 0, 0, 0],
                   inertia=[0.005833, 0.0, 0.0, 0.005833, 0.0, 0.005], 
                   radius=0.1, 
                   ball_radius=0.01,
                   length=0.2, 
                   color=[1.0, 1.0, 1.0, 1.0],
                   ball_color=[0.5, 0.5, 0.5, 1.0],
                   name="bar"):
  cylinder_sdf = f"""<?xml version="1.0"?>
  <sdf version="1.7">
    <model name="{name}">
      <pose>{' '.join(map(str, pose))}</pose>
      <link name="{name}_link">
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

      <link name="{name}_ball_1">
        <pose relative_to="{name}_link">0 0 {length/2} 0 0 0</pose>
        <inertial>
          <mass>0.001</mass>
        </inertial>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>{ball_radius}</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>{ball_radius}</radius>
            </sphere>
          </geometry>
          <material>
            <diffuse>{' '.join(map(str, ball_color))}</diffuse>
          </material>
        </visual>
      </link>

      <link name="{name}_ball_2">
      <pose relative_to="{name}_link">0 0 -{length/2} 0 0 0</pose>
        <inertial>
          <mass>0.001</mass>
        </inertial>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>{ball_radius}</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>{ball_radius}</radius>
            </sphere>
          </geometry>
          <material>
            <diffuse>{' '.join(map(str, ball_color))}</diffuse>
          </material>
        </visual>
      </link>


      <joint name="{name}_joint" type="fixed">
        <parent>{name}_link</parent>
        <child>{name}_ball_1</child>
        <pose>0 0 0 0 0 0</pose>
      </joint>

      <joint name="{name}_joint2" type="fixed">
        <parent>{name}_link</parent>
        <child>{name}_ball_2</child>
        <pose>0 0 0 0 0 0</pose>
      </joint>

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
                <size>10.5 10.1 0.05</size>
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
                <size>10.5 10.1 0.05</size>
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
    simulator = Simulator(diagram)
    simulator.Initialize()
    # simulator.set_target_realtime_rate(1.0)
    return simulator

def run_simulation(diagram, meshcat, finish_time=5):
    simulator = initialize_simulation(diagram)
    meshcat.StartRecording()
    simulator.AdvanceTo(finish_time)
    meshcat.PublishRecording()

    return simulator

def create_middle_platform_sdf(name="middle_platform", 
                               mass=1.0,
                               inertia=[0.005833, 0.0, 0.0, 0.005833, 0.0, 0.005], 
                               pose=[0, 0, 1, 0, 0, 0]):
    cylinder_sdf = f"""<?xml version="1.0"?>  
    <sdf version="1.7">  
      <model name="{name}">  
        # <pose>{' '.join(map(str, pose))}</pose>  
        <pose>0 0 1 0 0 0</pose>  
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
          <collision name="middle_platform_collision">  
            <geometry>  
              <box>  
                <size>0.15 0.15 0.15</size>  
              </box>  
            </geometry>  
          </collision>  

          <visual name="middle_platform_visual">  
            <geometry>  
              <box>  
                <size>0.15 0.15 0.15</size>  
              </box>  
            </geometry>  
            <material>  
              <ambient>1 0.90000000000000002 0 0.5</ambient>  
              <diffuse>1 0.90000000000000002 0 0.5</diffuse>  
            </material>  
          </visual>  
        </link>  
        <link name="inside_ball">  
            <inertial>  
              <mass>6.5999999999999996</mass>  
              <inertia>  
                <ixx>0.12</ixx>  
                <iyy>0.12</iyy>  
                <izz>0.12</izz>  
              </inertia>  
            </inertial>  

            <collision name="inside_ball_collision">  
              <geometry>  
                <box>  
                  <size>0.1 0.1 0.1</size>  
                </box>  
              </geometry>  
            </collision>  

            <visual name="inside_ball_visual">  
              <geometry>  
                <box>  
                  <size>0.1 0.1 0.1</size>  
                </box>  
              </geometry>  
              <material>  
                <ambient>0 0.90000000000000002 0 1</ambient>  
                <diffuse>0 0.90000000000000002 0 1</diffuse>  
              </material>  
            </visual>  
          </link>
          <joint name="x_control" type="revolute">  
            <parent>middle_platform</parent>  
            <child>inside_ball</child>  
            <axis>  
              <xyz>1 0 0</xyz>  
            </axis>  
          </joint>  
      </model>  
    </sdf>  
        """  
    return cylinder_sdf 


def create_bar_sdf(start_point=[0, 0, 0], end_point=[0, 0, 0],  
                   mass=1.0,  
                   inertia=[0.005833, 0.0, 0.0, 0.005833, 0.0, 0.005],  
                   radius=0.1,  
                   ball_radius=0.01,  
                   color=[1.0, 1.0, 1.0, 1.0],  
                   ball_color=[0.5, 0.5, 0.5, 1.0],  
                   name="bar"):  
    
    # Calculate the length and the pose of the cylinder  
    direction = np.array(end_point) - np.array(start_point)  
    length = np.linalg.norm(direction)  
    u = direction / length
    # print(u)
    # pose = [(start_point[i] + end_point[i]) / 2 for i in range(3)] + [0, np.arcsin(u[2]), np.arctan2(u[1], u[0])] 
    if u[2] == 1:
        pose = [(start_point[i] + end_point[i]) / 2 for i in range(3)] + [0, 0, 0]
    else:
      roll = np.arccos(u[2])
      yaw = np.arcsin(u[0]/np.sin(roll))
      pose = [(start_point[i] + end_point[i]) / 2 for i in range(3)] + [-roll, 0, -yaw]
    # print(pose)

    # Generate SDF  
    cylinder_sdf = f"""<?xml version="1.0"?>  
    <sdf version="1.7">  
      <model name="{name}">  
        <pose>{' '.join(map(str, pose))}</pose>  
        <link name="{name}_link">  
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

        <link name="{name}_ball_1">  
          <pose relative_to="{name}_link">0 0 {length/2} 0 0 0</pose>  
          <inertial>  
            <mass>0.001</mass>  
          </inertial>  
          <collision name="collision">  
            <geometry>  
              <sphere>  
                <radius>{ball_radius}</radius>  
              </sphere>  
            </geometry>  
          </collision>  
          <visual name="visual">  
            <geometry>  
              <sphere>  
                <radius>{ball_radius}</radius>  
              </sphere>  
            </geometry>  
            <material>  
              <diffuse>{' '.join(map(str, ball_color))}</diffuse>  
            </material>  
          </visual>  
        </link>  

        <link name="{name}_ball_2">  
          <pose relative_to="{name}_link">0 0 -{length/2} 0 0 0</pose>  
          <inertial>  
            <mass>0.001</mass>  
          </inertial>  
          <collision name="collision">  
            <geometry>  
              <sphere>  
                <radius>{ball_radius}</radius>  
              </sphere>  
            </geometry>  
          </collision>  
          <visual name="visual">  
            <geometry>  
              <sphere>  
                <radius>{ball_radius}</radius>  
              </sphere>  
            </geometry>  
            <material>  
              <diffuse>{' '.join(map(str, ball_color))}</diffuse>  
            </material>  
          </visual>  
        </link>  

        <joint name="{name}_joint" type="fixed">  
          <parent>{name}_link</parent>  
          <child>{name}_ball_1</child>  
          <pose>0 0 0 0 0 0</pose>  
        </joint>  

        <joint name="{name}_joint2" type="fixed">  
          <parent>{name}_link</parent>  
          <child>{name}_ball_2</child>  
          <pose>0 0 0 0 0 0</pose>  
        </joint>  

      </model>  
    </sdf>  
    """  
    return cylinder_sdf  