import robotic as ry
# import manipulation as manip
import numpy as np
#from importlib import reload
import time
import random

C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

C.addFrame('box', 'table') \
    .setJoint(ry.JT.rigid) \
    .setShape(ry.ST.ssBox, [.15,.06,.06,.005]) \
    .setRelativePosition([-.0,.3-.055,.095]) \
    .setContact(1) \
    .setMass(.1)

C.addFrame('obstacle', 'table') \
    .setShape(ry.ST.ssBox, [.06,.15,.06,.005]) \
    .setColor([.1]) \
    .setRelativePosition([-.15,.3-.055,.095]) \
    .setContact(1)

C.delFrame('panda_collCameraWrist')

# for convenience, a few definitions:
qHome = C.getJointState()
gripper = 'l_gripper'
palm = 'l_palm'
box = 'box'
table = 'table'
boxSize = C.getFrame(box).getSize()

C.view()

#reload(manip)

C.setJointState(qHome)
# C.view_raise()

C.getFrame(box).setRelativePosition([-.0,.3-.055,.095])
C.getFrame(box).setRelativeQuaternion([1.,0,0,0])

for i in range(7):
        qStart = C.getJointState()

        graspDirection = 'yz' #random.choice(['xz', 'yz'])
        placeDirection = 'z'
        place_position = [(i%3)*.3-.3, .2]
        place_orientation = [-(i%2),((i+1)%2),0.]
        info = f'placement {i}: grasp {graspDirection} place {placeDirection} place_pos {place_position} place_ori {place_orientation}'
        print('===', info)

        M = ry.KOMO_ManipulationHelper()
        M.setup_pick_and_place_waypoints(C, gripper, box, homing_scale=1e-1, joint_limits=False)
        M.grasp_top_box(1., gripper, box, graspDirection)
        M.place_box(2., box, table, palm, placeDirection)
        M.target_relative_xy_position(2., box, table, place_position)
        M.target_x_orientation(2., box, place_orientation)
        M.solve()
        if not M.feasible:
                continue

        M2 = M.sub_motion(0)
        M2.retract([.0, .2], gripper)
        M2.approach([.8, 1.], gripper)
        M2.solve()
        if not M2.ret.feasible:
            continue

        M3 = M.sub_motion(1)
#         M3.retract([.0, .2], box, distance=.05)
#         M3.approach([.8, 1.], box, distance=.05)
        M3.no_collisions([], [table, box])
        M3.no_collisions([], [box, 'obstacle'])
        M3.bias(.5, qHome, 1e0)
        M3.solve()
        if not M3.ret.feasible:
            continue

        M2.play(C)
        # fake/simulate pick
        C.attach(gripper, box)
        M3.play(C)
        # fake/simulate place
        C.attach(table, box)

del M
