#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)

    R            : toggle recording images to disk
    CTRL + {R,P} : {toggle,start} recording of simulation (replacing any previous)
    CTRL + {+,-} : {increments,decrements} the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import logging

import carla

from examples.manual_control import (World, HUD, KeyboardControl, CameraManager,
                                     CollisionSensor, LaneInvasionSensor, GnssSensor, IMUSensor)

import os
import platform
import sys
import mmap
import struct
import ctypes
import argparse
import collections
import datetime
import math
import random
import re
import weakref
import time
from tempfile import gettempdir as tmpd

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


collision_support  = True
imu_sensor_support = False
gnss_support       = False

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class WorldSR(World):

    restarted = False
    world_compass = True
    world_lane    = False

    def restart(self):
        if self.restarted:
            return
        self.restarted = True

        logging.debug("WorldSR::restart()")

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        while self.player is None:
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    logging.debug("Ego vehicle found")
                    self.player = vehicle
                    break
        
        self.player_name = self.player.type_id

        # Set up the global sensors.
        if collision_support:
            self.collision_sensor = CollisionSensor(self.player, self.hud)
        if gnss_support:
            self.gnss_sensor = GnssSensor(self.player)
        if imu_sensor_support:
            self.imu_sensor = IMUSensor(self.player)

        # Set up the local sensors.
        if self.lane_support:
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)

        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        # ~ GHadj: Always start from the driver's view camera
        # ~ self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False

        self.hud.tick(self, clock)
        return True

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    metrics_temp_file= os.getenv("CARLA_TMP_KEY_STROKES",None)
    if metrics_temp_file is None:
        metrics_temp_file = tmpd() + '/mmaptest_carla_simmulator_key_strokes.tmp'
    metrics_temp_file_max_size = int((mmap.PAGESIZE ** 2)/4)
    logging.info('Using %s for saving simulator\'s metric data, size = %d', metrics_temp_file, metrics_temp_file_max_size)
    logging.info('\tUnix path: %s',metrics_temp_file.replace(os.sep, '/'))
    offset = 0
    # Create new empty file to back memory map on disk
    fd = os.open(metrics_temp_file.replace(os.sep, '/'),os.O_CREAT|os.O_TRUNC|os.O_RDWR)
    
    os.lseek(fd, 0, os.SEEK_SET)

    # initialise the file with zeros
    assert os.write(fd, b'\x00' * metrics_temp_file_max_size) == metrics_temp_file_max_size
    os.fsync(fd)
    os.lseek(fd, 0, os.SEEK_SET)

    # Create the mmap instace with the following params:
    # fd: File descriptor which backs the mapping or -1 for anonymous mapping
    # length: Must in multiples of PAGESIZE (usually 4 KB)
    # flags: MAP_SHARED means other processes can share this mmap
    # prot: PROT_WRITE means this process can write to this mmap
    if platform.system() == "Linux":
        buf = mmap.mmap(fd, metrics_temp_file_max_size, mmap.MAP_SHARED, mmap.PROT_WRITE)
    elif platform.system() == "Windows":
        buf = mmap.mmap(fd, metrics_temp_file_max_size, access=mmap.ACCESS_WRITE)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        # ~ Calling display.Info before set_mode will give us the current screen resolution
        # ~ print(pygame.display.Info())

        user_mode_found = False
        mode=32
        modes = pygame.display.list_modes(mode)
        if not modes:
            logging.warning('%d color bit not supported', mode)
            mode=24
            modes = pygame.display.list_modes(mode)
            # ~ logging.debug("Modes: %s", modes)
            if not modes:
                logging.warning('%d color bit not supported', mode)
            else:
                for x in modes:
                    if (args.width, args.height) == x:
                        logging.debug('%d bit: Found User Resolution: %s', mode, x)
                        user_mode_found = True
                        break
                    # ~ logging.info('%d bit: Supported Resolution: %s', mode, x)
        else:
            for x in modes:
                logging.info('%d bit: Found Resolution: %dx%d', mode, args.width, args.height)

        if not user_mode_found:
            logging.error('User mode %s not found, exiting', x)
            exit(-20)

        '''
        open gl related, not working for carla 0.9.10.1 binary version
        gl_version = (4,3)
        # By setting these attributes we can choose which Open GL Profile
        # to use, profiles greater than 3.2 use a different rendering path
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, gl_version[0])
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, gl_version[1])
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        '''

        display = pygame.display.set_mode(
            size=x,
            flags=pygame.HWSURFACE | pygame.DOUBLEBUF,
            depth=mode,
            vsync=1
        )
        #print(pygame.display.Info())

        hud = HUD(args.width, args.height, collision_support, gnss_support, imu_sensor_support)

        world = WorldSR(client.get_world(), hud, args)
        logging.debug("calling world.restart()")
        world.restart()
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()

        metrics_data = dict()

        while True:
            msecs_for_previous_frame_render = clock.tick_busy_loop(args.max_fps)

            # if we draw earlier than expected then something may be wrong. for now lets simple quit. TODO: define an expected behaviour for the future
            if msecs_for_previous_frame_render < math.floor(1000/args.max_fps):
                logging.error('Frame draw quicker that expected(',msecs_for_previous_frame_render,'<',1000/args.max_fps,')!')

            # if we draw earlier than expected then something may be wrong. for now lets simple quit. TODO: define an expected behaviour for the future
            if msecs_for_previous_frame_render > args.human_reaction:
                logging.error('Frame draw after the minimum human reaction time: %s > %s', msecs_for_previous_frame_render, args.human_reaction)

            if controller.parse_events(client, world, clock, metrics_data):
                logging.error('Controller parse_events() return true, exiting')
                return
            if not world.tick(clock):
                return
            world.render(display)
            pygame.display.flip()

            if len(metrics_data) != 0:
                # ~ logging.debug('{:03d}'.format(metrics_data['index'].value),") -------Accel:",format(metrics_data['accel'].value, '.2f'),"--S",format(metrics_data['steer'].value, '.2f'),"--B",metrics_data['break'].value,"--HB",metrics_data['hbreak'].value,"--R",metrics_data['reverse'].value,"--MG",metrics_data['m_gear_shift'].value,"--G", metrics_data['gear'].value,"-------")
                prev_offset = offset
                for key, value in metrics_data.items():
                    # print('--------->', key, value.datatype, str(value.value), offset)

                    # Create a metrics_data[key].datatype in the memory mapping
                    i = getattr(ctypes,value.datatype).from_buffer(buf, offset)
                    if value.value!=-1:
                        i.value = value.value
                    else:
                        # reverse gear
                        i.value = b'\xFF'

                    #deserialize time_ns to localtime
                    # ~ print(time.localtime(i.value/(1e9)))

                    # Find the offset of the next free memory address within the mmap
                    offset += struct.calcsize(i._type_)
                    # The offset should be uninitialized ('\x00')
                    assert buf[offset] == 0
                    prev_offset=offset
                offset+=3

    finally:
        #~ Todo: Change print with logging mechanism.
        logging.warning('Finally Cancelled by user. Bye!')
        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():

    log_level = os.getenv("CARLA_PYTHON_DEBUG","False")
    if log_level == "False":
        log_level = logging.WARNING
    else:
        log_level = logging.DEBUG
    logging.basicConfig(format='**MC******%(levelname)s: %(message)s', level=log_level)

    platform_system = platform.system()
    if platform_system == "Linux":
        logging.info('Platform Linux')
    elif platform_system == "Windows":
        logging.info('Platform Windows')
    else:
        logging.error("Exiting: Unknown platform %s",platform.system())
        sys.exit(3)

    if platform.python_version() < "3.7.0":
        sys.exit("Python version greater than 3.7 is required")

    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--timeout',
        metavar='T',
        default=2.0,
        type=float,
        help='Server response timeout (default: 2.0)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--max_fps',
        metavar='FPS',
        default=60,
        type=int,
        help='Max FPS (default: 60)')
    argparser.add_argument(
        '--human_reaction',
        metavar='H_R',
        default=100,
        type=int,
        help='Min Human Reaction in msecs (default: 100)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        logging.warning('\nCancelled__ by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()
