# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class TestController(KesslerController):
    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        self.eval_frames = 0

        # Fuzzy variables
        # Antecedent
        bullet_time = ctrl.Antecedent(np.arange(0,1.01,0.01), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi, math.pi, 0.1), 'theta_delta')
        asteroid_dist = ctrl.Antecedent(np.arange(0, 800, 1), 'asteroid_dist')

        # Consequents
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        ship_thrust = ctrl.Consequent(np.arange(-480, 480, 1), 'ship_thrust')
        ship_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_mine')

        # Membership functions
        # Bullet time
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 0.1)

        # Theta delta
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3, -1*math.pi/6)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/2, -1*math.pi/4, 0])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6, -1*math.pi/12, math.pi/30])
        theta_delta['Z']  = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, 0, math.pi/30])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-math.pi/30, math.pi/12, math.pi/6])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [0, math.pi/4, math.pi/2])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi/6, math.pi/3)

        # Asteroid distance
        asteroid_dist['Close']  = fuzz.zmf(asteroid_dist.universe, 0, 200)
        asteroid_dist['Medium'] = fuzz.trimf(asteroid_dist.universe, [150, 400, 650])
        asteroid_dist['Far']    = fuzz.smf(asteroid_dist.universe, 600, 800)

        # Ship turn
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -90])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-135, -90, -45])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90, -45, 0])
        ship_turn['Z']  = fuzz.trimf(ship_turn.universe, [-10, 0, 10])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0, 45, 90])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [45, 90, 135])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [90, 180, 180])

        # Ship fire
        ship_fire['No']  = fuzz.trimf(ship_fire.universe, [-1, -1, 0])
        ship_fire['Yes'] = fuzz.trimf(ship_fire.universe, [0, 1, 1])

        # Ship thrust
        ship_thrust['Back'] = fuzz.trimf(ship_thrust.universe, [-480, -480, 0])
        ship_thrust['Stop'] = fuzz.trimf(ship_thrust.universe, [-10, 0, 10])
        ship_thrust['Fwd']  = fuzz.trimf(ship_thrust.universe, [0, 480, 480])

        # Ship mine
        ship_mine['No']  = fuzz.trimf(ship_mine.universe, [-1, -1, 0])
        ship_mine['Yes'] = fuzz.trimf(ship_mine.universe, [0, 1, 1])
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """

        thrust = 50
        turn_rate = -90
        fire = True
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Test Controller"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"
