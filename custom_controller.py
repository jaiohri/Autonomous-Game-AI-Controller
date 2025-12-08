# ECE 449 Intelligent Systems Engineering
# Group Project Assignment
# Fall 2025
# --------------------------------------------------------------
# MEMBERS:
# - Jai Ohri (ohri1)
# - Orion Arthur Warawa (owarawa)
# - Haris Nisar Tanoli (htanoli)
# --------------------------------------------------------------
# REFERENCES:
# [1] D. Wilczak, "EasyGA: A Python genetic algorithm library," GitHub 
#     repository, 2021. [Online]. Available: 
#     https://github.com/danielwilczak101/EasyGA

# [2] Thales Group, "Kessler Game: Asteroids arcade game implementation 
#     for AI research," GitHub repository, 2023. [Online]. Available: 
#     https://github.com/ThalesGroup/kessler-game

# [3] ECE 449: Intelligent Systems Engineering, University of Alberta, 
#     Fall 2024. Lab 4: Fuzzy Logic Controllers.
#
# [4] ECE 449: Intelligent Systems Engineering, University of Alberta, 
#     Fall 2024. Lab 5: Genetic Algorithms.
# 
# [5] "scikit-fuzzy Examples," Python Hosted. [Online]. Available: 
#     https://pythonhosted.org/scikit-fuzzy/auto_examples/index.html



from kesslergame import KesslerController
from kesslergame import Scenario, TrainerEnvironment, GraphicsType
from typing import Dict, Tuple, Optional
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import random
import time
import EasyGA


# Setup membership functions for the fuzzy system
def setup_membership_functions(bullet_time, theta_delta, asteroid_dist, collision_risk, ship_speed,
                                ship_turn, ship_fire, ship_thrust, ship_mine, 
                                theta_delta_boundary=math.pi/90, asteroid_dist_close=200):
    
    # Bullet time membership functions (fixed values)
    bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
    bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
    bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 0.1)
    
    # Theta delta membership functions
    max_boundary = min(theta_delta_boundary, 2*math.pi/90) 
    theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30, -2*math.pi/90)
    theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
    theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-max_boundary, -max_boundary/2, 0])
    theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0, max_boundary/2, max_boundary])
    theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90, 2*math.pi/90, math.pi/30])
    theta_delta['PL'] = fuzz.smf(theta_delta.universe, 2*math.pi/90, math.pi/30)
    
    # Asteroid distance membership functions
    asteroid_dist['Close'] = fuzz.zmf(asteroid_dist.universe, 0, asteroid_dist_close)
    asteroid_dist['Medium'] = fuzz.trimf(asteroid_dist.universe, [asteroid_dist_close*0.5, asteroid_dist_close*1.5, asteroid_dist_close*2.5])
    asteroid_dist['Far'] = fuzz.smf(asteroid_dist.universe, asteroid_dist_close*2, 800)
    
    # Collision risk membership
    collision_risk['Low'] = fuzz.trimf(collision_risk.universe, [0.0, 0.0, 0.3])
    collision_risk['Medium'] = fuzz.trimf(collision_risk.universe, [0.2, 0.5, 0.8])
    collision_risk['High'] = fuzz.trimf(collision_risk.universe, [0.6, 1.0, 1.0])
    
    # Ship speed membership (normalized 0-1)
    ship_speed['Low'] = fuzz.zmf(ship_speed.universe, 0, 0.3)
    ship_speed['Medium'] = fuzz.trimf(ship_speed.universe, [0.2, 0.5, 0.8])
    ship_speed['High'] = fuzz.smf(ship_speed.universe, 0.7, 1.0)
    
    # Ship turn membership functions
    ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
    ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
    ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
    ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
    ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
    ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])
    
    # Ship fire membership functions
    ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
    ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])
    
    # Ship thrust membership functions
    ship_thrust['Reverse'] = fuzz.trimf(ship_thrust.universe, [-480, -200, -50])
    ship_thrust['Stop'] = fuzz.trimf(ship_thrust.universe, [-100, 0, 100])
    ship_thrust['Low'] = fuzz.trimf(ship_thrust.universe, [50, 150, 250])
    ship_thrust['Medium'] = fuzz.trimf(ship_thrust.universe, [200, 300, 400])
    ship_thrust['Full'] = fuzz.trimf(ship_thrust.universe, [350, 480, 480])
    
    # Ship mine membership functions
    ship_mine['N'] = fuzz.trimf(ship_mine.universe, [-1, -1, 0.0])
    ship_mine['Y'] = fuzz.trimf(ship_mine.universe, [0.0, 1, 1])


def create_rules(bullet_time, theta_delta, asteroid_dist, collision_risk, ship_speed, ship_turn, ship_fire, ship_thrust, ship_mine):
    
    return [
        
        # Firing rules as a function of bullet time (how long until impact) and aim error.
        ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        
        # Distance and risk-based tweaks to thrust, firing, and mine usage.
        ctrl.Rule(asteroid_dist['Close'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Close'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Close'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Close'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Close'] & (collision_risk['Low'] | collision_risk['Medium']),
                  (ship_mine['N'], ship_thrust['Stop'])),
        ctrl.Rule(asteroid_dist['Far'] & theta_delta['NS'], (ship_thrust['Low'], ship_fire['Y'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Far'] & theta_delta['PS'], (ship_thrust['Low'], ship_fire['Y'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Medium'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Medium'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Medium'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(asteroid_dist['Medium'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Stop'], ship_mine['N'])),
        
        # Collision risk adjusts how aggressively we thrust and fire.
        ctrl.Rule(collision_risk['High'] & asteroid_dist['Close'],
                  (ship_thrust['Medium'], ship_fire['Y'], ship_mine['Y'])),
        ctrl.Rule(collision_risk['Medium'] & asteroid_dist['Medium'],
                  (ship_thrust['Low'], ship_fire['Y'], ship_mine['N'])),
        ctrl.Rule(collision_risk['Low'],
                  (ship_thrust['Stop'], ship_mine['N'])),
        
        # Speed-based braking when no immediate threat
        ctrl.Rule(collision_risk['Low'] & ship_speed['High'],
                  (ship_thrust['Reverse'], ship_mine['N'])),
        ctrl.Rule(collision_risk['Low'] & ship_speed['Medium'],
                  (ship_thrust['Stop'], ship_mine['N'])),
        ctrl.Rule(collision_risk['Low'] & ship_speed['Low'] & asteroid_dist['Far'],
                  (ship_thrust['Low'], ship_mine['N'])),
        
        # Don't fire if asteroid is extremely far AND aim is way off AND bullet time is long (very wasteful shot)
        ctrl.Rule(asteroid_dist['Far'] & bullet_time['L'] & (theta_delta['NL'] | theta_delta['PL']),
                  (ship_fire['N'],)),
        
        # fire whenever there's an asteroid in range (liberal firing because why not, we've got unlimited bullets)
        ctrl.Rule(asteroid_dist['Close'] | asteroid_dist['Medium'] | asteroid_dist['Far'],
                  (ship_fire['Y'],)),
        
        # stop when low/medium risk and no specific speed/distance conditions
        ctrl.Rule(collision_risk['Low'] & ship_speed['Low'] & (asteroid_dist['Close'] | asteroid_dist['Medium']),
                  (ship_thrust['Stop'],)),
    ]


class CustomController(KesslerController):
    
    
    # ga_params is used by the GA to optimize the parameters of the fuzzy system
    def __init__(self, ga_params: Optional[list] = None):
        self.eval_frames = 0
        self.last_mine_drop_frame = -999
        
        # Default parameters (optimized using GA)
        if ga_params is None:
            ga_params = [0.24822909454568576, 0.2168790098988791, 0.15056300061439054, 0.49207323935401404]
        
        # Scale GA parameters [0, 1] to actual parameter ranges
        # ga_params[0]: theta_delta_boundary range [π/180 (1°), π/18 (10°)]
        theta_delta_boundary = math.pi/180 + ga_params[0] * (math.pi/18 - math.pi/180)
        # ga_params[1]: asteroid_dist_close range [100, 400] units
        asteroid_dist_close = 100 + ga_params[1] * (400-100)
        # ga_params[2]: fire_threshold range [-0.85, -0.55] (more negative = more liberal)
        fire_threshold = -0.85 + ga_params[2] * (-0.55 - (-0.85))
        # ga_params[3]: turn_scale range [0.5, 2.0] (multiplier for turn rate)
        turn_scale = 0.5 + ga_params[3] * (2.0 - 0.5)
        
        # Build fuzzy control system
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30, math.pi/30, 0.1), 'theta_delta')
        asteroid_dist = ctrl.Antecedent(np.arange(0, 800, 1), 'asteroid_dist')
        collision_risk = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'collision_risk')
        ship_speed = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'ship_speed')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')
        ship_thrust = ctrl.Consequent(np.arange(-480, 480, 1), 'ship_thrust')
        ship_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_mine')
        
        # Set up membership functions
        setup_membership_functions(bullet_time, theta_delta, asteroid_dist, collision_risk, ship_speed,
                                  ship_turn, ship_fire, ship_thrust, ship_mine,
                                  theta_delta_boundary, asteroid_dist_close)
        
        # Create rules
        rules = create_rules(bullet_time, theta_delta, asteroid_dist, collision_risk, ship_speed,
                            ship_turn, ship_fire, ship_thrust, ship_mine)
        
        # Build control system
        self.targeting_control = ctrl.ControlSystem()
        for rule in rules:
            self.targeting_control.addrule(rule)
        
        # Store GA parameters for use in actions()
        self.fire_threshold = fire_threshold
        self.turn_scale = turn_scale

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        
        # Ship state
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        ship_radius = float(ship_state["radius"])
        ship_speed = float(ship_state["speed"])
        ship_vel_x, ship_vel_y = ship_state["velocity"]
        
        # helper function to calculate the difference between two angles
        def angle_diff(a, b):
            return (a - b + math.pi) % (2 * math.pi) - math.pi

        # helper function to check for threats and calculate escape direction along with risk score
        def threat_check():
            # Find most urgent collision threat and escape direction (within 300 units)
            max_check_distance = 300.0
            collision_horizon = 3.0 # how far into the future to check for threats
            safety_margin = 10.0 # how much space to leave between the ship and the threat

            best_threat_time = None
            best_escape_theta = None
            best_d2 = None
            best_safe_radius = None

            # Loop through all asteroids and check for threats
            for asteroid in game_state["asteroids"]:
                ax, ay = asteroid["position"]
                avx, avy = asteroid["velocity"]
                aradius = float(asteroid["radius"])

                # Calculate relative position and distance
                rx = ax - ship_pos_x
                ry = ay - ship_pos_y
                current_dist = math.sqrt(rx * rx + ry * ry)
                
                if current_dist > max_check_distance:
                    continue  # Skip distant asteroids
                
                # Calculate relative velocity
                rvx = avx - ship_vel_x
                rvy = avy - ship_vel_y
                v2 = rvx * rvx + rvy * rvy
                if v2 < 1e-6:
                    continue  # Skip if relative velocity too small

                # Compute time to closest approach
                rdotv = rx * rvx + ry * rvy
                t_ca = - rdotv / v2

                if t_ca <= 0.0 or t_ca > collision_horizon:
                    continue  # Skip if past or too far in future

                # Calculate distance at closest approach
                rca_x = rx + rvx * t_ca
                rca_y = ry + rvy * t_ca
                d2 = rca_x * rca_x + rca_y * rca_y

                # Check if collision is imminent
                safe_radius = ship_radius + aradius + safety_margin
                if d2 <= safe_radius * safe_radius:
                    escape_theta = math.atan2(-ry, -rx)  # Escape opposite to threat

                    # Track most urgent threat (earliest collision)
                    if (best_threat_time is None) or (t_ca < best_threat_time):
                        best_threat_time = t_ca
                        best_escape_theta = escape_theta
                        best_d2 = d2
                        best_safe_radius = safe_radius

            # Calculate risk score (0.0 = safe, 1.0 = imminent collision)
            risk = 0.0
            if best_threat_time is not None:
                t_score = max(0.0, min(1.0, (collision_horizon - best_threat_time) / collision_horizon))
                d_min = math.sqrt(best_d2)
                d_score = max(0.0, min(1.0, (best_safe_radius - d_min) / best_safe_radius))
                risk = max(t_score, d_score)
            return best_escape_theta, risk

        # helper function to determine if the ship should drop a mine based on fuzzy output and conditions
        def should_drop_mine(fuzzy_mine_output: float) -> bool:
            # Require strong fuzzy signal (0.8+) to drop mine because mines create chaos and are used as a last resort. Plus, we only have 3 mines.
            fuzzy_mine_signal = bool(fuzzy_mine_output >= 0.8)
            
            # Check mine availability
            if "mines_remaining" in ship_state:
                has_mines = ship_state["mines_remaining"] > 0
            else:
                has_mines = False
            
            if "can_deploy_mine" in ship_state:
                can_drop = ship_state["can_deploy_mine"]
            else:
                can_drop = False
            
            # Only drop mine if fuzzy system says yes AND mines are available
            return bool(fuzzy_mine_signal and has_mines and can_drop)

        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                if closest_asteroid["dist"] > curr_dist:
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800
        
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        

        # Thrust limits from ShipState
        thrust_min, thrust_max = ship_state["thrust_range"]
        forward_thrust = float(thrust_max)
        reverse_thrust = float(thrust_min)

        heading_rad = (math.pi / 180.0) * ship_state["heading"]
        
        escape_theta, risk = threat_check()
        
        # Normalize ship speed for fuzzy system
        if "max_speed" in ship_state:
            max_spd = float(ship_state["max_speed"])
        else:
            max_spd = 240.0
        normalized_speed = min(1.0, ship_speed / max_spd) if max_spd > 0 else 0.0

        # Compute fuzzy system outputs
        sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        sim.input['bullet_time'] = bullet_t
        sim.input['theta_delta'] = shooting_theta
        sim.input['asteroid_dist'] = closest_asteroid["dist"]
        sim.input['collision_risk'] = risk
        sim.input['ship_speed'] = normalized_speed
        sim.compute()
        
        # Get turn rate from fuzzy system
        turn_rate = float(sim.output['ship_turn']) * self.turn_scale
        tr_min, tr_max = ship_state["turn_rate_range"]
        turn_rate = max(tr_min, min(tr_max, turn_rate))
        
        # fire based on fuzzy system output
        fire = bool(sim.output['ship_fire'] >= self.fire_threshold)
        
        # Constrained mine dropping: only drop when absolutely required
        drop_mine = should_drop_mine(sim.output['ship_mine'])
        
        # Get fuzzy thrust output as baseline
        fuzzy_thrust = float(sim.output['ship_thrust'])
        
        # Constrained thrust logic with mine escape priority
        mine_escape_frames = 30
        frames_since_mine_drop = self.eval_frames - self.last_mine_drop_frame
        
        # PRIORITY 1: Escape from recently dropped mine (safety override)
        if drop_mine:
            self.last_mine_drop_frame = self.eval_frames
            thrust = reverse_thrust  # Full reverse to escape
        elif frames_since_mine_drop < mine_escape_frames:
            # Continue escaping from mine
            thrust = reverse_thrust * 0.8
            
        # PRIORITY 2: Avoid incoming asteroid collisions (safety override)
        elif escape_theta is not None and risk > 0.3:
            df = abs(angle_diff(escape_theta, heading_rad))
            db = abs(angle_diff(escape_theta, heading_rad + math.pi))
            
            if db < df:
                thrust = reverse_thrust * 0.4
            else:
                thrust = forward_thrust * 0.3
                
        # PRIORITY 3: Use fuzzy thrust output for normal operation
        else:
            thrust = fuzzy_thrust
        
        thrust = max(thrust_min, min(thrust_max, thrust))
        
        self.eval_frames +=1
        
        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Custom Controller"


# Simple Genetic Algorithm for Optimization
def extract_genes(chromosome):
    if hasattr(chromosome, 'gene_list'):
        genes = chromosome.gene_list
    elif hasattr(chromosome, 'genes'):
        genes = chromosome.genes
    else:
        genes = chromosome
    
    return [float(g.value if hasattr(g, 'value') else g) for g in genes]

# Genetic Algorithm for Optimization
if __name__ == "__main__":
    # Fitness function (lower is better)
    def fitness(chromosome):
        try:
            params = extract_genes(chromosome)
            controller = CustomController(ga_params=params)
            
            scenario = Scenario(
                name='GA Test',
                num_asteroids=10,
                ship_states=[{'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3}],
                map_size=(1000, 800),
                time_limit=20,
                ammo_limit_multiplier=0,
                stop_if_no_ammo=False
            )
            
            game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}
            game = TrainerEnvironment(settings=game_settings)
            
            score, _ = game.run(scenario=scenario, controllers=[controller])
            team_score = score.teams[0]
            
            # Fitness: maximize hits, penalize deaths (lower is better)
            # Negative of hits (so more hits = lower fitness = better)
            # Add penalty for deaths to avoid reckless behavior
            return -team_score.asteroids_hit * 10 + team_score.deaths * 50
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return 10000.0  # Bad fitness on error
    
    # Setup and run GA
    print("Running Simple GA Optimization...")
    ga = EasyGA.GA()
    ga.chromosome_length = 4
    ga.population_size = 15
    ga.generation_goal = 30
    ga.target_fitness_type = "min"
    ga.gene_impl = lambda: random.random()
    ga.fitness_function_impl = fitness
    
    print(f"Population: {ga.population_size}, Generations: {ga.generation_goal}")
    print("Starting evolution...\n")
    
    # Start stopwatch
    start_time = time.perf_counter()
    ga.evolve()
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\nEvolution completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
    
    # Get best chromosome
    if hasattr(ga.population, 'chromosome_list') and ga.population.chromosome_list:
        chromosomes = sorted(ga.population.chromosome_list, 
                           key=lambda c: c.fitness if hasattr(c, 'fitness') else float('inf'))
        best_chromosome = chromosomes[0]
        best_params = extract_genes(best_chromosome)
        
        print(f"\nBest fitness: {best_chromosome.fitness if hasattr(best_chromosome, 'fitness') else 'N/A'}")
        print(f"Best parameters: {best_params}")
    else:
        print("Could not extract best chromosome.")
