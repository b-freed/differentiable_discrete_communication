import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
from threading import Lock
import sys
from matplotlib.colors import hsv_to_rgb
import math


'''
    Observation: (position maps of current agent, current goal, other agents, other goals, obstacles)
        
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''
ACTION_COST      = -0.3
IDLE_COST        = -.4
CLOSER_REWARD    = 0.2
GOAL_REWARD      = 0.0
COLLISION_REWARD = -.4
FINISH_REWARD    = 0.0
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
msg_length       = 10

class State(object):
    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    '''
    def __init__(self, world0, goals, diagonal, num_agents=1):
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        self.state                    = world0.copy()
        self.goals                    = goals.copy()
        self.num_agents               = num_agents
        self.msg_buffer               = np.zeros((self.num_agents, msg_length))
        self.placedID                 = np.zeros(world0.shape)
        self.agents, self.agents_past, self.agent_goals = self.scanForAgents()
        self.diagonal=diagonal
        assert(len(self.agents) == num_agents)
#        print("Initialized world state!")

    def scanForAgents(self):
        agents = [(-1,-1) for i in range(self.num_agents)]
        agents_last = [(-1,-1) for i in range(self.num_agents)]        
        agent_goals = [(-1,-1) for i in range(self.num_agents)]        
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if(self.state[i,j]>0):
                    agents[self.state[i,j]-1] = (i,j)
                    agents_last[self.state[i,j]-1] = (i,j)
                if(self.goals[i,j]>0):
                    agent_goals[self.goals[i,j]-1] = (i,j)
        assert((-1,-1) not in agents and (-1,-1) not in agent_goals)
        assert(agents==agents_last)
        return agents, agents_last, agent_goals

    def getPos(self, agent_id):
        return self.agents[agent_id-1]

    def getPastPos(self, agent_id):
        return self.agents_past[agent_id-1]

    def getGoal(self, agent_id):
        return self.agent_goals[agent_id-1]
    
    def diagonalCollision(self, agent_id, newPos):
        '''diagonalCollision(id,(x,y)) returns true if agent with id "id" collided diagonally with 
        any other agent in the state after moving to coordinates (x,y)
        agent_id: id of the desired agent to check for
        newPos: coord the agent is trying to move to (and checking for collisions)
        '''
#        def eq(f1,f2):return abs(f1-f2)<0.001
        def collide(a1,a2,b1,b2):
            '''
            a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
            '''
            return np.isclose( (a1[0]+a2[0]) /2. , (b1[0]+b2[0])/2. ) and np.isclose( (a1[1]+a2[1])/2. , (b1[1]+b2[1])/2. )
        assert(len(newPos) == 2);
        #up until now we haven't moved the agent, so getPos returns the "old" location
        lastPos = self.getPos(agent_id)
        for agent in range(1,self.num_agents+1):
            if agent == agent_id: continue
            aPast = self.getPastPos(agent)
            aPres = self.getPos(agent)
            if collide(aPast,aPres,lastPos,newPos): return True
        return False

    #try to move agent and return the status
    def moveAgent(self, direction, agent_id):
        ax=self.agents[agent_id-1][0]
        ay=self.agents[agent_id-1][1]

        # Not moving is always allowed
        if(direction==(0,0)):
            self.agents_past[agent_id-1]=self.agents[agent_id-1]
            return 1 if self.goals[ax,ay]==agent_id else 0

        # Otherwise, let's look at the validity of the move
        dx,dy =direction[0], direction[1]
        if(ax+dx>=self.state.shape[0] or ax+dx<0 or ay+dy>=self.state.shape[1] or ay+dy<0):#out of bounds
            return -1
        if(self.state[ax+dx,ay+dy]<0):#collide with static obstacle
            return -2
        if(self.state[ax+dx,ay+dy]>0):#collide with robot
            return -3
        # check for diagonal collisions
        if(self.diagonal):
            if self.diagonalCollision(agent_id,(ax+dx,ay+dy)):
                return -3
        # No collision: we can carry out the action
        self.state[ax,ay] = 0
        self.state[ax+dx,ay+dy] = agent_id
        self.agents_past[agent_id-1]=self.agents[agent_id-1]
        self.agents[agent_id-1] = (ax+dx,ay+dy)
        if self.goals[ax+dx,ay+dy]==agent_id:
            return 1
        elif self.goals[ax+dx,ay+dy]!=agent_id and self.goals[ax,ay]==agent_id:
            return 2
        else:
            return 0

    # try to execture action and return whether action was executed or not and why
    #returns:
    #     2: action executed and left goal
    #     1: action executed and reached goal (or stayed on)
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    #    -3: collision with robot
    def act(self, action, agent_id):
        # 0     1  2  3  4 
        # still N  E  S  W
        direction = self.getDir(action)
        moved = self.moveAgent(direction,agent_id)
        return moved

    def getDir(self,action):
        dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
        return dirDict[action]

    # Compare with a plan to determine job completion
    def done(self):
        numComplete = 0
        for i in range(1,len(self.agents)+1):
            agent_pos = self.agents[i-1]
            if self.goals[agent_pos[0],agent_pos[1]] == i:
                numComplete += 1
        return numComplete==len(self.agents) #, numComplete/float(len(self.agents))


class MAPFEnv(gym.Env):
    def getFinishReward(self):
        return FINISH_REWARD
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    def __init__(self, num_agents=1, world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=10, PROB=(.2,.2), FULL_HELP=False):
        """
        Args:
            DIAGONAL_MOVEMENT: if the agents are allowed to move diagonally
            SIZE: size of a side of the square grid
            PROB: range of probabilities that a given block is an obstacle
            FULL_HELP
        """
        # Initialize member variables
        self.num_agents        = num_agents       
        self.SIZE              = SIZE
        self.PROB              = PROB
        self.fresh             = True
        self.FULL_HELP         = FULL_HELP
        self.finished          = False
        self.mutex             = Lock()
        self.DIAGONAL_MOVEMENT = DIAGONAL_MOVEMENT
        self.individual_rewards = np.zeros(self.num_agents)

        # Initialize data structures
        self._setWorld(world0,goals0)
        if DIAGONAL_MOVEMENT:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])
        self.viewer           = None

    def isConnected(self,world0):
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def firstFree(world0):
            for x in range(world0.shape[0]):
                for y in range(world0.shape[1]):
                    if world0[x,y]==0:
                        return x,y
        def floodfill(world,i,j):
            sx,sy=world.shape[0],world.shape[1]
            if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                return
            if(world[i,j]==-1):return
            world[i,j] = -1
            floodfill(world,i+1,j)
            floodfill(world,i,j+1)
            floodfill(world,i-1,j)
            floodfill(world,i,j-1)

        i,j = firstFree(world0)
        floodfill(world0,i,j)
        if np.any(world0==0):
            return False
        else:
            return True

    def getObstacleMap(self):
        return (self.world.state==-1).astype(int)
    
    def getGoals(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(self.world.getGoal(i))
        return result
    
    def getPositions(self):
        result=[]
        for i in range(1,self.num_agents+1):
            result.append(self.world.getPos(i))
        return result
    
    def _setWorld(self, world0=None, goals0=None):
        #defines the State object, which includes initializing goals and agents
        #sets the world to world0 and goals, or if they are None randomizes world
        if not (world0 is None and goals0 is None):
            self.initial_world = world0
            self.initial_goals = goals0
            self.world = State(world0,goals0,self.DIAGONAL_MOVEMENT,self.num_agents)
            return

        #otherwise we have to randomize the world
        #RANDOMIZE THE STATIC OBSTACLES
        prob=np.random.uniform(self.PROB[0],self.PROB[1])
        world     = -(np.random.rand(self.SIZE,self.SIZE)<prob).astype(int)
        while(not self.isConnected(world)):
            world = -(np.random.rand(self.SIZE,self.SIZE)<prob).astype(int)

        #RANDOMIZE THE POSITIONS OF AGENTS
        agent_counter = 1
        while agent_counter<=self.num_agents:
            x,y       = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
            if(world[x,y] == 0):
                world[x,y]=agent_counter
                agent_counter += 1        
        
        #RANDOMIZE THE GOALS OF AGENTS
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        while goal_counter<=self.num_agents:
            x,y = np.random.randint(0,goals.shape[0]),np.random.randint(0,goals.shape[1])
            if(goals[x,y]==0 and world[x,y]==0):
                goals[x,y]    = goal_counter
                goal_counter += 1

        self.initial_world = world
        self.initial_goals = goals
        self.world = State(world,goals,self.DIAGONAL_MOVEMENT,num_agents=self.num_agents)

#      
    # Returns an observation of an agent
    def _observe_actor(self,agent_id):
        assert(agent_id>0)

        # 1. Position map (one-hot matrix, gives agent's position)
        pos_map              = np.zeros(self.world.state.shape)
        agent_pos            = self.world.getPos(agent_id)
        px, py               = int(agent_pos[0]), int(agent_pos[1])
        pos_map[px,py]       = 1

        #2 Other agents goals
        goal_map_other       = np.zeros(self.world.state.shape)
        for agent in range(1,self.num_agents+1):
            if(agent==agent_id): continue #ignore the current agent
            goal             = self.world.getGoal(agent)
            px,py            = int(goal[0]),int(goal[1])
            goal_map_other[px,py] = 1

        # message mat
        msg_mat = []

        for i in range(self.num_agents):
            if i != agent_id - 1:
                msg_mat.append(self.world.msg_buffer[i,:])
        msg_mat = np.array(msg_mat)

        return [pos_map, goal_map_other], msg_mat


    def _observe_critic(self,agent_id):
        '''
        returns observation of full state from particular agent's "vantage point"
        returns:
        [1) position map for agent_id
        2) goal map for agent_id
        3) position map for other agent
        4) goal map for agent_id]
        5) obstacles
        6) message_mat
        '''

        assert(agent_id>0)

        # 1. Position map (one-hot matrix, gives agent's position)
        pos_map              = np.zeros(self.world.state.shape)
        agent_pos            = self.world.getPos(agent_id)
        px, py               = int(agent_pos[0]), int(agent_pos[1])
        pos_map[px,py]       = 1



        #2 goal map: position of agent's goal
        goal_map              = np.zeros(self.world.goals.shape)
        goal_pos              = self.world.getGoal(agent_id)
        px, py                = int(goal_pos[0]), int(goal_pos[1])        
        goal_map[px,py]       = 1

        #3 Other agents map
        poss_map             = np.zeros(self.world.state.shape)
        for agent in range(1,self.num_agents+1):
            if(agent==agent_id): continue #ignore the current agent
            pos              = self.world.getPos(agent)
            px, py           = int(pos[0]), int(pos[1])
            poss_map[px,py]  = 1

        #4 Other agents goals
        goals_map            = np.zeros(self.world.state.shape)
        for agent in range(1,self.num_agents+1):
            if(agent==agent_id): continue #ignore the current agent
            goal             = self.world.getGoal(agent)
            px,py            = int(goal[0]),int(goal[1])
            goals_map[px,py] = 1


        return [pos_map, goal_map, poss_map, goals_map]



    # Resets environment

    # Resets environment
    def _reset(self, agent_id,world0=None,goals0=None):
        self.finished = False
        self.mutex.acquire()

        if not self.fresh:
            # Initialize data structures
            self._setWorld(world0,goals0)
            self.fresh = True
        
        self.mutex.release()
        
        on_goal = self.world.getPos(agent_id) == self.world.getGoal(agent_id)
        #we assume you don't start blocking anyone (the probability of this happening is insanely low)
        return self._listNextValidActions(agent_id), on_goal,self.isBlocked(agent_id)

    def _complete(self):
        return self.world.done()
    
    def getAstarCosts(self,start, goal):
        #returns a numpy array of same dims as self.world.state with the distance to the goal from each coord
        def lowestF(fScore,openSet):
            #find entry in openSet with lowest fScore
            assert(len(openSet)>0)
            minF=2**31-1
            minNode=None
            for (i,j) in openSet:
                if (i,j) not in fScore:continue
                if fScore[(i,j)]<minF:
                    minF=fScore[(i,j)]
                    minNode=(i,j)
            return minNode      
        def getNeighbors(node):
            #return set of neighbors to the given node
            n_moves=9 if self.DIAGONAL_MOVEMENT else 5
            neighbors=set()
            for move in range(1,n_moves):#we dont want to include 0 or it will include itself
                direction=self.world.getDir(move)
                dx=direction[0]
                dy=direction[1]
                ax=node[0]
                ay=node[1]
                if(ax+dx>=self.world.state.shape[0] or ax+dx<0 or ay+dy>=self.world.state.shape[1] or ay+dy<0):#out of bounds
                    continue
                if(self.world.state[ax+dx,ay+dy]==-1):#collide with static obstacle
                    continue
                neighbors.add((ax+dx,ay+dy))
            return neighbors
        
        #NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
        start,goal=goal,start
        
        # The set of nodes already evaluated
        closedSet = set()
    
        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        openSet =set()
        openSet.add(start)
    
        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = dict()
    
        # For each node, the cost of getting from the start node to that node.
        gScore =dict()#default value infinity
    
        # The cost of going from start to start is zero.
        gScore[start] = 0
    
        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        fScore = dict()#default infinity
    
        #our heuristic is euclidean distance to goal
        heuristic_cost_estimate = lambda x,y:math.hypot(x[0]-y[0],x[1]-y[1])
        
        # For the first node, that value is completely heuristic.        
        fScore[start] = heuristic_cost_estimate(start, goal)
    
        while len(openSet) != 0:
            #current = the node in openSet having the lowest fScore value
            current = lowestF(fScore,openSet)
    
            openSet.remove(current)
            closedSet.add(current)
            for neighbor in getNeighbors(current):
                if neighbor in closedSet:
                    continue		# Ignore the neighbor which is already evaluated.
            
                if neighbor not in openSet:	# Discover a new node
                    openSet.add(neighbor)
                
                # The distance from start to a neighbor
                #in our case the distance between is always 1
                tentative_gScore = gScore[current] + 1
                if tentative_gScore >= gScore.get(neighbor,2**31-1):
                    continue		# This is not a better path.
            
                # This path is the best until now. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal) 
    
        #parse through the gScores
        costs=self.world.state.copy()
        for (i,j) in gScore:
            costs[i,j]=gScore[i,j]
        return costs
    
   
            
    def isBlocked(self,agent_id):
        def reachable(world,c1,c2,ignore):
            (a,b),(c,d)=c1,c2
            #requires: neither a,b or c,d are in walls
            #returns whether (a,b) is reachable from (c,d), with the option to count any
            #tile(s) as open ("ignore" parameter is a list of tiles to count as empty)
            visited=set()
            def floodfill(c3,c4):
                (i,j),(x,y)=c3,c4
                #determines if (x,y) is reachable from (i,j)
                if (i,j)==(x,y):return True
                if (i,j) in visited: return False
                sx,sy=world.shape[0],world.shape[1]
                if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                    return False
                if(world[i,j]!=0 and (i,j) not in ignore):return False
                visited.add((i,j))
                return floodfill((i+1,j),(x,y)) or floodfill((i,j+1),(x,y)) or \
                        floodfill((i-1,j),(x,y)) or floodfill((i,j-1),(x,y))
            return floodfill((a,b),(c,d))
        return not reachable(self.world.state,self.world.getGoal(agent_id),self.world.getPos(agent_id),[])

    def distance_to_goal(self, agent_id):
        '''
        takes agent id and returns current manhattan (L1) distance to goal
        '''

        pos  = self.world.getPos(agent_id)
        goal = self.world.getGoal(agent_id)

        return np.abs(goal[0] - pos[0]) + np.abs(goal[1] - pos[1])


    # Executes an action by an agent
    def _step(self, action_input,episode=0):
        #episode is an optional variable which will be used on the reward discounting
        self.fresh = False
        n_actions = 9 if self.DIAGONAL_MOVEMENT else 5


        # Check action input
        assert len(action_input) == 3, 'Action input should be a tuple with the form (agent_id, action, message)'
        assert action_input[1] in range(n_actions), 'Invalid action'
        assert action_input[0] in range(1, self.num_agents+1)

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]
        msg      = action_input[2]

        #get current distance to goal for this agent
        d2g = self.distance_to_goal(agent_id)

        # Lock mutex (race conditions start here)
        self.mutex.acquire()

        #get start location of agent
        agentStartLocation = self.world.getPos(agent_id)
        
        # Execute action & determine reward
        action_status = self.world.act(action,agent_id)
        valid_action=action_status >=0
        #     2: action executed and left goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot
        # blocking=self.isBlocked(agent_id)

        #get new distance to goal
        d2g_new = self.distance_to_goal(agent_id)

        if action==0:#staying still
            if action_status == 1:#stayed on goal
                reward=GOAL_REWARD

            elif action_status == 0:#stayed off goal
                reward=IDLE_COST
        else:#moving
            if (action_status == 1): # reached goal
                reward = GOAL_REWARD
            elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                reward = COLLISION_REWARD
            elif (action_status == 2): #left goal
                reward=ACTION_COST + CLOSER_REWARD*(d2g - d2g_new)
            else:
                reward=ACTION_COST + CLOSER_REWARD*(d2g - d2g_new)


        #Add reward to individual_rewards
        self.individual_rewards[agent_id - 1] = reward

        #insert message in message buffer
        self.world.msg_buffer[agent_id - 1, :] = msg

        # Perform observation
        state = self._observe_actor(agent_id) 

        # Done?
        done = self.world.done()
#         if done:
#             reward=FINISH_REWARD
        self.finished |= done

        # next valid actions
        nextActions = self._listNextValidActions(agent_id, action,episode=episode)

        # on_goal estimation
        on_goal = self.world.getPos(agent_id) == self.world.getGoal(agent_id)
        
        # Unlock mutex
        self.mutex.release()
        return state, None, done, nextActions, on_goal, False, valid_action

    def compute_reward(self):
        sum_r = np.sum(self.individual_rewards)
        if self.world.done():
            sum_r += FINISH_REWARD
        return sum_r

    def _listNextValidActions(self, agent_id, prev_action=0,episode=0):
        available_actions = [0] # staying still always allowed

        # Get current agent position
        agent_pos = self.world.getPos(agent_id)
        ax,ay     = agent_pos[0],agent_pos[1]
        n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

        for action in range(1,n_moves):
            direction = self.world.getDir(action)
            dx,dy     = direction[0],direction[1]
            if(ax+dx>=self.world.state.shape[0] or ax+dx<0 or ay+dy>=self.world.state.shape[1] or ay+dy<0):#out of bounds
                continue
            if(self.world.state[ax+dx,ay+dy]<0):#collide with static obstacle
                continue
            if(self.world.state[ax+dx,ay+dy]>0):#collide with robot
                continue
            # check for diagonal collisions
            if(self.DIAGONAL_MOVEMENT):
                if self.world.diagonalCollision(agent_id,(ax+dx,ay+dy)):
                    continue          
            #otherwise we are ok to carry out the action
            available_actions.append(action)

        # if opposite_actions[prev_action] in available_actions:
        #     available_actions.remove(opposite_actions[prev_action])

        return available_actions

    def drawStar(self, centerX, centerY, diameter, numPoints, color):
        outerRad=diameter//2
        innerRad=int(outerRad*3/8)
        #fill the center of the star
        angleBetween=2*math.pi/numPoints#angle between star points in radians
        for i in range(numPoints):
            #p1 and p3 are on the inner radius, and p2 is the point
            pointAngle=math.pi/2+i*angleBetween
            p1X=centerX+innerRad*math.cos(pointAngle-angleBetween/2)
            p1Y=centerY-innerRad*math.sin(pointAngle-angleBetween/2)
            p2X=centerX+outerRad*math.cos(pointAngle)
            p2Y=centerY-outerRad*math.sin(pointAngle)
            p3X=centerX+innerRad*math.cos(pointAngle+angleBetween/2)
            p3Y=centerY-innerRad*math.sin(pointAngle+angleBetween/2)
            #draw the triangle for each tip.
            poly=rendering.FilledPolygon([(p1X,p1Y),(p2X,p2Y),(p3X,p3Y)])
            poly.set_color(color[0],color[1],color[2])
            poly.add_attr(rendering.Transform())
            self.viewer.add_onetime(poly)

    def create_rectangle(self,x,y,width,height,fill):
        ps=[(x,y),((x+width),y),((x+width),(y+width)),(x,(y+width))]
        rect=rendering.FilledPolygon(ps)
        rect.set_color(fill[0],fill[1],fill[2])
        rect.add_attr(rendering.Transform())
        self.viewer.add_onetime(rect)

    def initColors(self):
        c={a+1:hsv_to_rgb(np.array([a/float(self.num_agents),1,1])) for a in range(self.num_agents)}
        return c

    def _render(self, mode='human', close=False,screen_width=800,screen_height=800):   
        #values is an optional parameter which provides a visualization for the value of each agent per step
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        size=screen_width/max(self.world.state.shape[0],self.world.state.shape[1])
        colors=self.initColors()
        if self.viewer==None:
            self.viewer=rendering.Viewer(screen_width,screen_height)        
        for i in range(self.world.state.shape[0]):
            for j in range(self.world.state.shape[1]):
                x=i*size
                y=j*size
                if(self.world.state[i,j]==0):#blank
                    color=(1,1,1)                
                    self.create_rectangle(x,y,size,size,color)                    
                elif(self.world.state[i,j]>0):#agent
                    color=colors[self.world.state[i,j]]
                    self.create_rectangle(x,y,size,size,color)
                elif(self.world.state[i,j]==-1):#wall
                    color=(.6,.6,.6)                
                    self.create_rectangle(x,y,size,size,color)
                if(self.world.goals[i,j]>0):#goal
                    color=colors[self.world.goals[i,j]]
                    self.drawStar(x+size/2,y+size/2,size,4,color)
                if(self.world.goals[i,j]>0 and self.world.goals[i,j]==self.world.state[i,j]):
                    color=(0,0,0)
                    self.drawStar(x+size/2,y+size/2,size,4,color)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__=='__main__':
    n_agents=10
    env=MAPFEnv(n_agents,PROB=(.2,.2),SIZE=10,DIAGONAL_MOVEMENT=True)
    import time
    while True:
        time.sleep(.1)
        for agent in range(1,n_agents+1):
            a=np.random.choice(np.array(env._listNextValidActions(agent)))
            a=np.random.randint(0,8)
            o,r,d,actions,to_goal=env._step((agent,a))
#         env._render()