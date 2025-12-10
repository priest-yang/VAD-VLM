def HierarchicalPlanning(historical_trajectory, future_trajecotry):
    prompt = f'''
        Your task here is to describe the trajectory of the ego vehicle based on the historical and future trajectory information using meta actions.
        
        To help you give precise description of the meta action, we provide the exact historical trajectory and future trajectory of the ego vehicle in the image. The historical trajectory is represented as a sequence of waypoints in the form of (x, y, z, yaw, velocity, velocity_yaw, acceleration, ) coordinates. The future trajectory is represented as a sequence of waypoints in the form of (x, y) coordinates. You can use this information to infer the driving decisions and actions.
        Historical Trajectory: {historical_trajectory} (-2s, -1s, -0.5s, 0s)
        Future Trajectory(x, y): {future_trajecotry} (0s-8s, interval: 1s)
        x's positive direction is the forward direction of the ego vehicle.
        y's positive direction is the left direction of the ego vehicle.
        
        ### Meta-actions:
        Based on the point above, you should grasp the position change of the ego car. You can use these meta-actions to describe the future trajectory of the ego vehicle. These actions include:
            - Speed-control actions: speed up, slow down, stop, wait
            - Turning actions: turn left, turn right, turn around
            - Lane-control actions: change lane, shift slightly to the left or right
        Notice that the summed duration of all meta actions should be 8s.
        
        If the meta-action relates to change lane, you should resort to the given bev image to determine the direction of the lane change. The front direction of the original ego car is the downward direction of the BEV image.

        For each meta-action, provide a combined decision description from 3 aspects:
            1. **Action **: The specific meta-action, such as 'turn left', 'turn right', 'accelerate', or 'stop'.
            2. **Subject **: The interacting object or lane involved, such as a pedestrian, vehicle, or lane.
            3. **Duration **: Specify the time frame for the action, including how long it should last like 3s.
        - Be careful to discriminate the left or right turn, and the lane change to the left or right. When necessary, you can resort to the given bev image to determine the direction of the lane change. The front direction of the original ego car is the downward direction of the BEV image.

        your output format should be a json object with the following structure:
        {{  
            "Meta_Actions": [
                "Action": "$$chosen from meta actions, must base your output on the Description of the Future Trajectory.$$",
                "Subject": "$$the interacting object or lane involved$$",
                "Duration": "$$Based on the time series of the Future Trajectory, figure out how long a meta action lasts.$$"
            ],
            "Decision_Description": "$$the instructions and reasons for the car to follow in a sentence$$"
        }}
        
    '''
    return prompt
