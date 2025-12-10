
def ScenePrompt(historical_trajectory, future_trajectory, description, scene_descrition):  # Corrected spelling
    prompt = f"""
        You are an expert in Driving Scene analysing. You will be shown the BEV picture of the ego car.
        
        To give a thorough analysis and planning for the scene, let's think step by step. You should output scene analysis and hierarchical planning based on the scene description:
        {scene_descrition}
        
        
        # Step 1: Scene Analysis
        Based on the previous description, please analyze the image and provide a comprehensive scene analysis, focusing on both the driving environment and critical objects:

        ### Scene Analysis:
        - Provide a detailed description of the driving environment, including weather, road type, time of day, and lane conditions.

        ### Critical Object Analysis: (One object may not include all three perspectives)
        For each critical object identified in the scene (you should only focus on the most important objects!), analyze it from the following three perspectives:
        1. **Static attributes**: Describe inherent properties of the object, such as visual cues (e.g., roadside billboards or oversized cargo on a truck).
        2. **Motion states**: Describe the object's dynamics, including its position, direction, and current action.
        3. **Particular behaviors**: Highlight any special actions or gestures that could influence the ego vehicle’s next decision (e.g., a pedestrian's gesture or sudden movement).
        
        To help you capture the motion of the other agents, we provide the following information:
            {description}
        
        Your output format should be a json object with the following structure:
        {{
            "Scene_Summary": "$$summary of the scene and predict the future of the scene$$",
                
            "Critical_Objects": [
            {{"class": "$$the most important objects in the scene$$", 
            "Characteristics": "$$characteristics of the objects from Static attributes, Motion states and Particular behaviour.$$, 
            "Influence": "$$potential influence of the objects on the ego vehicle$$", 
            }}
            ]
        }}

        # Step 2: Trajectory Description
        Your task here is to describe the trajectory of the ego vehicle based on the historical and future trajectory information, and the BEV image.
        
        In the BEV (Bird's Eye View) image, the downward direction represents the forward direction of the vehicle. Based on previous analysis, please analyze the bird's-eye view (BEV) image, paying special attention to the plotted trajectory,(red points denote the past position and white ones denote future trajectory) and provide a summary of the trajectory.
        
        You can't output the meta-actions by your judgement based on your previous analysis. You should output the meta-actions based on the white trajectory in the BEV pic and the given future states!!!!!!!!!!

        To help you give precise description of the meta action, we provide the exact historical trajectory and future trajectory of the ego vehicle in the image. The historical trajectory is represented as a sequence of waypoints in the form of (x, y, z, yaw, velocity, velocity_yaw, acceleration, ) coordinates. The future trajectory is represented as a sequence of waypoints in the form of (x, y, z, yaw) coordinates. You can use this information to infer the driving decisions and actions.
        Historical Trajectory: {historical_trajectory} (-2s, -1s, -0.5s, 0s)
        Future Trajectory(x, y, z, yaw): {future_trajectory} (0s-8s, frequency: 10Hz)
        
        ### Meta-actions:
        - Firstly, you should locate the bright white line (future trajectory) on the picture based on Future Trajectory. Identify and describe the full-term driving decisions based on the trajectory in the image. These actions include:
            - Speed-control actions: speed up, slow down, stop, wait
            - Turning actions: turn left, turn right, turn around
            - Lane-control actions: change lane, shift slightly to the left or right
        Notice that the summed duration of all meta actions should be 8s.


        - Meta-action should cover the change of the whole length of white plotted trajectory. For each meta-action, provide a combined decision description from 3 aspects:
            1. **Action **: The specific meta-action, such as 'turn left', 'turn right', 'accelerate', or 'stop'.
            2. **Subject **: The interacting object or lane involved, such as a pedestrian, vehicle, or lane.
            3. **Duration **: Specify the time frame for the action, including how long it should last or when it should begin.
        - Be careful to discriminate the left or right turn, and the lane change to the left or right. You should tell the left and right from the perspective of the ego car. The front direction of the ego car is the downward direction of the BEV image.

        your output format should be a json object with the following structure:
        {{  "Description of the Future Trajectory": "$$Provide a detailed description of the white spotted trajectory visualized in the bev image from the perspective of ego car. Describe its changes and trends. Its begining is on the ego car.$$",
            "Meta_Actions": [
                "Action": "$$chosen from meta actions, must base your output on the Description of the Future Trajectory.$$",
                "Subject": "$$the interacting object or lane involved$$",
                "Duration": "$$the time(s) for the action, directly output Xs$$"
            ],
            "Decision_Description": "$$the instructions and reasons for the car to follow in a sentence$$"
        }}
        
        You must combine three parts as one json like(ensure your output is a valid json):
        {{
            "Scene Analysis": ...
            "Hierarchical Planning": ...
        }}
        
        

    """
    return prompt
