def SceneAnalysis(description=None, scene_descrition=None):
    prompt = f"""
        You are an expert in Driving Scene analysing. You will be shown the BEV picture of the ego car. The ego car is at the origin of the coordinate system. Heading y-axis is the forward direction.
        
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
        
        To help you capture the motion of the other agents, we provide the following agents information in ego coordinate system:
        x's positive direction is the forward direction of the ego vehicle.
        y's positive direction is the left direction of the ego vehicle.
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
        """
    return prompt