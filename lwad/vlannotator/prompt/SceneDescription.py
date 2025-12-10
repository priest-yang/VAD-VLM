def SceneDescription():
    prompt = """
        You are an expert in Driving Scene analysing. You will be shown the its FPV picture.
        
        # Scene Description
        
        Please describe the FPV image content and provide a formatted output for the environment description and critical object identification:
        
        ### Environment Description:
        - **E_weather**: Describe the weather condition (e.g., sunny, rainy, snowy, etc.), and explain its impact on visibility and vehicle traction.
        - **E_time**: Describe the time of day (e.g., daytime or nighttime), and explain how it affects driving strategies due to visibility changes.
        - **E_road**: Describe the type of road (e.g., urban road or highway), and explain the challenges associated with it for driving.
        - **E_lane**: Describe the current lane positioning and possible maneuvers, particularly focusing on lane selection and control decisions.
        
        ### Critical Object Identification:
        - Identify each critical object in the scene. For each object, provide:
        1. Object category (e.g., pedestrian, vehicle, traffic signal).
        2. Approximate bounding box coordinates in the format (x1, y1, x2, y2).
        3. Explain the significance of the object to the current driving scenario.
        
        ### Scenario Categories:
        - Based on the scene information, select the category that best matches the scene from the following classifications:
        categories = [
            "Road Construction",
            "Close-range Cut-ins",
            "Roundabout",
            "Animals Crossing Road",
            "Traffic Police Officers",
            "Blocking Traffic Lights",
            "Cutting into Other Vehicle",
            "Ramp",
            "Debris on the Road",
            "Narrow Roads",
            "Pedestrians Popping Out",
            "People on Bus Posters",
            "Complex Intersections",
            "Near Multiple Vehicles",
            "On Pickup Dropoff",
            "Turning at Intersection",
            "Waiting for Traffic Lights",
            "Emergency Vehicles",
            "Parking Lot",     
        ]
        
        Your output format should be a json object with the following structure:
        {{
            "E_weather": "$$weather condition$$",
            "E_time": "$$judged by the brightness of the image$$",
            "E_road": "$$road type$$",
            "E_lane": "$$lane positioning and possible maneuvers, in less than 3 words$$",
            "Critical_Objects": [
                {{
                    "Category": "$$The category of the object which may influence the ego vehicle. You can omit those which won't interact with the ego vehicle's driving.$$",
                    "BoundingBox": "$$the location of the object on the picture. the whole height and width of a pic is 1, so you should give a tuple composed of 4 0.x, like (x_min, y_min, x_max, y_max). The first two ones represent the bottom left corner, and the second two ones represent the up right corner.$$",
                    "Description": "$$description of the object$$"
                }},
            ]
            "Scenario Category": $$category most matching the scene based on the pic you see$$
        }}
        """
    return prompt