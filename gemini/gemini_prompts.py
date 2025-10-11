YELLOW_CAR_PROMPT = (
    "Locate the yellow school bus toy in the image. "
    "Identify two specific points along its long axis: "
    "1. The FRONT of the bus. "
    "2. The REAR of the bus. "
    "Return your response as JSON in this format: "
    '[{"front_point": [y, x], "rear_point": [y, x]}]'
    "Coordinates must be normalized between 0 and 1000."
)
