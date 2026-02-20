from collections import deque

class WristValleyDetector:
    """
    Time-series logic to detect the "Wrist Valley" (Impact Frames) of video clips
    Tracks the Y-coordinate trajectory of the wrist of the body to find local minimums
    """

    def __init__(self, history_size = 5):
        self.y_history = deque(maxlen = history_size)

    def process_frame(self, current_y):
        """
        Feeds a new Y coordinate into the time series
        Pops off oldest Y coordinate to record averaged frames
        Returns true if a frame is the bottom of the valley

        :param self: The instance of the WristValleyDetector.
        :param current_y: Current Y-value being read by pose estimation model
        """

        self.y_history.append(current_y)

        #We need at least 3 frames to know if a flip happened
        if len(self.y_history) < 3:
            return False
        
        #Calculate velocity through dy
        # current_dy: Velocity between NOW and 1 frame ago
        current_dy = self.y_history[-1] - self.y_history[-2]

        #prev_dy: Velocity between previous frame and 2 frames ago
        prev_dy = self.y_history[-2] - self.y_history[-3]

        #Gradient Flip Logic
        #Note that y increases as you go further down on the screen
        #If previous velocity was positive, we are moving down
        #If previous velocity was negative, we are swinging up
        #When sign changes, we know that we have hit the bottom of swing arc
        if prev_dy > 0 and current_dy < 0:
            return True
        
        return False