import subprocess
import os
import time

class FrankaGripperController:
    def __init__(self,
             workspace_path="/home/iitgn-robotics/Debojit_WS/franka_ros2_ws",
             namespace="/NS_1"):

        """
        Initializes the Franka Gripper Controller using Shell Commands.
        
        Args:
            workspace_path (str): Path to franka_ros2_ws
            namespace (str): The namespace of the robot (e.g. "/NS_1")
        """
        self.ws_path = workspace_path
        self.setup_file = os.path.join(self.ws_path, "install/setup.bash")
        
        # Construct Action Names based on your 'ros2 action list'
        # We handle the trailing slash in namespace automatically
        ns = namespace.rstrip('/')
        self.move_action = f"{ns}/franka_gripper/move"     # For Opening
        self.grasp_action = f"{ns}/franka_gripper/grasp"   # For Closing/Grasping
        self.homing_action = f"{ns}/franka_gripper/homing" # For Calibration
        
        # Action Types (Standard franka_msgs)
        self.move_type = "franka_msgs/action/Move"
        self.grasp_type = "franka_msgs/action/Grasp"
        self.homing_type = "franka_msgs/action/Homing"

    def _send_action_goal(self, action_name, action_type, goal_args):
        """
        Internal function to construct the shell command for 'ros2 action send_goal'.
        """
        # Construct the full bash command
        # Syntax: ros2 action send_goal <action_name> <action_type> "<yaml_data>"
        command = (
            f"source {self.setup_file} && "
            f"ros2 action send_goal {action_name} {action_type} \"{goal_args}\""
        )
        
        print(f"[FrankaGripper] Sending Goal: {command}")

        try:
            # executable='/bin/bash' is required for 'source' to work
            result = subprocess.run(
                command, 
                shell=True, 
                executable='/bin/bash',
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("[FrankaGripper] Action Sent Successfully.")
                # Optional: Print output if you want to see feedback
                # print(result.stdout) 
                return True
            else:
                print("[FrankaGripper] Error sending action:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"[FrankaGripper] Exception occurred: {e}")
            return False

    def open_gripper(self, width=0.08, speed=0.1):
        """
        Opens the gripper using the 'Move' action.
        """
        print(f"Opening Gripper (width={width}, speed={speed})...")
        # Construct YAML for Move action
        args = f"{{width: {width}, speed: {speed}}}"
        return self._send_action_goal(self.move_action, self.move_type, args)

    def close_gripper(self, width=0.04, speed=0.1, force=1.0, epsilon_inner=0.04, epsilon_outer=0.04):
        """
        Closes/Grasps using the 'Grasp' action.
        Requires epsilon (tolerance) for inner/outer grasp.
        """
        print(f"Closing Gripper (width={width}, force={force})...")
        # Construct YAML for Grasp action (Note the nested epsilon object)
        args = (
            f"{{width: {width}, "
            f"speed: {speed}, "
            f"force: {force}, "
            f"epsilon: {{inner: {epsilon_inner}, outer: {epsilon_outer}}}}}"
        )
        return self._send_action_goal(self.grasp_action, self.grasp_type, args)

    def homing_gripper(self):
        """
        Performs homing (calibration).
        """
        print("Homing Gripper...")
        # Homing usually takes an empty goal
        args = "{}"
        return self._send_action_goal(self.homing_action, self.homing_type, args)

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Point to your Franka Workspace
    WS_PATH = "/home/iitgn-robotics/Debojit_WS/franka_ros2_ws"
    
    # 2. Initialize (Make sure namespace matches your 'ros2 action list' output)
    franka = FrankaGripperController(workspace_path=WS_PATH, namespace="/NS_1")
    
    # --- TEST SEQUENCE ---
    
    # 1. Homing (Good practice to do once at start)
    #franka.homing_gripper()
    time.sleep(2) # Homing takes time
    
    # 2. Open Gripper
    franka.open_gripper(width=0.08)
    time.sleep(2)
    
    # 3. Close Gripper (Grasp)
    # Trying to grasp something at 4cm width
    #franka.close_gripper(width=0.04, force=1.0)