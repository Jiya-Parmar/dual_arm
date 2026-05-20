import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GripperDefaults:
    # Workspace + ROS2 service defaults
    workspace_path: str = "/home/iitgn-robotics/Debojit_WS/addverb_ws"
    service_name: str = "/gripper_controller/command"
    service_type: str = "addverb_cobot_msgs/srv/Gripper"
    setup_relpath: str = "install/setup.bash"

    # Command defaults
    open_position: float = 1.0
    close_position: float = 0.0
    default_force: float = 100.0


class GripperController:
    """
    Simple CLI-based gripper controller:
      - sources <ws>/install/setup.bash
      - calls: ros2 service call /gripper_controller/command addverb_cobot_msgs/srv/Gripper "{position: X, grasp_force: Y}"

    Notes:
      - This is blocking (subprocess.run).
      - Works even when called from scripts without sourcing the workspace externally.
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        service_name: Optional[str] = None,
        service_type: Optional[str] = None,
        setup_file: Optional[str] = None,
        default_force: Optional[float] = None,
        open_position: Optional[float] = None,
        close_position: Optional[float] = None,
        verbose: bool = True,
    ):
        d = GripperDefaults()

        self.ws_path = workspace_path or d.workspace_path
        self.setup_file = setup_file or os.path.join(self.ws_path, d.setup_relpath)
        self.service_name = service_name or d.service_name
        self.service_type = service_type or d.service_type

        self.default_force = float(d.default_force if default_force is None else default_force)
        self.open_position = float(d.open_position if open_position is None else open_position)
        self.close_position = float(d.close_position if close_position is None else close_position)

        self.verbose = bool(verbose)

        if not os.path.isfile(self.setup_file):
            raise FileNotFoundError(
                f"setup.bash not found at: {self.setup_file}\n"
                f"Check workspace_path='{self.ws_path}' or build/source your workspace."
            )

    def _build_command(self, position: float, force: float) -> str:
        # YAML-like request string (no extra escaping beyond outer quotes)
        args = f"{{position: {float(position)}, grasp_force: {float(force)}}}"
        # Use bash so `source` works.
        return (
            f"source {self.setup_file} && "
            f"ros2 service call {self.service_name} {self.service_type} \"{args}\""
        )

    def _run(self, command: str, timeout_s: float) -> subprocess.CompletedProcess:
        if self.verbose:
            print(f"[GripperController] Executing:\n  {command}")
        return subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def _send_command(
        self,
        *,
        position: float,
        force: Optional[float] = None,
        timeout_s: float = 3.0,
    ) -> bool:
        force_val = self.default_force if force is None else float(force)
        cmd = self._build_command(position=position, force=force_val)

        try:
            result = self._run(cmd, timeout_s=timeout_s)

            if result.returncode == 0:
                if self.verbose:
                    print("[GripperController] Success:")
                    if result.stdout.strip():
                        print(result.stdout)
                return True

            # Non-zero exit: show useful info
            print("[GripperController] ERROR (non-zero return code).")
            if result.stderr.strip():
                print(result.stderr)
            if self.verbose and result.stdout.strip():
                print("[GripperController] stdout:")
                print(result.stdout)
            return False

        except subprocess.TimeoutExpired:
            print(f"[GripperController] ERROR: command timed out after {timeout_s:.1f}s")
            return False
        except Exception as e:
            print(f"[GripperController] Exception occurred: {e}")
            return False

    # Public API (kept same names you already used)
    def close_gripper(self, *, force: Optional[float] = None, timeout_s: float = 3.0) -> bool:
        if self.verbose:
            print("Closing Gripper...")
        return self._send_command(position=self.close_position, force=force, timeout_s=timeout_s)

    def open_gripper(self, *, force: Optional[float] = None, timeout_s: float = 3.0) -> bool:
        if self.verbose:
            print("Opening Gripper...")
        return self._send_command(position=self.open_position, force=force, timeout_s=timeout_s)

    # Optional convenience:
    def set(self, position: float, *, force: Optional[float] = None, timeout_s: float = 3.0) -> bool:
        """Set an arbitrary position in [0,1] (or whatever your driver expects)."""
        return self._send_command(position=float(position), force=force, timeout_s=timeout_s)


# --- Usage Example ---
if __name__ == "__main__":
    gripper = GripperController()  # uses all defaults

    #gripper.close_gripper()        # default force
    #import time
    #time.sleep(2.0)
    gripper.open_gripper(force=40) # override force
