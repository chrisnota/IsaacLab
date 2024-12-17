import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab_tasks.utils import parse_env_cfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo script to create an environment, render episodes, and save to video.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--video_path", type=str, required=True, help="Path to save the video file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    # parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # wrap environment to record videos
    env = RecordVideo(env, args_cli.video_path, step_trigger=lambda step: step == 0, video_length=200, disable_logger=True)

    # initialize simulation context
    sim = SimulationContext()
    # set initial camera position
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample random actions
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
