# matrix_lerobot_arm
This is where the UMD Matrix Lab's LeRobot SO101 Arm Workspace lives which encapsulates
imitation and reinforcement learning on various policies.

Everything within this repository is taken from the official HuggingFace [tutorials](https://huggingface.co/docs/lerobot/en/il_robots)
Additionally, the LeRobot [Github](https://github.com/huggingface/lerobot) has good resources/examples

## Installation
Getting started with this repository is fairly simple since all of the resources are
python packages or located here.

1) Clone the repository to your system
    ```shell
    cd ~/workspace_name/ && git clone https://github.com/CursedRock17/matrix_lerobot_arm.git
    ```
2) Create a new conda environment so we have a local variant of our packages
    2a) If not installed, install miniforge
    ```shell
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && bash Miniforge3-$(uname)-$(uname -m).sh
    ```

    2b) Create our workspace and add some packages
    ```shell
    conda create -y -n workspace_name python=3.12
    conda activate workspace_name
    conda install ffmpeg -c conda-forge
    ```

3) Enter the lerobot source code (current tag: v0.5.1) and add to conda
    ```shell
    cd lerobot && pip install -e .
    ```

You're now ready to start building!

## Subsections
There are sections for the various parts of the process which contain higher level 
shell scripts or lower level (more verbose) python scripts for you to take advantage of.

- [Recording](recording) : Scripts for Recording SO101 Arms
- [Training](training) : Scripts for Training (Finetuning) Various VLA Models
- [Evaluating](evaluating) : Scripts for Evaluating (Deploying) the Finetuned Models


