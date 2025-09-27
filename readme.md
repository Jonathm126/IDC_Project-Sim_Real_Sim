# Installation Guide

## 1. Clone the repo and submodules
Open a new folder and clone the repository there. This will also pull `lerobot` and `gym-aloha` as submodules.
```bash
git clone jonathm126/IDC_Project-Sim_Real_Sim
cd IDC_Project-Sim_Real_Sim
git submodule update --init --recursive
```

### Policies and Datasets
Large files are not stored in this repo. They are hosted on Hugging Face Hub.
The repo is configured to fetch them automatically.

## 2. Apply local patch to lerobot
Some minor fixes and changes are applied.
```bash
cd libs/lerobot
git apply ../lerobot-ubuntu.patch
cd ../..
```

## 3. Create and activate a new conda environment
Optionally rename using `-n`.
This will install `lerobot` and `gym-aloha` as submodules, and also apply package overrides that are specified in `requirements.override.txt`. This is because some packages are installed by default in headless mode, and some additional packages are needed on top of the base `lerobot` install.
```bash
conda env create -f environment-ubuntu.yml
conda activate lerobot-sim-real-sim
```

## 4. Camera Support
```bash
sudo apt install libgtk2.0-dev pkg-config v4l-utils ffmpeg
```




# Custom Scripts

# Repo Explanation

# Useful Debbugging tools
### USB Debbugging
`sudo apt install lsusb`

### Camera Settings
`sudo apt install guvcview`
Make sure to set expusure time to constant, disable dynamic framerate.