# VLM Data Annotation Pipeline

This package contains a script designed for annotating visual language model (VLM) datasets, leveraging different API-based language models to generate scene descriptions, scene analyses, and hierarchical planning annotations. The script works with NuPlan dataset files and supports multiprocessing for efficiency.

**Note:** Before running the VLM data annotation pipeline, you have to firstly generate the raster images through ``tools/nuplan/raster_navigation.py``.

---

## Table of Contents
- [VLM Data Annotation Pipeline](#vlm-data-annotation-pipeline)
  - [Table of Contents](#table-of-contents)
  - [VLM Annotation (data\_annotation.py)](#vlm-annotation-data_annotationpy)
    - [Setup](#setup)
    - [Configuration](#configuration)
    - [Example Command](#example-command)
    - [Options](#options)
    - [API Keys and Configuration](#api-keys-and-configuration)
      - [ZhipuAI](#zhipuai)
      - [Qwen](#qwen)
      - [Azure OpenAI (for `gpt-4o`)](#azure-openai-for-gpt-4o)
    - [Troubleshooting](#troubleshooting)
  - [Annotation Merger (merge\_ann2pickle.py)](#annotation-merger-merge_ann2picklepy)
    - [Features](#features)
    - [Usage](#usage)
    - [Processing Details](#processing-details)
    - [Notes](#notes)
  - [Post-Processing (lwad/vlannotator/post\_processing.py)](#post-processing-lwadvlannotatorpost_processingpy)
    - [Features](#features-1)
    - [Meta-Action Categories](#meta-action-categories)
    - [Usage](#usage-1)
    - [Constants and Mappings](#constants-and-mappings)
      - [META\_ACTIONS](#meta_actions)
      - [EXCEPTION\_MAP](#exception_map)
      - [TARGET\_ACTION\_MAP](#target_action_map)
      - [TARGET\_ACTIONS](#target_actions)
      - [Mapping Flow](#mapping-flow)
    - [Output Format](#output-format)
    - [Note](#note)

---

## VLM Annotation (data_annotation.py)

### Setup

Ensure you are in the `vlmdatagen` directory when running the script.

```sh
cd vlmdatagen
```

Install any necessary dependencies, if not already installed.

### Configuration

Choose one of the available API models for annotation:
- `zhipuai`
- `qwen`
- `gpt-4o` (Azure OpenAI)

Set up the necessary API key for your chosen model. Add the API key as an environment variable.

### Example Command

```bash
python data_annotation.py \
    --metadata_root /data/ceph/data/nuplan/ann_files/nuplan_test \
    --save_root /data/ceph/data/nuplan/dataset/vlm_ann_data \
    --nuplan_root /data/ceph/data/nuplan/dataset\
    --raster_path /data/ceph/data/nuplan/dataset/raster/gt \
    --text_path /data/ceph/data/nuplan/cache/text \
    --nuplan_sensor_root /data/ceph \
    --downsample_rate 10 \
    --model gpt-4o \
    --use_multiprocessing
```

### Options

- `--metadata_root`: Path to the root directory of metadata files to be annotated.
- `--save_root`: Directory where annotated data will be saved.
- `--model`: Choose the model to use for annotation (default: `gpt-4o`).
- `--nuplan_root`: Path to the NuPlan dataset root.
- `--raster_path`: Path to the raster image root for scene analysis.
- `--text_path`: Cache Path to save text-based annotations.
- `--nuplan_sensor_root`: Path to NuPlan sensor data root.
- `--use_multiprocessing`: Use multiprocessing to speed up the annotation process.
- `--downsample_rate`: Set a frame index gap for downsampling during annotation.

### API Keys and Configuration

For each model, set up API keys as follows:

#### ZhipuAI
```sh
export ZHIPUAI_API_KEY=<YOUR_API_KEY>
```

#### Qwen
```sh
export DASHSCOPE_API_KEY=<YOUR_API_KEY>
```

#### Azure OpenAI (for `gpt-4o`)
```sh
export AZURE_OPENAI_API_KEY=<YOUR_API_KEY>
```

*For Azure, you must also provide the endpoint in `api/config.py` by setting the `AZURE_ENDPOINT` variable.*

```python
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT"
```

> Note: OpenAI is restricted in some regions. Azure OpenAI is recommended as an alternative if OpenAI access is unavailable.

**In case OpenAI Key is available, using OPENAI API KEY is recommnded**

```sh
export OPENAI_API_KEY=<YOUR_API_KEY>
```

### Troubleshooting

- Ensure all environment variables are correctly set.
- Use the `--use_multiprocessing` flag to utilize all CPU cores, reducing processing time for large datasets.
- Make sure paths to NuPlan dataset and raster data are accurate to avoid file not found errors.

--- 

For more details on the NuPlan dataset or API usage, refer to the official documentation.

## Annotation Merger (merge_ann2pickle.py)


This script merges VLM (Vision-Language Model) annotations with existing pickle files containing frame data. It processes temporal meta-actions and assigns them to corresponding frames while maintaining scene continuity.

### Features

- Merges annotations from an Arrow dataset with existing pickle files
- Handles various duration formats (seconds, minutes, ranges)
- Supports multiprocessing for faster processing
- Maintains scene continuity by segmenting continuous frames
- Generates unique scene tokens for frame segments
- Includes comprehensive logging for debugging and monitoring

### Usage

```bash
python merge_ann2pickle.py \
  --index_root_path /path/to/arrow/dataset \
  --pkl_dir /path/to/pickle/files \
  --output_dir /path/to/output \
  --use_multiprocessing True \
  --num_processes 100 \
  --overwrite False
```

**Arguments**

- `--index_root_path`: Path to the directory containing Arrow dataset files with annotations
- `--pkl_dir`: Directory containing the original pickle files with frame data
- `--output_dir`: Directory where merged pickle files will be saved
- `--use_multiprocessing`: Enable parallel processing (True/False)
- `--num_processes`: Number of processes to use for parallel processing
- `--overwrite`: Whether to overwrite existing files in the output directory

### Processing Details

1. **Data Loading**:
   - Loads annotations from Arrow dataset
   - Loads frame data from pickle files

2. **Frame Processing**:
   - Maps meta-actions to frames based on timestamps
   - Validates duration of actions
   - Assigns meta-actions to corresponding frames

3. **Scene Management**:
   - Groups continuous frames into segments
   - Generates unique scene tokens for each segment
   - Maintains temporal consistency

4. **Output**:
   - Saves processed frames with merged annotations as pickle files
   - Maintains original frame structure with added meta-action information


### Notes

- Frame timestamps are expected to be at 0.5-second intervals
- Total duration for meta-actions should typically be 8 seconds
- Scene tokens are generated using UUID5 with DNS namespace



## Post-Processing (lwad/vlannotator/post_processing.py)

This script processes and combines annotated pickle files containing driving actions, mapping them to standardized meta-actions and target categories. It performs several key operations:

### Features
- Combines multiple annotated pickle files into a single dataset
- Handles exceptions in LLM annotations through standardized mapping
- Maps ground truth actions to predefined meta-actions
- Converts actions to one-hot encoded vectors for VAD (Visual Action Detection)
- Preserves original commands while adding new mappings

### Meta-Action Categories
| **Category**          | **Meta-actions**                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| **Speed-control actions** | Speed up, Slow down, Slow down rapidly, Go straight slowly, Go straight at a constant speed, Stop, Wait, Reverse |
| **Turning actions**    | Turn left, Turn right, Turn around                                                                  |
| **Lane-control actions** | Change lane to the left, Change lane to the right, Shift slightly to the left, Shift slightly to the right      |


### Usage
```bash
python post_processing.py [--input_dir INPUT_DIR] [--output_path OUTPUT_PATH]
```

**Arguments**

- `--input_dir`: Directory containing the input pickle files
- `--output_path`: Path for the output combined pickle file

### Constants and Mappings

The script uses several predefined sets and dictionaries to standardize and map driving actions:

#### META_ACTIONS
A set of standardized meta-actions that serve as the base vocabulary for all driving commands:
```python
META_ACTIONS = {
    # Speed control actions
    "speed up", "slow down", "slow down rapidly", 
    "go straight slowly", "go straight at a constant speed", 
    "stop", "wait", "reverse",
    
    # Turning actions
    "turn left", "turn right", "turn around",
    
    # Lane change actions
    "change lane to the left", "change lane to the right",
    "shift slightly to the left", "shift slightly to the right"
}
```

#### EXCEPTION_MAP
A comprehensive dictionary that handles variations and non-standard descriptions from LLM annotations. Examples include:
```python
EXCEPTION_MAP = {
    # Speed Control variations
    "accelerate": "speed up",
    "maintain constant speed": "go straight at a constant speed",
    
    # Turn Action variations
    "turn slightly left": "turn left",
    "turn sharply right": "turn right",
    
    # Lane Change variations
    "slight left shift": "shift slightly to the left",
    "shift to the right lane": "change lane to the right"
    # ... many more mappings
}
```

#### TARGET_ACTION_MAP
Maps meta-actions to five fundamental driving categories used by the VAD model:
```python
TARGET_ACTION_MAP = {
    # Speed control -> FORWARD
    "speed up": "FORWARD",
    "slow down": "FORWARD",
    # ... other speed controls
    
    # Turning -> LEFT/RIGHT
    "turn left": "LEFT",
    "turn right": "RIGHT",
    
    # Lane changes -> CHANGE_LANE_LEFT/RIGHT
    "change lane to the left": "CHANGE_LANE_LEFT",
    "change lane to the right": "CHANGE_LANE_RIGHT"
}
```

#### TARGET_ACTIONS
The final set of action categories used for one-hot encoding:
```python
TARGET_ACTIONS = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "CHANGE_LANE_LEFT",
    "CHANGE_LANE_RIGHT"
]
```

####  Mapping Flow
The script processes actions through these constants in the following order:
1. Raw LLM annotation → Check if in `META_ACTIONS`
2. If not in `META_ACTIONS` → Look up in `EXCEPTION_MAP`
3. Meta-action → Map to target category using `TARGET_ACTION_MAP`
4. Target category → Convert to one-hot vector using `TARGET_ACTIONS` order

### Output Format
The script generates a pickle file containing:
- Combined annotated data with standardized actions
- One-hot encoded action vectors
- Original commands preserved in ``'gt_ego_fut_cmd_old'``

### Note
The script includes extensive exception handling for various action descriptions, mapping them to standardized meta-actions. If an action cannot be mapped, it defaults to "go straight at a constant speed".