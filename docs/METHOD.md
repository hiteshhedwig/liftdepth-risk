# Method

## Goal

The goal is to turn a single monocular driving camera sequence into a structured BEV perception representation.

The final system is not just a depth visualization. It tries to answer:

> Which road-relevant objects are present, where are they in top-down space, and how risky are they relative to the ego vehicle?

---

## Why not raw dense-depth BEV only?

A dense monocular depth map can be projected into BEV, but dynamic objects such as buses and cars often become curved or smeared structures. This happens because each object pixel has slightly different predicted depth, and projecting every pixel independently produces distorted BEV shapes.

Instead, this project uses object-aware BEV:

```text
detected object bbox
   ↓
median DA2 depth inside bbox
   ↓
project object center to BEV
   ↓
draw compact rectangular footprint
```

This produces cleaner and more interpretable road-user occupancy.

---

## Components

### 1. DA2 metric depth

Depth Anything V2 metric-depth model estimates per-pixel metric depth.

The depth map is used for:

- object distance estimation
- road-mask projection into BEV
- optional dense occupancy visualization

### 2. Road segmentation

SegFormer fine-tuned on Cityscapes is used for road and sidewalk masks.

The road mask is used as a context gate:

```text
cars / buses / trucks must touch the dilated road mask
pedestrians are kept if near road or sidewalk
```

This reduces false detections from parked or irrelevant side objects.

### 3. YOLO tracking

YOLO tracks the important road-user classes:

```text
person
car
bus
truck
```

The system intentionally ignores less useful classes for this demo.

### 4. Stable project IDs

YOLO IDs can flicker when class labels fluctuate, for example:

```text
car → truck → car
```

The final script includes a post-processing stable-ID layer that merges recent detections using:

- superclass: vehicle/person
- bbox IoU
- normalized bbox center distance
- short max-age memory

For display, the most common class label is used.

### 5. Object projection

For each tracked object:

```text
bbox lower-middle region → median depth
bbox center-bottom point + KITTI intrinsics → camera 3D
camera X/Z → BEV side/forward coordinates
```

KITTI rectified camera convention:

```text
X = right
Y = down
Z = forward
```

BEV convention:

```text
side = X
forward = Z
```

### 6. Object occupancy grid

Each object becomes a rectangular BEV footprint.

Approximate footprints:

```text
person: 0.8m × 0.8m
car:    2.0m × 4.2m
bus:    2.8m × 9.0m
truck:  2.8m × 7.0m
```

### 7. Temporal smoothing

Two smoothing steps are used:

```text
object position smoothing
grid smoothing
```

This reduces jitter from both detection boxes and depth estimates.

### 8. Risk map

Risk is computed from object occupancy with distance and ego-path weighting:

```text
risk = occupancy × closeness × center-corridor-weight
```

Object-level risk additionally considers:

- distance
- lateral position
- object class
- approach motion
- lateral movement toward ego center

---

## Interactive HTML demo

The final HTML demo shows six synchronized panels:

```text
RGB + live boxes
DA2 depth map
road / sidewalk segmentation
projected road BEV
object occupancy grid
risk heatmap
```

Boxes and labels are drawn by the browser from JSON metadata, not burned into the images.
