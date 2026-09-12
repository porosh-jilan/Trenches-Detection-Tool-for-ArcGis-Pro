# -*- coding: utf-8 -*-
"""
TrenchML Detection v11 - ArcMap 10.8 Python Toolbox
4-level U-Net CNN trained on 10 sites (6 original + 4 Zoha sites).
"""
from __future__ import print_function

import os
import sys

import arcpy

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import trench_ml_core_v11 as trench_ml_core
try:
    reload(trench_ml_core)
except Exception:
    pass


def _param(display_name, name, datatype, parameter_type, direction):
    return arcpy.Parameter(
        displayName=display_name,
        name=name,
        datatype=datatype,
        parameterType=parameter_type,
        direction=direction)


class Toolbox(object):
    def __init__(self):
        self.label = "TrenchML Detection v11 CNN ArcMap 10.8"
        self.alias = "TrenchML_v11_CNN"
        self.tools = [TrainTrenchMLModel, DetectTrenchesML,
                      DetectTrenchesCNN]


class TrainTrenchMLModel(object):
    def __init__(self):
        self.label = "Train Trench ML Model from DEM and Ground Truth"
        self.description = (
            "Train a supervised trench classifier from a 1 m DEM, manual "
            "ground-truth trench polylines, and optional hard negative "
            "candidate lines. ArcPy + NumPy only.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []
        p0 = _param("Input 1m DEM Raster", "in_dem",
                             "DERasterDataset", "Required", "Input")
        params.append(p0)
        p1 = _param("Ground Truth Trench Polylines", "ground_truth",
                             "DEFeatureClass", "Required", "Input")
        params.append(p1)
        p2 = _param("Optional Hard Negative False Lines", "hard_negative_lines",
                             "DEFeatureClass", "Optional", "Input")
        params.append(p2)
        p3 = _param("Output Model File (.npz)", "out_model",
                             "DEFile", "Required", "Output")
        p3.value = "trench_ml_model_v2.npz"
        params.append(p3)
        p4 = _param("Positive Buffer Around Ground Truth (m)",
                             "positive_buffer_m", "GPDouble",
                             "Optional", "Input")
        p4.value = 2.5
        params.append(p4)
        p5 = _param("Ignore Buffer Around Ground Truth (m)",
                             "ignore_buffer_m", "GPDouble",
                             "Optional", "Input")
        p5.value = 9.0
        params.append(p5)
        p6 = _param("Maximum Training Samples", "max_samples",
                             "GPLong", "Optional", "Input")
        p6.value = 260000
        params.append(p6)
        p7 = _param("Training Tile Size (pixels)", "tile_size_px",
                             "GPLong", "Optional", "Input")
        p7.value = 900
        params.append(p7)
        p8 = _param("Model Iterations", "iterations",
                             "GPLong", "Optional", "Input")
        p8.value = 340
        params.append(p8)
        return params

    def execute(self, parameters, messages):
        def val(i, default):
            return float(parameters[i].value) if parameters[i].value is not None else default

        def ival(i, default):
            return int(parameters[i].value) if parameters[i].value is not None else default

        out_model = parameters[3].valueAsText
        if not out_model.lower().endswith(".npz"):
            out_model += ".npz"
        trench_ml_core.train_model(
            parameters[0].valueAsText,
            parameters[1].valueAsText,
            out_model,
            arcpy,
            hard_negative_lines=parameters[2].valueAsText,
            positive_buffer_m=val(4, 2.5),
            ignore_buffer_m=val(5, 9.0),
            max_samples=ival(6, 260000),
            tile_size_px=ival(7, 900),
            iterations=ival(8, 340))
        return


class DetectTrenchesML(object):
    def __init__(self):
        self.label = "Detect Trenches with Trained ML Model (old logistic)"
        self.description = (
            "Apply a trained TrenchML .npz model to a DEM and write smooth "
            "trench centreline polylines. (Kept for backwards compatibility - "
            "for new work use the CNN tool below.)")
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []
        p0 = _param("Input 1m DEM Raster", "in_dem",
                             "DERasterDataset", "Required", "Input")
        params.append(p0)
        p1 = _param("Trained Model File (.npz)", "model_file",
                             "DEFile", "Required", "Input")
        params.append(p1)
        p2 = _param("Output Trench Polyline", "out_trenches",
                             "DEFeatureClass", "Required", "Output")
        params.append(p2)
        p3 = _param("Probability Threshold", "probability_threshold",
                             "GPDouble", "Optional", "Input")
        p3.value = 0.56
        params.append(p3)
        p4 = _param("Minimum Depression Depth Guard (m)",
                             "min_depth_m", "GPDouble", "Optional", "Input")
        p4.value = 0.01
        params.append(p4)
        p5 = _param("Minimum Output Line Length (m)",
                             "min_length_m", "GPDouble", "Optional", "Input")
        p5.value = 8.0
        params.append(p5)
        p6 = _param("Tile Size (pixels)", "tile_size_px",
                             "GPLong", "Optional", "Input")
        p6.value = 900
        params.append(p6)
        p7 = _param("Tile Overlap (pixels)", "tile_overlap_px",
                             "GPLong", "Optional", "Input")
        p7.value = 48
        params.append(p7)
        p8 = _param("Close Small Gaps (pixels)", "gap_close_px",
                             "GPLong", "Optional", "Input")
        p8.value = 3
        params.append(p8)
        p9 = _param("Join Nearby Segments (m)", "join_gap_m",
                             "GPDouble", "Optional", "Input")
        p9.value = 20.0                # v11.4: was 10 - cuts endpoint miss
        params.append(p9)
        p10 = _param("Simplify/Smooth Output (m)",
                              "simplify_tolerance_m", "GPDouble",
                              "Optional", "Input")
        p10.value = 1.5
        params.append(p10)
        p11 = _param("Final Completion Join Gap (m)",
                              "final_join_gap_m", "GPDouble",
                              "Optional", "Input")
        p11.value = 28.0               # v11.4: was 18 - paired with join_gap
        params.append(p11)
        p12 = _param("Calibrated Final Minimum Length (m)",
                              "final_min_length_m", "GPDouble",
                              "Optional", "Input")
        p12.value = 25.0
        params.append(p12)
        p13 = _param("Calibrated Final Minimum Mean Score",
                              "final_min_prob_mean", "GPDouble",
                              "Optional", "Input")
        p13.value = 0.10
        params.append(p13)
        p14 = _param("Calibrated Final Minimum Max Score",
                              "final_min_prob_max", "GPDouble",
                              "Optional", "Input")
        p14.value = 0.10
        params.append(p14)
        p15 = _param("Calibrated Final Minimum QScore",
                              "final_min_qscore", "GPDouble",
                              "Optional", "Input")
        p15.value = 20.0
        params.append(p15)
        p16 = _param("Extend Endpoints (v3)", "extend_endpoints",
                              "GPBoolean", "Optional", "Input")
        p16.value = True
        params.append(p16)
        p17 = _param("Endpoint Extend Depth Threshold (m)",
                              "extend_depth_m", "GPDouble",
                              "Optional", "Input")
        p17.value = 0.015         # v11.5: was 0.020 - small extra endpoint gain
        params.append(p17)
        p18 = _param("Maximum Endpoint Extend (m)",
                              "max_extend_m", "GPDouble",
                              "Optional", "Input")
        p18.value = 40.0          # v11.3: was 30.0
        params.append(p18)
        p19 = _param("Endpoint Trim Depth Threshold (m)",
                              "trim_depth_m", "GPDouble",
                              "Optional", "Input")
        p19.value = 0.025
        params.append(p19)
        p20 = _param("Maximum Endpoint Trim (m)",
                              "max_trim_m", "GPDouble",
                              "Optional", "Input")
        p20.value = 6.0
        params.append(p20)
        p21 = _param("Endpoint Fail Step Tolerance",
                              "endpoint_fail_steps", "GPLong",
                              "Optional", "Input")
        p21.value = 5             # v11.3: was 3 - allows extension through faint stretches
        params.append(p21)
        p22 = _param("Centerline Lock Radius (m, v3 fallback)",
                              "centerline_lock_radius_m", "GPDouble",
                              "Optional", "Input")
        p22.value = 1.0
        params.append(p22)
        p23 = _param("Width-Midpoint Snap (v3.1)", "widmid_snap",
                              "GPBoolean", "Optional", "Input")
        p23.value = True
        params.append(p23)
        p24 = _param("Width-Mid Scan Radius (m)", "widmid_radius_m",
                              "GPDouble", "Optional", "Input")
        p24.value = 4.0
        params.append(p24)
        p25 = _param("Width-Mid Max Shift (m)", "widmid_max_shift_m",
                              "GPDouble", "Optional", "Input")
        p25.value = 0.5
        params.append(p25)
        p26 = _param("Width-Mid Wall Threshold (fraction of depth)",
                              "widmid_wall_frac", "GPDouble",
                              "Optional", "Input")
        p26.value = 0.4
        params.append(p26)
        p27 = _param("Human-Style Centerline Pass (v3.1)",
                              "human_style_pass", "GPBoolean",
                              "Optional", "Input")
        p27.value = True
        params.append(p27)
        p28 = _param("Human-Style Vertex Spacing (m)",
                              "human_style_spacing_m", "GPDouble",
                              "Optional", "Input")
        p28.value = 5.0
        params.append(p28)
        p29 = _param("Human-Style Snap Radius (m)",
                              "human_style_lock_radius_m", "GPDouble",
                              "Optional", "Input")
        p29.value = 2.0
        params.append(p29)
        p30 = _param("Cell-Centre Snap (v3.2, recommended ON)",
                              "human_style_cellcentre", "GPBoolean",
                              "Optional", "Input")
        p30.value = True
        params.append(p30)
        p31 = _param("Split at Breaks (v3.3, recommended ON)",
                              "split_at_breaks", "GPBoolean",
                              "Optional", "Input")
        p31.value = True
        params.append(p31)
        p32 = _param("Split: Min Cross-Section Depth (m)",
                              "split_min_depth_m", "GPDouble",
                              "Optional", "Input")
        p32.value = 0.04
        params.append(p32)
        p33 = _param("Split: Min Break Length (m)",
                              "split_min_break_length_m", "GPDouble",
                              "Optional", "Input")
        p33.value = 5.0
        params.append(p33)
        p34 = _param("Split: Min Fragment Length (m)",
                              "split_min_fragment_length_m", "GPDouble",
                              "Optional", "Input")
        p34.value = 8.0
        params.append(p34)
        p35 = _param("Split: Cross-Section Half-Width (m)",
                              "split_half_width_m", "GPDouble",
                              "Optional", "Input")
        p35.value = 1.5
        params.append(p35)
        p36 = _param("Remove Duplicate Lines (v9, recommended ON)",
                              "remove_duplicates", "GPBoolean",
                              "Optional", "Input")
        p36.value = True
        params.append(p36)
        p37 = _param("Duplicate: Overlap Distance (m)",
                              "duplicate_distance_m", "GPDouble",
                              "Optional", "Input")
        p37.value = 3.0
        params.append(p37)
        p38 = _param("Duplicate: Cover Ratio (0-1)",
                              "duplicate_cover_ratio", "GPDouble",
                              "Optional", "Input")
        p38.value = 0.6
        params.append(p38)
        p39 = _param("Add Result to Current Map", "add_to_map",
                              "GPBoolean", "Optional", "Input")
        p39.value = True
        params.append(p39)
        return params

    def execute(self, parameters, messages):
        def val(i, default):
            return float(parameters[i].value) if parameters[i].value is not None else default

        def ival(i, default):
            return int(parameters[i].value) if parameters[i].value is not None else default

        def bval(i, default):
            return bool(parameters[i].value) if parameters[i].value is not None else default

        out_fc = trench_ml_core.detect_trenches_ml(
            parameters[0].valueAsText,
            parameters[1].valueAsText,
            parameters[2].valueAsText,
            arcpy,
            probability_threshold=val(3, 0.56),
            min_depth_m=val(4, 0.01),
            min_length_m=val(5, 8.0),
            tile_size_px=ival(6, 900),
            tile_overlap_px=ival(7, 48),
            gap_close_px=ival(8, 3),
            join_gap_m=val(9, 20.0),
            simplify_tolerance_m=val(10, 1.5),
            final_join_gap_m=val(11, 28.0),
            final_min_length_m=val(12, 25.0),
            final_min_prob_mean=val(13, 0.10),
            final_min_prob_max=val(14, 0.10),
            final_min_qscore=val(15, 20.0),
            extend_endpoints=bval(16, True),
            extend_depth_m=val(17, 0.015),
            max_extend_m=val(18, 40.0),
            trim_depth_m=val(19, 0.025),
            max_trim_m=val(20, 6.0),
            endpoint_fail_steps=ival(21, 5),
            centerline_lock_radius_m=val(22, 1.0),
            widmid_snap=bval(23, False),
            widmid_radius_m=val(24, 4.0),
            widmid_max_shift_m=val(25, 0.5),
            widmid_wall_frac=val(26, 0.4),
            human_style_pass=bval(27, True),
            human_style_spacing_m=val(28, 5.0),
            human_style_lock_radius_m=val(29, 2.0),
            human_style_cellcentre=bval(30, True),
            split_at_breaks=bval(31, True),
            split_min_depth_m=val(32, 0.04),
            split_min_break_length_m=val(33, 5.0),
            split_min_fragment_length_m=val(34, 8.0),
            split_half_width_m=val(35, 1.5),
            remove_duplicates=bval(36, True),
            duplicate_distance_m=val(37, 3.0),
            duplicate_cover_ratio=val(38, 0.6),
            overwrite=True)
        add_to_map = bval(39, True)
        if add_to_map:
            try:
                arcpy.MakeFeatureLayer_management(out_fc, "Detected_Trenches_ML_v11")
                arcpy.AddMessage("Result layer added as Detected_Trenches_ML_v11")
            except Exception as exc:
                arcpy.AddWarning("Could not add layer to map: " + str(exc))
        return


class DetectTrenchesCNN(object):
    """v11: CNN (U-Net) trench detector trained on 10 sites. Uses the
    bundled pre-trained cnn_unet_weights.npz - no separate model training
    needed."""

    def __init__(self):
        self.label = "Detect Trenches with CNN (v11, recommended)"
        self.description = (
            "Apply the trained U-Net CNN (10-site training set) to a 1 m DEM "
            "and write trench centreline polylines. Uses the bundled "
            "cnn_unet_weights.npz; pure NumPy inference, no PyTorch needed.")
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []
        p0 = _param("Input 1m DEM Raster", "in_dem",
                    "DERasterDataset", "Required", "Input")
        params.append(p0)
        p1 = _param("Output Trench Polyline", "out_trenches",
                    "DEFeatureClass", "Required", "Output")
        params.append(p1)
        p2 = _param("CNN Probability Threshold", "prob_threshold",
                    "GPDouble", "Optional", "Input")
        p2.value = 0.45
        params.append(p2)
        p3 = _param("Minimum Output Line Length (m)", "min_length_m",
                    "GPDouble", "Optional", "Input")
        p3.value = 8.0
        params.append(p3)
        p4 = _param("Calibrated Final Minimum Length (m)",
                    "final_min_length_m", "GPDouble", "Optional", "Input")
        p4.value = 25.0
        params.append(p4)
        p5 = _param("Tile Size (pixels)", "tile_size_px",
                    "GPLong", "Optional", "Input")
        p5.value = 1200
        params.append(p5)
        p6 = _param("Maximum Endpoint Extend (m)", "max_extend_m",
                    "GPDouble", "Optional", "Input")
        p6.value = 40.0           # v11.3: was 30.0
        params.append(p6)
        p7 = _param("Split at Breaks", "split_at_breaks",
                    "GPBoolean", "Optional", "Input")
        p7.value = True
        params.append(p7)
        p8 = _param("Remove Duplicate Lines", "remove_duplicates",
                    "GPBoolean", "Optional", "Input")
        p8.value = True
        params.append(p8)
        p9 = _param("CNN Weights File (.npz, blank = bundled)",
                    "cnn_weights", "DEFile", "Optional", "Input")
        params.append(p9)
        p10 = _param("Add Result to Current Map", "add_to_map",
                     "GPBoolean", "Optional", "Input")
        p10.value = True
        params.append(p10)
        # v11.1: junction-aware tracing (fixes bridging of touching trenches)
        p11 = _param("Junction-Aware Tracing (v11.1, recommended ON)",
                     "junction_aware", "GPBoolean", "Optional", "Input")
        p11.value = True
        params.append(p11)
        p12 = _param("Junction Max Through-Turn (deg)",
                     "junction_max_turn_deg", "GPDouble", "Optional", "Input")
        p12.value = 60.0
        params.append(p12)
        p13 = _param("Final-Join Max Angle (deg, lower = fewer bridges)",
                     "final_join_max_angle_deg", "GPDouble",
                     "Optional", "Input")
        p13.value = 60.0
        params.append(p13)
        p14 = _param("Test-Time Augmentation (0=off fast, 4=balanced, 8=max)",
                     "cnn_tta", "GPLong", "Optional", "Input")
        p14.value = 0
        params.append(p14)
        p15 = _param("Extension CNN-Prob Threshold (v11.6, lower = longer tails)",
                     "ext_prob_threshold", "GPDouble", "Optional", "Input")
        p15.value = 0.30
        params.append(p15)
        p16 = _param("Second-Model Recall Boost (v11.7, ~2x slower, +recall)",
                     "recall_boost", "GPBoolean", "Optional", "Input")
        p16.value = True
        params.append(p16)
        p17 = _param("River Mode (v12.4) - loads the river model + river "
                     "geometry: one unbroken line per river, strong dedup, "
                     "declutter. ON for rivers/roads, OFF for narrow trenches",
                     "merge_parallel", "GPBoolean", "Optional", "Input")
        p17.value = False
        params.append(p17)
        p18 = _param("Merge: Max Feature Width (m)", "merge_max_gap_m",
                     "GPDouble", "Optional", "Input")
        p18.value = 40.0
        params.append(p18)
        p19 = _param("Valley-Following Snap (v12.2, tighter centre-lines)",
                     "thalweg_snap", "GPBoolean", "Optional", "Input")
        p19.value = True
        params.append(p19)
        p20 = _param("Road Mode (v12.8) - road-side ditches: STOP each trench "
                     "at a pipe/culvert and keep parallel ditches separate. ON "
                     "for road-side trenches, OFF for fields/rivers",
                     "road_mode", "GPBoolean", "Optional", "Input")
        p20.value = False
        params.append(p20)
        p21 = _param("Second-Model Ensemble (v12.9) - an MLP partner refines the "
                     "CNN with shape features for a smarter, more accurate "
                     "decision. Recommended ON.",
                     "second_model", "GPBoolean", "Optional", "Input")
        p21.value = True
        params.append(p21)
        p22 = _param("Steep Terrain Mode (v12.9.7) - loads a CNN fine-tuned on "
                     "steep/hilly DEMs (drainage channels). Catches faint/parallel "
                     "trenches the default model misses on steep ground (measured "
                     "held-out recall ~64% -> ~80%). ON for steep/hilly DEMs, OFF "
                     "for flat agricultural fields.",
                     "steep_mode", "GPBoolean", "Optional", "Input")
        p22.value = False
        params.append(p22)
        p23 = _param("Sharp Road Centerline (v12.9.8) - loads a CNN fine-tuned "
                     "(thin labels on 100%-accurate Trainin1/3 + your road "
                     "corrections) to fire a NARROW ridge on the ditch centre so "
                     "the line sits truer on shallow road-side ditches (measured: "
                     ">1m drift 14% -> 11%, median 0.51 -> 0.44 m). ON for road-side "
                     "ditches, OFF for fields/rivers/steep.",
                     "sharp_mode", "GPBoolean", "Optional", "Input")
        p23.value = False
        params.append(p23)
        p24 = _param("Half-pixel Grid Correction (v12.9.10) - turn ON for DEMs "
                     "exported from SAGA / .sdat. Those carry a world-file (.tfw) "
                     "that sits half a pixel off the GeoTIFF grid, so every line "
                     "lands ~0.5 m NW of the true channel. Measured on your DEM 1 "
                     "& DEM 2: enabling this brings the lines onto the digitised "
                     "truth (median 0.75 -> 0.43 m). Leave OFF for standard "
                     "GeoTIFFs (no .tfw).",
                     "reg_half_pixel", "GPBoolean", "Optional", "Input")
        p24.value = False
        params.append(p24)
        return params

    def execute(self, parameters, messages):
        def val(i, d):
            return float(parameters[i].value) if parameters[i].value is not None else d

        def ival(i, d):
            return int(parameters[i].value) if parameters[i].value is not None else d

        def bval(i, d):
            return bool(parameters[i].value) if parameters[i].value is not None else d

        river_on = bval(17, False)
        steep_on = bval(22, False) and not river_on   # river takes precedence
        sharp_on = bval(23, False) and not river_on and not steep_on

        weights = parameters[9].valueAsText
        if not weights:
            # v12.4 River / v12.9.7 Steep / v12.9.8 Sharp-centerline each load a
            # dedicated 5ch model; the flat agricultural model stays UNTOUCHED.
            if river_on:
                weights = os.path.join(_THIS_DIR, "cnn_unet_river_weights.npz")
            elif steep_on:
                weights = os.path.join(_THIS_DIR, "cnn_unet_steep_weights.npz")
            elif sharp_on:
                weights = os.path.join(_THIS_DIR, "cnn_unet_sharp_weights.npz")
            else:
                weights = os.path.join(_THIS_DIR, "cnn_unet_weights.npz")
        # v12.7: agricultural + steep models are 5-channel -> 5ch norm; the river
        # model is 4-channel -> 4ch norm. Inference auto-slices to the model's
        # channel count.
        norm = os.path.join(
            _THIS_DIR,
            "cnn_norm_stats_4ch.npz" if river_on
            else "cnn_norm_stats.npz")

        # Shared options.
        kw = dict(
            probability_threshold=val(2, 0.45),
            min_length_m=val(3, 8.0),
            final_min_length_m=val(4, 25.0),
            tile_size_px=ival(5, 1200),
            max_extend_m=val(6, 40.0),
            remove_duplicates=bval(8, True),
            junction_aware=bval(11, True),
            junction_max_turn_deg=val(12, 60.0),
            cnn_tta=ival(14, 0),
            ext_prob_threshold=val(15, 0.30),
            merge_max_gap_m=val(18, 40.0),
            thalweg_snap=bval(19, True),
            road_mode=bval(20, False),
            second_model=bval(21, True),
            reg_half_pixel=bval(24, False),   # v12.9.10 SAGA/.sdat half-pixel fix
            overwrite=True)

        # v12.9.10: gentle nudge - if the input DEM ships a world file (.tfw) and
        # the user has NOT enabled the half-pixel correction, point it out. SAGA/
        # .sdat exports put the .tfw half a pixel off the GeoTIFF grid, which shifts
        # every line ~0.5 m off the true channel (measured on DEM 1 & DEM 2).
        if not bval(24, False):
            try:
                dem_path = parameters[0].valueAsText
                base = os.path.splitext(dem_path)[0]
                if any(os.path.exists(base + e) for e in
                       (".tfw", ".tifw", ".wld", ".sdat", ".sgrd")):
                    arcpy.AddMessage(
                        "NOTE: this DEM has a world file (.tfw/.sdat). If the lines "
                        "sit ~0.5 m off the channel, enable 'Half-pixel Grid "
                        "Correction' (v12.9.10).")
            except Exception:
                pass

        if river_on:
            # v12.4 river geometry profile, measured best on the held-out
            # Training-3 river crop (pieces/river 2.42 -> 1.46, single-piece
            # rivers 7 -> 17 / 24, double-lines 45 -> 3, coverage held at 90%):
            #  * wide collinear final join (100 m) bridges mid-channel breaks
            #  * tight join angle (40 deg) so tributaries are NOT mixed in
            #  * break-splitting off (rivers are one continuous feature)
            #  * isolated-fragment declutter runs in the core (gated on river)
            #  * second-model recall boost off (that model is agricultural)
            kw.update(merge_parallel=True, split_at_breaks=False,
                      final_join_max_angle_deg=40.0, join_gap_m=40.0,
                      final_join_gap_m=100.0, recall_boost=False,
                      ridge_cut=False)  # a river is ONE unbroken line
        else:
            kw.update(merge_parallel=False, split_at_breaks=bval(7, True),
                      final_join_max_angle_deg=val(13, 60.0),
                      recall_boost=bval(16, True))

        if steep_on:
            # v12.9.7 Steep Mode geometry (measured best on the held-out steep
            # box C): the steep CNN's extra detections survive better with the
            # min-length set to the final 25 m, and the second-model MLP is OFF
            # (it was trained on the DEFAULT model's probability distribution).
            kw.update(second_model=False, min_length_m=25.0)
        if sharp_on:
            # v12.9.8/9 Sharp-centerline model: 2nd-model MLP OFF (tied to the
            # default prob); centre on the CNN prob-ridge CENTROID instead of the
            # elevation dip (shallow road ditches have no usable dip). Measured on
            # the user's road corrections: drift 0.50 -> 0.25 m, >1m 12% -> 0%.
            kw.update(second_model=False, prob_centroid_thalweg=True)

        # ArcGIS Pro (64-bit, Python 3): no 2 GB memory limit, so the WHOLE
        # DEM is processed in a single pass - identical to the original tool,
        # no block splitting. Large DEMs work as long as the machine has enough
        # free RAM (roughly 4-8 GB for a few-hundred-million-pixel raster).
        out_fc = trench_ml_core.detect_trenches_cnn(
            parameters[0].valueAsText,
            weights,
            norm,
            parameters[1].valueAsText,
            arcpy,
            **kw)
        if bval(10, True):
            try:
                arcpy.MakeFeatureLayer_management(
                    out_fc, "Detected_Trenches_CNN_v11")
                arcpy.AddMessage(
                    "Result layer added as Detected_Trenches_CNN_v11")
            except Exception as exc:
                arcpy.AddWarning("Could not add layer to map: " + str(exc))
        return
