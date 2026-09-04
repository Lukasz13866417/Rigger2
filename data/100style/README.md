# 100STYLE audit subset

The local audit uses fourteen forward-walking BVH takes and the authoritative
frame-cut table downloaded from the [100STYLE dataset page](https://www.ianxmason.com/100style/)
for the fixed-rig real-data audit. These downloaded files are not committed to Git; place them
in this directory before running the real-data audit:

- `Akimbo_FW.bvh`
- `Angry_FW.bvh`
- `ArmsBehindBack_FW.bvh`
- `ArmsBySide_FW.bvh`
- `ArmsFolded_FW.bvh`
- `Depressed_FW.bvh`
- `Elated_FW.bvh`
- `GracefulArms_FW.bvh`
- `HandsInPockets_FW.bvh`
- `Heavyset_FW.bvh`
- `LookUp_FW.bvh`
- `Neutral_FW.bvh`
- `Old_FW.bvh`
- `Proud_FW.bvh`
- `Frame_Cuts.csv`

100STYLE was created by Ian Mason, Sebastian Starke, and Taku Komura. The data
is distributed under the Creative Commons Attribution 4.0 International
license. The source files are 60 fps Xsens BVH in centimetres. Style names are
retained only as source identities; style is not a target in this experiment. Every take is
subject to the same fixed cleanliness gate. A downloaded take that fails remains documented but
is excluded from generated training/evaluation samples without loosening thresholds.
