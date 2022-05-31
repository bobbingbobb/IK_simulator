# IK Simulator
ik simulator

## packages
scipy 1.7.1
ikpy 3.2.2

## File Structure
```
├─ data
│  ├─ raw_data  (put them in ssd for faster query)
│  └─ result  
├─ scripts  
│  ├─ ik_simulator  
│  ├─ data_gen  
│  ├─ runner  
│  ├─ experiments  
│  └─ others...  
├─ webots  
│  └─ ...  
```

## raw data
h5py (abandoned)
```
franka_data
├─ end_posture[8,8,9]
│  └─ pose ('[0.0 0.0 0.0]')
│     ├─ joints ('[0.0 0.0 0.0 0.0 0.0 0.0 0.0]'), split by |
│     ├─ ee_vector ('[0.0 0.0 0.0]'), split by |
│     └─ corresponding (index), split by |
├─ joints_position[]
│  └─ 7joints position[,7]  
#  ├─ ee_vector[]
#  │  └─  ee_vector ('[0.0 0.0 0.0]')
```

rtree
```
position [1,3]
   ├─ 7joints position [3,7]  
   ├─ joints [1,7]
   ├─ ee_vector [1,3]
```

## Classes
```
class Robot:
    description:
        robot infos.
    arguments:
        dh:     robot dh.
        joints: joint restricts.
    def fk:
        fk calculation.

class DataCollection
    description:
        gathering raw data.
    def without_colliding_detect:
        collecting data.

class IKTable
    description:
        provide position-to-joints table, and searching tree.
    arguments:
        searching table:  searching (nearby position) table.
        joints:           raw joints data.
        pos_table:        position-joints_chain corresponding table(end point based).
    def load_data:
        turns raw position data (end point only) into position-joints_chain corresponding table.
    def kd_tree:
        kd tree.
    def searching:
        find existing positions.
    def range_search:
        find positions in range. (kd_tree.query_ball_point)

    table_v1: build, search
    kd_tree: build, search

class IKSimulator
    description:
        work as IK.
    def find:
        find data around target.
    def get_postures:
        find all different postures(based on collected data).
    def get_all_posture:
        compute all postures on target.

    pure approx: move all joints
    vector portion: (not yet)

```

#### TODOs
ik_simulator
- class IKSimulator: how to interpolate
- class IKSimulator: if there exists the position, what to do ? (simply print out?)
- class IKSimulator: rtree

franka_cal
- dimension_portion: interpolation test (approximation?)

others
- constants management
