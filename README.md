# IK Simulator
ik simulator

## File Structure
```
├─ data  
│  ├─ raw_data  
│  └─ table  
├─ scripts  
│  ├─ ik_simulator  
│  └─ others...  
├─ webots  
│  └─ ...  
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
- class IKSimulator: kd tree

franka_cal
- dimension_portion: interpolation test (approximation?)

others
- constants management
