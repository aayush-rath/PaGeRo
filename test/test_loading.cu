#include "loader.cuh"

int main() {
    Robot robot1 = load_urdf("../robots/3DOFRoboticArm.urdf");
    Robot robot2 = load_urdf("../robots/4DOFRoboticArm.urdf");
    return 0;
}