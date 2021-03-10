def get_gripper_velocities(diagram, simulator, systems):
    sim_context = simulator.get_mutable_context()
    pid_context = diagram.GetMutableSubsystemContext(
                          systems["pid"], sim_context)
    sp_context = diagram.GetMutableSubsystemContext(
                          systems["sp_control"], sim_context)
    station_context = diagram.GetMutableSubsystemContext(
                          systems["station"], sim_context)
    velocities = []
    for arm_name in ["left", "right"]:
        state = systems["station"].get_model_state(station_context, arm_name)
        velocities.extend(state["velocity"])
    return velocities
