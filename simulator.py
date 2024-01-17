import Quad_Dynamics
import Runge_Kutta
import matplotlib.pyplot as plt
import numpy as np
import PID_controller
import Learning4_test_code
import utiles
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def __plot_rec(state_rec):
    component_names = ['Position', 'Velocity', 'Euler Angles', 'Angular Velocity']
    axis_name = ['X', 'Y', 'Z']

    timestep = list(range(len(state_rec)))

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for i, component in enumerate(zip(*state_rec)):  # Transpose the state_rec list
        row = i // 2
        col = i % 2
        axs[row, col].set_title(f'{component_names[i]} Over Time')

        if component_names[i] == 'Euler Angles':
            # Convert quaternion to Euler angles using your Q function
            euler_angles_values = [utiles.quaternion_to_euler(quat) for quat in component]
            for j in range(3):
                axs[row, col].plot(timestep, [angles[j] for angles in euler_angles_values],
                                   label=f'Euler Angle {axis_name[j]}')
        else:
            for j in range(len(component[0])):
                component_values_i = [comp[j] for comp in component]
                axs[row, col].plot(timestep, component_values_i, label=f'{component_names[i]} {axis_name[j]}-axis')

        axs[row, col].set_xlabel('Time [0.01sec/step]')
        axs[row, col].set_ylabel(f'{component_names[i]}')
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()

def __3D_plot(state_rec):
    x_positions = [step[0][0] for step in state_rec]
    y_positions = [step[0][1] for step in state_rec]
    z_positions = [step[0][2] for step in state_rec]

    # 3D 플롯 생성
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], label='Drone Position', linewidth=2)
    scatters = ax.scatter(x_positions[0], y_positions[0], z_positions[0], s=30, c='green', marker='o', label='Start Point')
    scatters2 = ax.scatter(x_positions[-1], y_positions[-1], z_positions[-1], s=30, c='red', marker='o', label='End Point')

    # 축의 스케일을 -5에서 5로 지정
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Drone Position Over Time')
    ax.legend()

    # 초기화 함수
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line, scatters, scatters2

    # 각 프레임마다 호출되는 함수
    def update(frame):
        x = x_positions[:frame]
        y = y_positions[:frame]
        z = z_positions[:frame]

        line.set_data(x, y)
        line.set_3d_properties(z)

        return line, scatters, scatters2

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=len(x_positions), init_func=init, blit=True, interval=1,repeat=False)

    # 플롯 표시
    plt.show()

#state parameters
#p_ref = [0,0,0]

p = [-1,5,3]
vel = [0,0,0]
quat= [1,0,0,0]
omega = [0,0,0]


state_curr = [p,vel,quat,omega]
state_rec = []

####여길 돌려야함

count = 0

while count < 2000:
    #output = SNN_generator.SNN(state_curr)
    output = Learning4_test_code.SNN(state_curr)

    Thrust, Torque = PID_controller.PID(state_curr)

    Thrust1 = [0,0,output[0]]
    Torque1 = [output[1],output[2],output[3]]
    Thrust2 = [0,0,Thrust]
    Torque2 = [Torque[0],Torque[1],Torque[2]]

    p2dot, omegadot = Quad_Dynamics.Quad(Thrust1, Torque1, state_curr)
    #p2dot, omegadot = Quad_Dynamics.Quad(Thrust2,Torque2,state_curr)

    pd, veld, quatd, omegad = Runge_Kutta.RK(p2dot, omegadot, state_curr)
    state_next = [pd, veld, quatd, omegad]
    state_rec.append(state_next)
    state_curr = state_next

    count += 1

__plot_rec(state_rec)
__3D_plot(state_rec)




