Demarrer la simulation :

Lancer 2 consoles

console 1 :
export LC_ALL=C; unset LANGUAGE
source /opt/ros/kinetic/setup.bash
source /home/jules/setup-opt-testrobotpkgarg.sh /opt/openrobots 1
roslaunch talos_gazebo talos_gazebo.launch

une fois que le launch s'est fait passer a la console 2

console 2 :
export LC_ALL=C; unset LANGUAGE
source /opt/ros/kinetic/setup.bash
source /home/jules/setup-opt-testrobotpkgarg.sh /opt/openrobots 1
roslaunch roscontrol_sot_talos sot_talos_controller_gazebo.launch

----------------

Pour faire bouger le robot :

export LC_ALL=C; unset LANGUAGE
source /opt/ros/kinetic/setup.bash
source /home/jules/setup-opt-testrobotpkgarg.sh /opt/openrobots 1
cd /opt/openrobots/share/sot-talos/tests/
python test.py