#!/usr/bin/env python3
#
# General Dynamixel Robot Driver
#
# Auth: N. Kosanovic
# Date: 12 June. 2025
# Desc: Concise general-purpose implementation of Dynamixel SDK functions for robots. 
#       For the most part, all you really need is to repeatedly read joint positions and command currents.
#       This enables surprisingly transparent (and functional) torque control!
# Vers: 1.0 -   FastSyncRead Implmented (thank GOD) to allow rapid (but not infalliable) 
#               reading of joint position, current, and temperature sensors.

from dynamixel_sdk import *
import numpy as np
import pandas as pd

class dxl_controller:
    ############### CONSTANTS ###############

    # Dynamixel Communication Protocol 2.0 is used for series-x Dyna's.
    PROTOCOL_VERSION    = 2.0

    # EEPROM Address-related Defines
    # https:#emanual.robotis.com/docs/en/dxl/x/xl330-m288/#control-table-of-eeprom-area

    # Item                           = (DATA_ADDRESS, DATA_LENGTH)
    EEPROM_MODEL_NUMBER              = (0x00, 2)# 2 bytes; R;  motor's model number
    EEPROM_MODEL_INFORMATION         = (2, 4)   # 4 bytes; R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_FIRMWARE_VERSION          = (6, 1)   # 1 byte;  R;  motor's firmware version
    EEPROM_ID                        = (7, 1)   # 1 byte;  RW; motor's ID number [0~252]
    EEPROM_BAUD_RATE                 = (8, 1)   # 1 byte;  RW; motor's baud rate (bit/s) [0~6]
    EEPROM_RETURN_DELAY_TIME         = (9, 1)   # 1 byte;  RW; instruction packet send time
    EEPROM_DRIVE_MODE                = (10, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_OPERATING_MODE            = (11, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_SECONDARY_ID              = (12, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_PROTOCOL_TYPE             = (13, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_HOMING_OFFSET             = (20, 4)  # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_MOVING_THRESHOLD          = (24, 4)  # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_TEMPERATURE_LIMIT         = (31, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_MAX_VOLTAGE_LIMIT         = (32, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_MIN_VOLTAGE_LIMIT         = (34, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_PWM_LIMIT                 = (36, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_CURRENT_LIMIT             = (38, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_VELOCITY_LIMIT            = (44, 4)  # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_MAX_POSITION_LIMIT        = (48, 4)  # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_MIN_POSITION_LIMIT        = (52, 4)  # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_STARTUP_CONFIGURATION     = (60, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_PWM_SLOPE                 = (62, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    EEPROM_SHUTDOWN                  = (63, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # RAM Address related Defines
    # https:#emanual.robotis.com/docs/en/dxl/x/xl330-m288/#control-table-of-ram-area

    RAM_TORQUE_ENABLE                = (64, 1)  # 1 byte;  RW; Enable/Disable XL-330 Torque.
    RAM_LED                          = (65, 1)  # 1 byte;  RW; Turn On/Off XL-330 LED
    RAM_STATUS_RETURN_LEVEL          = (68, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_REGISTERED_INSTRUCTION       = (69, 1)  # 1 byte;  R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_HARDWARE_ERROR_STATUS        = (70, 1)  # 1 byte;  R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_VELOCITY_I_GAIN              = (76, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_VELOCITY_P_GAIN              = (78, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_POSITION_D_GAIN              = (80, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_POSITION_I_GAIN              = (82, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_POSITION_P_GAIN              = (84, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_FEEDFORWARD_2ND_GAIN         = (88, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_FEEDFORWARD_1ST_GAIN         = (90, 2)  # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_BUS_WATCHDOG                 = (98, 1)  # 1 byte;  RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_GOAL_PWM                     = (100, 2) # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_GOAL_CURRENT                 = (102, 2) # 2 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_GOAL_VELOCITY                = (104, 4) # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_PROFILE_ACCELERATION         = (108, 4) # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_PROFILE_VELOCITY             = (112, 4) # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_GOAL_POSITION                = (116, 4) # 4 bytes; RW; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_REALTIME_TICK                = (120, 2) # 2 bytes; R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_MOVING                       = (122, 1) # 1 byte;  R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_MOVING_STATUS                = (123, 1) # 1 byte;  R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_PRESENT_PWM                  = (124, 2,   -885,   885) # 2 bytes; R; Lower Limit; Upper Limit !!!!!!
    RAM_PRESENT_CURRENT              = (126, 2,   -200,   200) # 2 bytes; R; Lower Limit; Upper Limit !!!!!!
    RAM_PRESENT_VELOCITY             = (128, 4,   -445,   445) # 4 bytes; R; Lower Limit; Upper Limit !!!!!!
    RAM_PRESENT_POSITION             = (132, 4, -10000, 10000) # 4 bytes; R; Lower Limit; Upper Limit !!!!!!
    RAM_VELOCITY_TRAJECTORY          = (136, 4) # 4 bytes; R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_POSITION_TRAJECTORY          = (140, 4) # 4 bytes; R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAM_PRESENT_INPUT_VOLTAGE        = (144, 2,     31,    70) # 2 bytes; R; Lower Limit; Upper Limit !!!!!!
    RAM_PRESENT_TEMPERATURE          = (146, 1,      0,    70) # 1 byte;  R; Lower Limit; Upper Limit !!!!!!
    RAM_BACKUP_READY                 = (147, 1) # 1 byte;  R;  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Dynamixel Control Modes:
    CURRENT_CONTROL_MODE = 0
    VELOCITY_CONTROL_MODE = 1
    POSITION_CONTROL_MODE = 3
    EXTENDED_POSITION_CONTROL_MODE = 4
    CURRENT_BASED_POSITION_CONTROL_MODE = 5
    PWM_CONTROL_MODE = 16

    # USER-Defined Communication Flags
    COMMS_SUCCESSFUL_FLAG            =  0
    COMMS_READ_FAILURE_FLAG          = -1
    COMMS_NOTHING_CAME_IN_FLAG       = -2
    COMMS_WRONG_HEADER_FLAG          = -3
    COMMS_NOT_ENOUGH_BYTES_FLAG      = -4
    COMMS_DATA_IS_NONSENSE_FLAG      = -5


    # Robot Specific Data
    config_data = pd.read_csv("robot_config.csv", index_col=0)

    UPPER_JOINT_LIMITS = config_data['UpperLim'].to_numpy()
    LOWER_JOINT_LIMITS = config_data['LowerLim'].to_numpy()
    JOINT_CURRENT_LIMITS = config_data['CurrentLim'].to_numpy()
    HOME_JOINT_POSITIONS = config_data['HomePosition'].to_numpy()
    MOTOR_OFFSETS = config_data['Offsets'].to_numpy()
    MOTOR_AXIS = config_data['Axis'].to_numpy()
    ROBOT_IDS = config_data['ID'].to_numpy()
    P_GAINS = config_data['P_gain'].to_numpy()
    I_GAINS = config_data['I_gain'].to_numpy()
    D_GAINS = config_data['D_gain'].to_numpy()

    ID_ALL_MOTORS = 0xFE

    ############### CLASS-RELATED FUNCTIONS ###############
    # Initialization.
    def __init__(self, port, e_stop_on_close=True, baudrate=2_000_000):
        """
        Initializes the robot.

        Args: 
            port (string): USB Port of the U2D2. e.g. "/dev/ttyUSB0"
            e_stop_on_close (bool): Flag used to disable torque when program shuts-down (Ctrl+C)
            baudrate (int): Communication rate of robot.

        Returns:
            None
        """
        
        # Initialize common aspects of the code
        self.port = port
        self.portHandler = PortHandler(port)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        self.baudrate = baudrate
        self.e_stop_on_close = e_stop_on_close

        self.open_port()

        # This needs to be done more generally
        self.goalPositionSyncWriter = GroupSyncWrite(self.portHandler, self.packetHandler, self.RAM_GOAL_POSITION[0], self.RAM_GOAL_POSITION[1])
        self.goalCurrentSyncWriter = GroupSyncWrite(self.portHandler, self.packetHandler, self.RAM_GOAL_CURRENT[0], self.RAM_GOAL_CURRENT[1])
        self.PositionSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.RAM_PRESENT_POSITION[0], self.RAM_PRESENT_POSITION[1])
        self.CurrentSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.RAM_PRESENT_CURRENT[0], self.RAM_PRESENT_CURRENT[1])

    # Destructor! Called on program end (or collapse/interrupt). 
    def __del__(self):
        """
        Closes the program. Depending on "e_stop_on_close", may elect to shut off joint torques before closing port. 

        Args: 
            None

        Returns:
            None
        """

        if self.e_stop_on_close == True:
            # self._set_dynamixel_current(np.zeros(20, np.int8))
            self.e_stop()

        self.close_port()
        print("\nPort Closed.")


    ############### COMMUNICATIONS SETUP FUNCTIONS ###############
    # Open the port, set the baud rate.
    def open_port(self):
        """
        Opens the previously-specified port at the previously-specified baudrate.
        
        Args: 
            None

        Returns:
            None
        """
        
        if self.portHandler.openPort():
            print("Successfully connected to port: ", self.portHandler.port_name)
        else: 
            print("Failed to open port; exiting...")
            quit()

        if self.portHandler.setBaudRate(self.baudrate):
            print("Successfully set baudrate to: ", self.portHandler.baudrate)
        else:
            print("Failed to set baudrate.  exiting...")        
        return 

    # Gracefully close the port.  
    def close_port(self):
        """
        Close the previously-specified port.

        Args:
            None

        Returns:
            None
        """

        self.portHandler.closePort()
        print("Port", self.port, "closed.")
        return


    ############### LOW-LEVEL DATA COMMUNICATIONS FUNCTIONS (Idiot-Level Boilerplate) ###############
    # Send a single cmd to a motor.
    def single_cmd(self, motor_ID, data_name, data):
        """
        Send a single command to a specified motor.

        Args: 
            motor_ID (int): ID of the targeted motor (or use ID_ALL_MOTORS to broadcast to all motors in robot)
            data_name (tuple): Tuple describing target data information, in the form of (motorDataMemoryPosition, motorDataMemoryLength).  
            data (int): value of data to be sent to that location.
            
        Returns:
            None
        """

        data_address = data_name[0]
        data_length  = data_name[1]

        if data_length == 1:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_ID, data_address, data)
        elif data_length == 2:
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_ID, data_address, data)
        elif data_length == 4:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_ID, data_address, data)
        else:
            print("Incorrect data length, nothing was sent.")    

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        
        return

    # Read a single value from a motor.
    def single_cmd_read(self, motor_ID, data_name):
        """
        Read a single a single location of a targetted motor's memory.

        Args: 
            motor_ID (int): ID of the targeted motor (or use ID_ALL_MOTORS to broadcast to all motors in robot)
            data_name (tuple): Tuple describing target data information, in the form of (motorDataMemoryPosition, motorDataMemoryLength).  

        Returns:
            dxl_data (int): Raw data read from the target motor.
        """


        data_address = data_name[0]
        data_length  = data_name[1]

        if data_length == 1:
            dxl_data, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, motor_ID, data_address)
        elif data_length == 2:
            dxl_data, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, motor_ID, data_address)
        elif data_length == 4:
            dxl_data, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, motor_ID, data_address)
        else:
            print("Incorrect data length, nothing was read.")    

        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        
        return dxl_data

    # Set motor positions in counts.
    def _set_dynamixel_pos(self, cmd_array=np.ones(len(ROBOT_IDS))*2048):
        """
        Lower-level function to command motor positions via dynamixel counts (values 0-4095).

        Args: 
            cmd_array (array (ints)): n x 1 array of ints of joint positions.  Defaults to value of 2048 (motor shaft pointing directly "up").

        Returns:
            None
        """

        # Convert motor position command [count] to [bytes]
        for cmd in range(len(cmd_array)):
            target = int(cmd_array[cmd])
            dxl_counts = [DXL_LOBYTE(DXL_LOWORD(target)), 
                          DXL_HIBYTE(DXL_LOWORD(target)),
                          DXL_LOBYTE(DXL_HIWORD(target)),
                          DXL_HIBYTE(DXL_HIWORD(target))]
            
            # Add bytes to outbuffer.
            dxl_addparam_result = self.goalPositionSyncWriter.addParam(self.ROBOT_IDS[cmd], dxl_counts)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % self.ROBOT_IDS[cmd])
                quit()

        # Once all positions have been converted to bytes and buffered, send the packet.
        dxl_comm_result = self.goalPositionSyncWriter.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        # Lastly, clear the buffer.
        self.goalPositionSyncWriter.clearParam()
        return

    # Read motor positions in counts.
    def _read_dynamixel_pos(self, id_array=np.zeros(len(ROBOT_IDS))):
        """
        Lower-level function to read target motor positions in dynamixel counts (0-4095).

        *Defaults to an ordered array of IDs specified by the config file.*

        Args: 
            id_array (array (ints)): m x 1 array of joints to read positions from. 

        Returns:
            data_array (array (ints)): Array of joint positions (in counts) for the relavent motors.
        """

        # Convert motor position command [count] to [bytes]
        data_array = np.zeros(len(id_array))

        # Fill the packet with which IDs we're targeting
        for id in id_array:
            dxl_addparam_result = self.PositionSyncRead.addParam(id)
            if dxl_addparam_result != True:
                print("[ID:%03d] PositionSyncRead addparam failed" % id)
                quit()
        
        # Run the fastSyncRead command.  REQUIRES NEWEST DXL WIZARD!
        dxl_comm_result = self.PositionSyncRead.fastSyncRead()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Decode the bytes that just came in.
        for dxl in range(0, len(id_array)):
            dxl_position = self.PositionSyncRead.getData(
                id_array[dxl],
                self.RAM_PRESENT_POSITION[0],
                self.RAM_PRESENT_POSITION[1])

            data_array[dxl] = dxl_position

        # Empty out the dictionary for later.
        self.PositionSyncRead.clearParam()

        return data_array

    # Set motor current in counts (1 count ~ 1 mA).
    def _set_dynamixel_current(self, cmd_array=np.zeros(len(ROBOT_IDS))):
        """
        Lower-level function to command motor currents via dynamixel counts (values 0-1750).

        Args: 
            cmd_array (array (ints)): n x 1 array of ints of joint currents.  Defaults to value of 0 [mA].

        Returns:
            None
        """

        # Convert motor position command [count] to [bytes]
        for cmd in range(len(cmd_array)):
            target = int(cmd_array[cmd])
            dxl_current_counts = [DXL_LOBYTE(target), 
                                  DXL_HIBYTE(target)]
            
            # Add bytes to outbuffer.
            dxl_addparam_result = self.goalCurrentSyncWriter.addParam(self.ROBOT_IDS[cmd], dxl_current_counts)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % self.ROBOT_IDS[cmd])
                quit()

        # Once all currents have been converted to bytes and buffered, send the packet.
        dxl_comm_result = self.goalCurrentSyncWriter.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Lastly, clear the buffer.
        self.goalCurrentSyncWriter.clearParam()
        return

    # Clamp joint current commands within its limits (in counts).
    def _joint_current_limiter(self, count, CURRENT_LIMIT):
        """
        Clamp a robot joint current command below its max.

        Args: 
            CURRENT_LIMIT (int): Max current of joint in [count]. 

        Returns:
            count (int): Clamped position value in [count].
        """

        if count > CURRENT_LIMIT:
            count = CURRENT_LIMIT

        if count < 0:
            if count < -CURRENT_LIMIT:
                count = -CURRENT_LIMIT

            # Two's compliment magic.
            count += 65535

        return count

    # Clamp joint positions within their limits (in counts).
    def _joint_limiter(self, count, LOWER_LIMIT, UPPER_LIMIT):
        """
        Clamp a robot joint position between its upper and lower limits.

        Args: 
            LOWER_LIMIT (int): Lower position limit of joint in [count]. 
            UPPER_LIMIT (int): Upper position limit of joint in [count].

        Returns:
            count (int): Clamped position value in [count].
        """

        if count > UPPER_LIMIT:
            count = UPPER_LIMIT
        elif count < LOWER_LIMIT:
            count = LOWER_LIMIT
        return count

    # Print communications status for FastSyncReads.
    def print_comms_status(self, comms_status_flag):
        """
        Prints communication status for FastSyncRead.  Useful debugging tool for fast loops that are falling apart.

        Args: 
            comms_status_flag (int): Flag describing what happened during comms.

        Returns:
            None
        """

        if comms_status_flag == self.COMMS_SUCCESSFUL_FLAG:
            print("[", time.time(), "]\t SUC: Communications successful.")

        elif comms_status_flag == self.COMMS_READ_FAILURE_FLAG:
            print("[", time.time(), "]\t ERR: Communications general failure.")

        elif comms_status_flag == self.COMMS_NOTHING_CAME_IN_FLAG:
            print("[", time.time(), "]\t ERR: No data came in.")

        elif comms_status_flag == self.COMMS_WRONG_HEADER_FLAG:
            print("[", time.time(), "]\t ERR: Data header is wrong.")

        elif comms_status_flag == self.COMMS_NOT_ENOUGH_BYTES_FLAG:
            print("[", time.time(), "]\t ERR: Incorrect read data length.")

        elif comms_status_flag == self.COMMS_DATA_IS_NONSENSE_FLAG:
            print("[", time.time(), "]\t ERR: Data outside of expected range.")

    # Read input currents from all motors.
    def _read_dynamixel_currents(self, id_array=np.zeros(len(ROBOT_IDS))):
        """
        Read input currents (in counts) for selected motors.

        Args: 
            id_array (array (int)): Array of which motors to read current commands from. 

        Returns:
            data_array (array (int)): Array of input currents in [count].
        """

        data_array = np.zeros(len(id_array))

        # Fill the packet with which IDs we're targeting
        for id in id_array:
            dxl_addparam_result = self.CurrentSyncRead.addParam(id)
            if dxl_addparam_result != True:
                print("[ID:%03d] CurrentSyncRead addparam failed" % id)
                quit()

        # Run the fastSyncRead command.  REQUIRES NEWEST DXL WIZARD!
        dxl_comm_result = self.CurrentSyncRead.fastSyncRead()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Decode the bytes that just came in.
        for dxl in range(0, len(id_array)):
            dxl_current = self.CurrentSyncRead.getData(
                id_array[dxl],
                self.RAM_PRESENT_CURRENT[0],
                self.RAM_PRESENT_CURRENT[1])

            data_array[dxl] = dxl_current

        # Empty out the dictionary for later.
        self.CurrentSyncRead.clearParam()

        return data_array


    ############### MED-LEVEL COMMUNICATIONS FUNCTIONS ###############

    # Read all robot joint positions in radians. 
    def read_joint_positions(self):
        """
        Read all robot joint positions simultaneously.

        Args: 
            None

        Returns:
            joint_positions (array (float)): Array of joint positions (0 is pointing straight up).
        """

        motor_positions = self._read_dynamixel_pos(self.ROBOT_IDS)

        # vals, comms_result = self.fast_sync_read()
        # if comms_result is not self.COMMS_SUCCESSFUL_FLAG:
        #     self.portHandler.ser.flush()
        #     return None, comms_result
        return list(np.asarray(motor_positions, dtype=float) * (2 * np.pi / 4096.0) - np.pi)

    # Set joint positions in radians. Includes joint limit checking.
    def set_joint_pose(self, rad_array=np.zeros(len(ROBOT_IDS))):
        """
        Command motor positions via Radians (where 0 rad points directly up).  Adheres to joint limits.

        Args: 
            cmd_array (array (floats)): n x 1 array of joint positions in [rads].  Defaults to value of 0 (motor shaft pointing directly "up").

        Returns:
            None
        """

        # TODO: SEE if this needs to stay here.
        # if len(rad_array) != 20:
        #     print("Incorrect Radian Array Input... Exiting.")
        #     quit()

        angle_conversion_factor = 4095 / 6.283
        
        dxl_counts_array = self.HOME_JOINT_POSITIONS

        for idx in range(len(rad_array)):
            rad_to_count = int(np.round(rad_array[idx] * self.MOTOR_AXIS[idx] * angle_conversion_factor + self.MOTOR_OFFSETS[idx]))
            dxl_counts_array[idx] = self._joint_limiter(rad_to_count, self.LOWER_JOINT_LIMITS[idx], self.UPPER_JOINT_LIMITS[idx])

        self._set_dynamixel_pos(dxl_counts_array)
        return

    # Set motor joint torque in [Nm]. Includes joint saturation.
    def set_joint_torque(self, torque_array=np.zeros(len(ROBOT_IDS))):
        """
        Command joints to output a desired torque in [Nm]. Uses a current-to-torque conversion.

        Args: 
            cmd_array (array (floats)): n x 1 array of ints of joint torques.  Defaults to value of 0.

        Returns:
            None
        """

        # TODO: See if this needs to be removed
        # if len(torque_array) != 20:
        #     print("Incorrect Torque Array Input... Exiting.")
        #     quit()

        dxl_current_counts_array = np.zeros(len(torque_array))

        # Convert each desired torque into a current, and then set that.
        for idx in range(len(torque_array)):
            current_to_count = int(np.round(self.torque_to_current(torque_array[idx], model_type='QUADRATIC_1')))

            # TODO:  950 is a magic number. ELIMINATE IT!
            dxl_current_counts_array[idx] = int(self._joint_current_limiter(current_to_count, 950))

        # Set current in [mA].
        self._set_dynamixel_current(dxl_current_counts_array)
        return

    # Read all robot joint currents [mA] simultaneously.
    def read_joint_currents(self):
        """
        Read input currents [mA] for all motors simultaneously.

        Args: 
            id_array (array (int)): Array of which motors to read current commands from. 

        Returns:
            data_array (array (int)): Array of input currents in [count].
        """
        count_to_mA = 1
        joint_currents = self._read_dynamixel_currents(self.ROBOT_IDS)        
        return joint_currents * count_to_mA

    # Read all robot joint torques [Nm] simultaneously, assuming that Current-Torque I/O models exist.
    def read_joint_torques(self):
        """
        Read all robot joint torques simultaneously.
        *Note: this assumes a I/O relationship between current and torque*

        Args: 
            None

        Returns:
            joint_torques (array (float)): Array of joint torques in [Nm].
        """
        # Collect joint currents in [mA]
        joint_currents = self.read_joint_currents()

        # Do a current-to-torque conversion. Now, joint_torques is in [Nm]
        joint_torques = self.torque_to_current(joint_currents)

        return joint_torques

    ############### ROBOT FUNCTIONS (If you gotta look at something, this is it) ###############

    # Set current, vel/acc profile, and PID gains from the config file
    def config_set_dynamics(self):
        """
        Uses the config file to set robot POSITION-CONTROLLER current limits, acceleration/velocity profiles, and PID gains.
        
        *Only works in Position Control Modes*

        Args: 
            None

        Returns:
            None
        """

        for joint_idx in self.config_data['ID'].to_list():
            # Set Motor Current, Speed, and Accel Profile Data
            self.single_cmd(self.config_data['ID'].to_list()[joint_idx - 1], self.RAM_GOAL_CURRENT, self.config_data['CurrentLim'].to_list()[joint_idx - 1])
            time.sleep(0.01)
            self.single_cmd(self.config_data['ID'].to_list()[joint_idx - 1], self.RAM_PROFILE_VELOCITY, self.config_data['MaxSpeed'].to_list()[joint_idx - 1])
            time.sleep(0.01)
            self.single_cmd(self.config_data['ID'].to_list()[joint_idx - 1], self.RAM_PROFILE_ACCELERATION, self.config_data['MaxAccel'].to_list()[joint_idx - 1])
            time.sleep(0.01)

            # Set motor PID gains.
            self.single_cmd(self.config_data['ID'].to_list()[joint_idx - 1], self.RAM_POSITION_P_GAIN, self.config_data['P_gain'].to_list()[joint_idx - 1])
            time.sleep(0.01)
            self.single_cmd(self.config_data['ID'].to_list()[joint_idx - 1], self.RAM_POSITION_I_GAIN, self.config_data['I_gain'].to_list()[joint_idx - 1])
            time.sleep(0.01)
            self.single_cmd(self.config_data['ID'].to_list()[joint_idx - 1], self.RAM_POSITION_D_GAIN, self.config_data['D_gain'].to_list()[joint_idx - 1])
            time.sleep(0.01)

        print("Dynamic properties set.")
        return

    # Set current, vel/acc profile on the fly.
    def set_dynamics(self, profile_velocity, profile_acceleration, goal_current):
        """
        Set goal current, profile velocity, and profile acceleration on the fly.
        
        *Only works in Current-Based Position Control Modes*

        Args: 
            profile_velocity (int): Maximum velocity of the actuators in [count]
            profile_acceleration (int): Maximum acceleration of the actuators in [count]
            goal_current (int): Maximum expendable current of the actuators in [count]

        Returns:
            None
        """

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_PROFILE_VELOCITY, profile_velocity)
        time.sleep(0.01)
        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_PROFILE_ACCELERATION, profile_acceleration)
        time.sleep(0.01)
        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_GOAL_CURRENT, goal_current)
        time.sleep(0.01)

    # Emergency stop: immediately disable motor torques.
    def e_stop(self):
        """
        Emergency stop: Immediately disable motor torques.

        Args: 
            None

        Returns:
            None
        """

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_TORQUE_ENABLE, 0)
        print("\nTorque Disabled.")
        return

    # Torque-to-current conversion, from [Nm] to [mA]. Different models too!
    def torque_to_current(self, desired_torque, model_type='LINEAR'):
        """
        Translate a desired torque [Nm] to a corresponding input current [mA] to simplify control. I/O data allows different models to be used.

        Args: 
            desired_torque (float): Desired torque in [Nm] 
            model_type (string): can be 'LINEAR', 'AFFINE', or 'QUADRATIC'.  More complicated models may be more accurate, but will take more time to compute.

        Returns:
            necessary_current (float): Associated current in [mA].  Needs to be turned into an int!
        """

        # TODO: FIGURE OUT THE FUCKING NUMBERS HERE, GET THIS TO WORK FOR MULTIPLE MOTORS
        if model_type == 'LINEAR':
            # Tau = a * i
            # i = Tau / a
            # a = 0.2615 [Nm / A]
            # a = 0.0002615 [Nm / mA]
            # a = 241.7100372
            a = 0.0002615 * 5 
            return desired_torque / a
        
        if model_type == 'AFFINE':
            # Tau = a * i + b
            # i = (Tau - b) / a
            # a = 0.5215 [Nm / A]
            # b = 0.1047 [Nm]
            a = 0.0005215 * 4
            b = 0.0001047 * 1
            return (desired_torque - b) / a
        
        if model_type == 'QUADRATIC_1':
            # Tau = a * i^2 + b * i + c
            # i = (-b + sqrt(b*b - 4*a*(c-tau)) / (2*a)

            # Parameters for 288T
            a = -1.0286 / 1000
            b = 1.1387 / 1000
            c = 0.0481 / 1000

            a = -1.4381
            b = 1.4588
            c = 0

            a = -.0001
            b = 0.0014
            c = 0

            # Parameters for 077T
            a = -0.3448
            b = 0.3774
            c = 0

            a = -0.0000003448
            b = 0.0003774
            c = 0

            if desired_torque < 0:
                return (-b + np.sqrt(b*b + 4 * a * desired_torque)) / (2 * a)
            if desired_torque > 0:
                return -(-b + np.sqrt(b*b - 4 * a * desired_torque)) / (2 * a)


    # Preprogrammed init sequence for compliant position control
    def compliant_position_control_init(self):
        """
        Preprogrammed init sequence for Compliant Position control.

        Args: 
            None

        Returns:
            None
        """

        # Startup Config for Each Motor
        self.INIT_SPEED = 100
        self.INIT_ACCEL = 10
        self.INIT_CURRENT = 100

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_TORQUE_ENABLE, 0)
        print("\nTorque Disabled.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.EEPROM_OPERATING_MODE, self.CURRENT_BASED_POSITION_CONTROL_MODE)
        print("Current-based Position Control Mode activated.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_TORQUE_ENABLE, 1)
        print("Torque Enabled.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_GOAL_CURRENT, self.INIT_CURRENT)
        print("Initial Goal Current Set.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_PROFILE_VELOCITY, self.INIT_SPEED)
        print("Initial Profile Velocity Set.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_PROFILE_ACCELERATION, self.INIT_ACCEL)
        print("Initial Profile Acceleration Set.")
        time.sleep(0.1)

        print("Homing...")
        self._set_dynamixel_pos(self.HOME_JOINT_POSITIONS)
        time.sleep(0.5)

        print("Ready to roll.")
        time.sleep(0.1)
        return

    # Enable torque control on whole robot.
    def torque_control_init(self):
        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_TORQUE_ENABLE, 0)
        print("\nTorque Disabled.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.EEPROM_OPERATING_MODE, self.POSITION_CONTROL_MODE)
        print("Position Control Mode for EVERYTHING activated to zero encoders.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.EEPROM_OPERATING_MODE, self.CURRENT_CONTROL_MODE)
        print("Current Control Mode for EVERYTHING activated. God help us all.")
        time.sleep(0.1)

        self.single_cmd(self.ID_ALL_MOTORS, self.RAM_TORQUE_ENABLE, 1)
        self.single_cmd(4, self.RAM_TORQUE_ENABLE, 0)

        print("Torque Enabled.")
        time.sleep(0.1)

        print("Ready to roll.")
        return


    # TODO: This shit does not work. Make it nice!
    # Read temperatures from all motors simultaneously. 
    # def read_temps(self):
    #     """
    #     Read all motor temperatures simultaneously.

    #     Args: 
    #         None
    #     Returns:
    #         vals (array (int)): Array of counts related to all robot joint temperatures.
    #         comms_result (int): Flag explaining how comms went.
    #     """

    #     vals, comms_result = self.fast_sync_read(self.RAM_PRESENT_TEMPERATURE, [self.RAM_PRESENT_TEMPERATURE[2], self.RAM_PRESENT_TEMPERATURE[3]])
    #     if comms_result is not self.COMMS_SUCCESSFUL_FLAG:
    #         return None, comms_result
    #     return (vals), comms_result
