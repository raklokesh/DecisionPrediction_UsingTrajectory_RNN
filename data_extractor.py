import matplotlib.pyplot as plt
import os
import read_matrixData
import numpy as np
import scipy.signal as sig


def extract_data(file_indexes, directory, sample_time, Condition_Dictionary, Condition_Time_Dictionary, Condition_Trials):
    print('Extracting trajectory and decision data')

    file_names = [name for name in os.listdir(directory) if name.endswith('mat')]

    START_CIRCLE_X = 0.55 # x coordinate of start circle
    START_CIRCLE_Y = 0.15 # y coordinate of start circle
    TARG_Y = 0.15 # y coordinate of target circles
    TARG_X = [0.1, -0.1] # x coordinates of the right and left target circles
    TARG_RADIUS = 0.02 # radius of the target circle
    SUBJECT_NO = 2
    TRIAL_NO = 150

    x_coordinate_data = np.zeros((len(file_indexes), len(Condition_Dictionary), TRIAL_NO, SUBJECT_NO, int(1500 / sample_time) + 1)) + np.nan
    y_coordinate_data = np.zeros_like(x_coordinate_data) + np.nan
    x_vel_data = np.zeros_like(x_coordinate_data) + np.nan
    y_vel_data = np.zeros_like(x_coordinate_data) + np.nan
    decision_data = np.zeros((len(file_indexes), len(Condition_Dictionary), TRIAL_NO, SUBJECT_NO))

    for file_no in file_indexes:
        print(f'Running file no {file_no}')
        Data = read_matrixData.retrieve_data(file_names[file_no], directory)

        Condition_Order = [int(Data['Data'][0, 0]['c3d'][0, 0]['BLOCK_TABLE']['TP_LIST'][0][0][0][i][0]) - 2
                        for i in range(1, len(Condition_Dictionary) + 1)]

        for condition_no, condition_trial in enumerate(Condition_Trials):
            max_time = Condition_Time_Dictionary[Condition_Order[condition_no]]

            print('Running condition no: {} and type: {}'.format(condition_no, Condition_Order[condition_no]))

            for trial_count, trial_number in enumerate(np.arange(condition_trial[0], condition_trial[1])):
                end_times = np.array([0, 0]) # target reach times of the two subjects A and B
                idx = len(Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS']['TIMES'][0][0][0]) - 1
                end_A = False # whether A reached target
                end_B = False # whether B reached target
                while idx > -1:
                    if Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS'][0][0][0][0][idx][0] == 'E_END_REACHED_A':
                        end_times[0] = int(
                            Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS']['TIMES'][0][0][0][idx] * 1000)
                        end_A = True
                    elif Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS'][0][0][0][0][idx][0] == 'E_END_REACHED_B':
                        end_times[1] = int(
                            Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS']['TIMES'][0][0][0][idx] * 1000)
                        end_B = True
                    elif Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS'][0][0][0][0][idx][0] == 'E_SOUND_SIGNAL':
                        start_time = int(Data['Data'][0, 0]['c3d'][0, trial_number]['EVENTS']['TIMES'][0][0][0][idx] * 1000)
                    idx -= 1

                if end_A:
                    end_times[0] = max_time + start_time

                if end_B:
                    end_times[1] = max_time + start_time

                # shifting origin of trajectories to 0, 0
                x_coord = np.zeros((2, len(Data['Data'][0, 0]['c3d'][0, trial_number]['Right_HandX'].ravel())))
                y_coord = np.zeros_like(x_coord)
                x_coord[0, :] = Data['Data'][0, 0]['c3d'][0, trial_number]['Right_HandX'].ravel() - START_CIRCLE_X
                x_coord[1, :] = Data['Data'][0, 0]['c3d'][0, trial_number]['Left_HandX'].ravel() + START_CIRCLE_X
                y_coord[0, :] = Data['Data'][0, 0]['c3d'][0, trial_number]['Right_HandY'].ravel() - START_CIRCLE_Y
                y_coord[1, :] = Data['Data'][0, 0]['c3d'][0, trial_number]['Left_HandY'].ravel() - START_CIRCLE_Y
                
                # filtering position data using low pass butterworth filter
                fs = 1 / 0.001 # sampling frequency
                f_c = 15 # cutoff frequency
                b, a = sig.butter(6, f_c / fs / 2, 'low')
                _x_coord = sig.filtfilt(b, a, x_coord)
                _y_coord = sig.filtfilt(b, a, y_coord)
                
                # 4th order central difference approximation for the velocity
                x_vel = (-_x_coord[:, 3:-1] + 8 * _x_coord[:, 2:-2] - 8 * _x_coord[:, 1:-3] + _x_coord[:, 0:-4]) / 12 * fs
                y_vel = (-_y_coord[:, 3:-1] + 8 * _y_coord[:, 2:-2] - 8 * _y_coord[:, 1:-3] + _y_coord[:, 0:-4]) / 12 * fs

                # extracting decision data
                trial_decision = np.zeros(SUBJECT_NO) # decisions of the two subjects in the trial: 0 - not reached, 1 - right target, 2 - left target
                for subj in range(SUBJECT_NO):
                    dist_R = ((x_coord[subj, end_times[subj]] - TARG_X[0]) ** 2 + (y_coord[subj, end_times[subj]] - TARG_Y) ** 2) ** 0.5
                    dist_L = ((x_coord[subj, end_times[subj]] - TARG_X[1]) ** 2 + (y_coord[subj, end_times[subj]] - TARG_Y) ** 2) ** 0.5

                    trial_decision[subj] = 0
                    if dist_R < TARG_RADIUS:
                        trial_decision[subj] = 1
                    elif dist_L < TARG_RADIUS:
                        trial_decision[subj] = 2
                        
                decision_data[file_no, Condition_Order[condition_no], trial_count, :] = trial_decision

                # extracting kinematics
                for subj in range(SUBJECT_NO):
                    sample_indices = np.arange(start_time, start_time + max_time + 1, sample_time)
                    sample_indices[sample_indices > end_times[subj]] = end_times[subj]

                    x_coordinate_data[file_no, Condition_Order[condition_no], trial_count, subj, :len(sample_indices)] =\
                        x_coord[subj, sample_indices]
                    y_coordinate_data[file_no, Condition_Order[condition_no], trial_count, subj, :len(sample_indices)] =\
                        y_coord[subj, sample_indices]
                    x_vel_data[file_no, Condition_Order[condition_no], trial_count, subj, :len(sample_indices)] =\
                        x_vel[subj, sample_indices]
                    y_vel_data[file_no, Condition_Order[condition_no], trial_count, subj, :len(sample_indices)] =\
                        y_vel[subj, sample_indices]


        del Data

    return decision_data, x_coordinate_data, y_coordinate_data, x_vel_data, y_vel_data