# ------------------------------------------------------------------------------------------------------modified by Anne
def get_flat_aruco_target(start_id: int, nr_aruco_y: int, nr_aruco_x: int, marker_size: float, gap_size_x: float, gap_size_y: float):
    """Generates the center object points of aruco markers relative to their origin minding the gap sizes and with corresponding aruco ids.

    Output is written into a .csv in the declared folder. The origin is the center point of the first marker with the start_id. A row is defined
    along the X-axis and a column along the Y-axis. Consecutive aruco marker ids are given in the X direction. Z is always 0. All units in mm.

    :param data_path: str   - Directory the json will be stored in.
    :param start_id: int    - Start aruco marker id, consecutive ids will be assigned.
    :param nr_aruco_y:      - Number of aruco markers in Y direction.
    :param nr_aruco_x:      - Number of aruco markers in X direction.
    :param marker_size:     - Length in mm of one side of the aruco marker.
    :param gap_size_x:      - Gap in mm between to neighbouring aruco markers in X.
    :param gap_size_y:      - Gap in mm between to neighbouring aruco markers in Y.
    :return:
    """
    aruco_count = nr_aruco_x * nr_aruco_y
    aruco_ids = list(range(start_id, start_id + aruco_count))
    file_name = f'aruco_point_pattern_{nr_aruco_y}x{nr_aruco_x}_{marker_size}mm_ids{start_id}-{aruco_ids[-1]}.ini'
    target_point_list_coo = []
    target_point_list_ids = []

    x_incr = marker_size + gap_size_x
    y_incr = marker_size + gap_size_y
    mrkr = 0
    for row in range(nr_aruco_y):
        y = row * y_incr
        for column in range(nr_aruco_x):
            x = column * x_incr
            target_point_list_coo.append([round(x, 3), round(y, 3), 0.0])
            target_point_list_ids.append(f'{aruco_ids[mrkr]}')
            mrkr += 1

    return target_point_list_coo, target_point_list_ids

    # table = tabulate(target_point_list, ['# id', 'x', 'y', 'z'], 'plain', stralign='left', numalign='left')
    # file = ini.ConfigParser(allow_no_value=True)
    # file.optionxform = lambda option: option  # neccessary to preserve case of string data entries
    # file.add_section('local aruco target points')
    # file.set('local aruco target points', table)
    #
    # with open(os.path.join(data_path, file_name), 'w',  newline='') as ini_file:
    #     file.write(ini_file)


if __name__ == '__main__':
    #get_flat_aruco_target('../data/03_calib_data_florian', 100, 5, 7, 60, 20, 20)
    target_point_list_coo, target_point_list_ids= get_flat_aruco_target(100, 5, 7, 60, 20, 20)
    print('Target Point IDs')
    print('\t', target_point_list_ids)
    print('Target Point coordinates')
    print('\t', target_point_list_coo)
