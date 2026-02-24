# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd
import tarfile
import gzip
import shutil
import json

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from .time_normalization_utils import normalize_cycle_times


@PREPROCESSORS.register()
class StanfordPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        raw_file = Path(parent_dir) / 'data.tar.gz'
        # Unzip the raw file
        # Skip extraction if the file already exists
        if not os.path.exists(raw_file.parent / 'data'):
            if not self.silent:
                tqdm.write('Unzipping Stanford dataset')
            with tarfile.open(raw_file, 'r:gz') as tar:
                tar.extractall(path=parent_dir)
        else:
            if not self.silent:
                tqdm.write('Skipping Stanford dataset, already exists')

        # Iterate through each file in the zip
        gz_path = raw_file.parent / 'data/maccor/'
        files_path = os.listdir(gz_path)
        gz_files = [i for i in files_path if i.endswith('.gz')]
        for file in tqdm(gz_files, disable=self.silent, desc='Unzipping'):
            gz_file = gz_path / file
            json_file = gz_file.with_suffix('')

            # Skip extraction if the file already exists
            if not os.path.exists(json_file):
                if not self.silent:
                    tqdm.write(f'Unzipping Stanford {file}')
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(json_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                if not self.silent:
                    tqdm.write(f'Skipping {file}, already exists')

        cells = [i for i in files_path if i.endswith('.json')]
        process_batteries_num = 0
        skip_batteries_num = 0
        for cell in tqdm(cells, desc=f'Processing Stanford cells'):
            cell_name = 'Stanford_' + cell.split('.')[0]
            if cell == 'Nova_Regular_197.034.json':
                continue
            # Step1: judge whether to skip the processed file if exists
            whether_to_skip = self.check_processed_file(cell_name)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            # Step2: get cycle data from files of each cell
            with open(str(gz_path) + '/' + cell, 'rb') as file:
                data = json.load(file)
            cell_data = data['cycles_interpolated']
            if len(cell_data) == 0:
                continue

            # clean data
            cleaned_df = clean_cell(cell_data)
            battery = organize_cell(cleaned_df, cell_name, 0.24)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, C):
    timeseries_df = timeseries_df.sort_values('test_time')
    cycle_data = []
    for cycle_index, df in timeseries_df.groupby('cycle_index'):
        cycle_data.append(CycleData(
            cycle_number=int(cycle_index),
            voltage_in_V=df['voltage'].tolist(),
            current_in_A=df['current'].tolist(),
            temperature_in_C=df['temperature'].tolist(),
            discharge_capacity_in_Ah=df['discharge_capacity'].tolist(),
            charge_capacity_in_Ah=df['charge_capacity'].tolist(),
            time_in_s=df['test_time'].tolist()
        ))
    # Charge Protocol is constant current
    charge_protocol = [CyclingProtocol(
        rate_in_C=1.0, start_soc=0.0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=0.75, start_soc=1.0, end_soc=0.0
    )]

    soc_interval = [0, 1]

    # Normalize time data across all cycles
    cycle_data = normalize_cycle_times(cycle_data, name)

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='pouch',
        anode_material='graphite',
        cathode_material='LiNi0.5Mn0.3Co0.2O2',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=C,
        min_voltage_limit_in_V=3,
        max_voltage_limit_in_V=4.4,
        SOC_interval=soc_interval
    )

def clean_cell(cell_data):
    df = pd.DataFrame(cell_data)

    # reset the cycle number and drop the first formation cycle
    cycle_number = set([index for index in df['cycle_index'].tolist() if index != 0])
    for current_index, new_index in zip(cycle_number, range(1, len(cycle_number) + 1)):
        df.loc[df['cycle_index'] == current_index, 'cycle_index'] = new_index
    df = df[~(df['cycle_index'] == 1)]
    df['cycle_index'] = df['cycle_index'] - 1

    # drop the abnormal current rows
    cycle_number = set([index for index in df['cycle_index'].tolist() if index != 0])
    for index in cycle_number:
        df = df[~((df['cycle_index'] == index) & (df['current'] > -0.001) & (df['current'] < 0))]
        df.loc[(df['cycle_index'] == index) & (df['current'] >= 0), "discharge_capacity"] = 0
        df.loc[(df['cycle_index'] == index) & (df['current'] < 0), "charge_capacity"] = 0

    return df