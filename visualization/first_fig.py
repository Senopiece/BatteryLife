import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import matplotlib
import pandas as pd

dataset_path = './dataset' # set the dataset path


font = {'family': 'Arial'}

matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42  # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42


# matplotlib.rc('font', **font0)
def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)


def set_ax_font_size(ax, fontsize=10):
    ax.tick_params(axis='y',
                   labelsize=fontsize  # y轴字体大小设置
                   )
    ax.tick_params(axis='x',
                   labelsize=fontsize  # x轴字体大小设置
                   )


def draw_ZN_sequence(fig):
    with open(f'{dataset_path}/ZN-coin/ZN-coin_204-1_20231205230212_07_1.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    plus_time = 0
    total_current = []
    total_voltage = []
    total_time = []
    for i in tqdm(range(length)):
        if i == 0:
            continue
        cycle_data = MATR_data['cycle_data'][i]
        current = cycle_data['current_in_A']
        current = [i * 1000 for i in current]
        voltage = cycle_data['voltage_in_V']
        times = cycle_data['time_in_s']
        new_times = []
        for time in times:
            h = float(time.split(':')[0])
            m = float(time.split(':')[1])
            s = float(time.split(':')[2])
            seconds = (h * 3600 + m * 60 + s) - 417000
            new_times.append(seconds)

        total_time = total_time + new_times
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 3:  # draw first 5 cycles
            break

    ax1 = plt.subplot(4, 1, 2)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)',  fontsize=15)
    ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    ax1.tick_params('y', colors=color)
    ax1.set_ylim(0, 2.0)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-5, 10)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    ax2.tick_params('y', colors=color)
    set_ax_linewidth(ax1)


def draw_CALB_sequence(fig):
    with open(f'{dataset_path}/CALB/CALB_0_B184.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    total_current = []
    total_voltage = []
    total_time = []
    for i in tqdm(range(length)):
        if i < 1:
            continue
        cycle_data = MATR_data['cycle_data'][i]
        current = cycle_data['current_in_A']
        indices = [i for i, x in enumerate(current) if x == 0]
        current = [i for i in current if i not in indices]
        voltage = cycle_data['voltage_in_V']
        voltage = [i for i in voltage if i not in indices]
        times = cycle_data['time_in_s']
        times = [i for i in times if i not in indices]
        new_times = []
        plus_time = 0
        for time in times:
            time = time.split(' ')[1]
            h = float(time.split(':')[0])
            m = float(time.split(':')[1])
            s = float(time.split(':')[2])
            seconds = (h * 3600 + m * 60 + s) + 2527 - 20000
            new_times.append(seconds)
        time = [time + plus_time for time in new_times]
        plus_time = max(time)
        print(min(time))

        total_time = total_time + time
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 3:  # draw first 5 cycles
            break

    ax1 = plt.subplot(4, 1, 4)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)',  fontsize=15)
    ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    ax1.tick_params('y', colors=color)
    ax1.set_ylim(1.0, 4.5)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-80, 120)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    ax2.tick_params('y', colors=color)
    set_ax_linewidth(ax1)


def draw_MATR_sequence(fig):
    with open(f'{dataset_path}/MATR/MATR_b1c0.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    plus_time = 0
    total_current = []
    total_voltage = []
    total_time = []
    for i in (range(length)):
        cycle_data = MATR_data['cycle_data'][i]
        current = cycle_data['current_in_A']
        voltage = cycle_data['voltage_in_V']
        time = cycle_data['time_in_s']
        time = [(i * 60) for i in time]
        time = [time + plus_time for time in time]
        plus_time = max(time)

        total_time = total_time + time
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 2:  # draw first 5 cycles
            break

    ax1 = plt.subplot(4, 1, 1)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)',  fontsize=15)
    ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    ax1.tick_params('y', colors=color)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-5, 7)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    ax2.tick_params('y', colors=color)
    set_ax_linewidth(ax1)
    # plt.title('Voltage-Current vs time Profile')


def draw_Tongji_sequence(fig):
    with open(f'{dataset_path}/Tongji/Tongji1_CY25-05_1-#1.pkl', 'rb') as f:
        MATR_data = pickle.load(f)

    length = len(MATR_data['cycle_data'])
    total_current = []
    total_voltage = []
    total_time = []
    for i in (range(length)):
        cycle_data = MATR_data['cycle_data'][i]
        current = cycle_data['current_in_A']
        voltage = cycle_data['voltage_in_V']
        time = cycle_data['time_in_s']

        total_time = total_time + time
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 4:  # draw first 5 cycles
            break

    ax1 = plt.subplot(4, 1, 1)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)',  fontsize=15)
    ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    ax1.tick_params('y', colors=color)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    ax2.tick_params('y', colors=color)
    ax2.set_ylim(-3.8, 2.5)
    set_ax_linewidth(ax1)

def draw_NA_sequence(fig):
    with open(f'{dataset_path}/NA_ion/2750-30_20250115171823_DefaultGroup_45_2.xlsx', 'rb') as f:
        data = pd.read_excel(f, sheet_name='Record')
    df = pd.DataFrame(data)
    length = len(df['Cycle'].unique())
    total_current = []
    total_voltage = []
    total_time = []
    for i in tqdm(range(1, length+1)):
        current = df.loc[df['Cycle'] == i, 'Current/A'].values.tolist()
        voltage = df.loc[df['Cycle'] == i, 'Voltage/V'].values.tolist()
        times = df.loc[df['Cycle'] == i, 'TestTime'].values.tolist()

        total_time = total_time + times
        total_current = total_current + current
        total_voltage = total_voltage + voltage
        if i == 3:  # draw first 5 cycles
            break

    ax1 = plt.subplot(4, 1, 3)
    color = sns.color_palette()[0]
    ax1.plot(total_time, total_voltage, '-', color=color)
    ax1.set_xlabel('Time(s)',  fontsize=15)
    ax1.set_ylabel('Voltage(V)', color=color,  fontsize=15)
    ax1.tick_params('y', colors=color)
    ax1.set_ylim(1.5, 4.2)

    ax2 = ax1.twinx()
    color = sns.color_palette()[3]
    ax2.set_ylim(-3.5, 5.0)
    ax2.plot(total_time, total_current, '-', color=color)
    ax2.set_ylabel('Current(A)', color=color, fontsize=15)
    ax2.tick_params('y', colors=color)
    set_ax_linewidth(ax1)


fig = plt.figure(figsize=(8, 10))  # set the size of the figure
draw_MATR_sequence(fig)
# draw_Tongji_sequence(fig)
draw_ZN_sequence(fig)
draw_NA_sequence(fig)
draw_CALB_sequence(fig)
fig.tight_layout()
# plt.show()
plt.savefig('./figures/first_fig.jpg', dpi=600)
plt.savefig('./figures/first_fig.pdf')
