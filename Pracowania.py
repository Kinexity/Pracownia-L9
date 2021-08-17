import csv
import matplotlib.pyplot as pt
import numpy as np
import scipy.optimize as sp

file_list = ["air1.csv","air2.csv","crone1.csv","crone2.csv","crone3.csv","nitrogen.csv","spark1.csv","spark2.csv"]

def fit_periodic_function(time, max, phase_shift, offset, exp_arg, frequency):
	cycle_full = (time + phase_shift) * frequency + 2
	cycle = np.trunc(cycle_full)
	return max * np.exp(- exp_arg * (cycle_full - cycle)) + offset

def fit_exp_function(time, max, offset, exp_arg):
	return max * np.exp(- exp_arg * time) + offset

def find_local_max(arr, first, last):
	if (first >= last or first < 0 or last > len(arr)):
		return []
	max_index = np.argmax(arr[first:last]) + first
	if arr[max_index] < 0.:
		return []
	before = find_local_max(arr, first, max_index - 1000)
	after = find_local_max(arr, max_index + 1000, last)
	return before + [max_index] + after

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def sum_impulses(data, peaks):
	peaks_np = np.array(peaks)
	dist_min = np.min(peaks_np[1:] - peaks_np[:-1]) - 100
	signal_sum = np.zeros(dist_min)
	include_last = (len(data) - peaks[-1]) > dist_min
	for peak in peaks[:-1]:
		signal_sum += np.array(data[peak: peak + dist_min])
	if (include_last):
		signal_sum += np.array(data[peaks[-1]: peaks[-1] + dist_min])
	return signal_sum

def main():
	for file in file_list:
		f = open(file)
		csv_reader = csv.reader(f, delimiter=',', dialect="excel")
		data = np.array(list(csv_reader))
		time_points = np.array(list(map(float, data[:,-5])))
		voltage_points = np.array(list(map(float, data[:,-1])))
		#pt.plot(time_points,voltage_points, ".r")
		#pt.show()
		pt.plot(time_points,voltage_points, ".r")
		max_voltage = np.max(voltage_points)
		min_voltage = np.min(voltage_points)
		min_freq = 9e3
		max_freq = 1.1e4
		expected_freq = 1e4
		print(np.sum(voltage_points > 0))
		last_maximum_time = time_points[int(len(voltage_points) * 29 / 30):][np.argmax(voltage_points[int(len(voltage_points) * 29 / 30):] > 0)]
		peeks_tp = [time_points[tp_index] for tp_index in find_local_max(voltage_points, 0, len(voltage_points))]
		initial_phase_shift = peeks_tp[0]
		#print(time_points[int(len(voltage_points) * 29 / 30)])
		#print((last_maximum_time - initial_phase_shift) * expected_freq)
		initial_fit_freq = (len(peeks_tp) - 1) / (peeks_tp[-1] - peeks_tp[0])
		print(initial_fit_freq)
		max_voltage_diff = np.abs(max_voltage - min_voltage)
		offset_init = np.average(np.sort(voltage_points)[:int(len(voltage_points) * 0.95)])
		lower_b = (max_voltage_diff * 0.8	, time_points[0]					, min_voltage	, 1.		, min_freq)
		upper_b = (max_voltage_diff * 2	, 1 / min_freq + time_points[0]		, max_voltage	, np.inf	, max_freq)
		initial = (max_voltage_diff		, initial_phase_shift				, offset_init	, 20		, initial_fit_freq)
		print(lower_b)
		print(upper_b)
		print(initial)
		par,cov = sp.curve_fit(fit_periodic_function, time_points, voltage_points, p0 = initial, bounds = (lower_b,upper_b), method='dogbox')
		#par[3] /= 1.5
		#par[0] *= 1.1
		#par,cov = sp.curve_fit(fit_function, time_points, voltage_points, p0=par,
		#bounds=(lower_b,upper_b), method='dogbox')
		print(par)
		print(np.sqrt(np.diagonal(cov)))
		fit_tp = np.arange(time_points[0],time_points[-1],(time_points[-1] - time_points[0]) / 1e6)
		pt.plot(fit_tp, fit_periodic_function(fit_tp, *par), "-g")
		#pt.plot(fit_tp, fit_function(fit_tp, *initial), "-b")
		pt.show()
		voltage_sum = sum_impulses(voltage_points, find_local_max(voltage_points, 0, len(voltage_points)))
		min_s = np.min(voltage_sum)
		new_time = time_points[:len(voltage_sum)] - time_points[0]
		par,cov = sp.curve_fit(fit_exp_function, new_time, voltage_sum, p0 = [1,0, np.min(voltage_sum)])
		print(par)
		print(np.sqrt(np.diagonal(cov)))
		pt.plot(new_time,voltage_sum, ".r")
		pt.plot(new_time, fit_exp_function(new_time, *par), "-g")
		pt.show()


main()