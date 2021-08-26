import csv
import matplotlib.pyplot as pt
import numpy as np
import scipy.optimize as sp
import math
import scipy.constants as con

file_list = ["nitrogen.csv","air1.csv","spark1.csv","spark2.csv","crone1.csv","crone2.csv","crone3.csv","air2.csv"]

def fit_exp_function(time, max, offset, inv_decay_t): #fit function
	return max * np.exp(- time * inv_decay_t) + offset

def find_local_max(arr, first, last): #function for finding voltage peaks after laser pulse
	if (first >= last or first < 0 or last > len(arr)):
		return []
	max_index = np.argmax(arr[first:last]) + first
	if arr[max_index] < 0.:
		return []
	before = find_local_max(arr, first, max_index - 1000)
	after = find_local_max(arr, max_index + 1000, last)
	return before + [max_index] + after

def sum_impulses(data, peaks): #function for summing voltage from all pulses
	dist_min = np.min(peaks[1:] - peaks[:-1]) - 100
	include_last = (len(data) - peaks[-1]) > dist_min
	return sum([np.array(data[peak: peak + dist_min]) for peak in peaks[:0 if include_last else -1]])

def main():
	data = np.loadtxt("NO2.txt", skiprows=1)
	cross_section = data[np.where(data[:,0]==411.),1][0,0]
	print(cross_section)
	inv_decay_t_base = 0.
	for file in file_list:
		f = open(file)
		csv_reader = csv.reader(f, delimiter=',', dialect="excel")
		data = np.array(list(csv_reader))
		time_points = np.array(list(map(float, data[:,-5])))
		voltage_points = np.array(list(map(float, data[:,-1])))
		voltage_sum = sum_impulses(voltage_points, np.array(find_local_max(voltage_points, 0, len(voltage_points))))[1:] #peaks not included because big errors
		max_voltage_diff = np.max(voltage_sum) - np.min(voltage_sum)
		new_time = time_points[:len(voltage_sum)] - (2 * time_points[0] - time_points[1])
		initial = (max_voltage_diff, np.min(voltage_sum), 2000000.)
		par,cov = sp.curve_fit(fit_exp_function, new_time, voltage_sum, p0 = initial)
		inv_decay_t = par[2]
		rounding = -int(math.floor(np.log10(inv_decay_t)) - 3)
		if file=="nitrogen.csv":
			print(file[:-4],"\t", "{:.3e}".format(round(inv_decay_t, rounding)))
			inv_decay_t_base = inv_decay_t
		else:
			N = (inv_decay_t - inv_decay_t_base) / (con.c * 100 * cross_section)
			ppb = 1e15 * con.k * N * 298.15 / float(con.physical_constants["standard atmosphere"][0])
			print(file[:-4],"\t","{:.3e}".format(np.round(inv_decay_t, rounding)),"\t","{:.3e}".format(N),"\t", "{:.3e}".format(ppb))
		px = 1 / pt.rcParams['figure.dpi']
		pt.subplots(figsize=(1200 * px, 800 * px))
		pt.xlabel("Czas od impulsu [s]")
		pt.ylabel("Suma napięć [V]")
		pt.title(file[:-4])
		pt.plot(new_time,voltage_sum, ".r", label="Suma impulsów")
		pt.plot(new_time, fit_exp_function(new_time, *par), "-g", label="Dopasowana krzywa zaniku")
		pt.legend()
		pt.savefig(file[:-4] + ".png", bbox_inches='tight')
		pt.subplots(figsize=(1200 * px, 800 * px))
		pt.xscale("log")
		pt.xlabel("Czas od impulsu [s]")
		pt.ylabel("Suma napięć [V]")
		pt.title(file[:-4])
		pt.plot(new_time,voltage_sum, ".r", label="Suma impulsów")
		pt.plot(new_time, fit_exp_function(new_time, *par), "-g", label="Dopasowana krzywa zaniku")
		pt.legend()
		pt.savefig(file[:-4] + "_log.png", bbox_inches='tight')

main()