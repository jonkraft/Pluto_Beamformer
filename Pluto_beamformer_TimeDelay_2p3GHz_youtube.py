"""
Jon Kraft, Jan 21 2023
https://github.com/jonkraft/Pluto_Beamformer
video walkthrough of this at:  https://www.youtube.com/@jonkraft

"""
# Copyright (C) 2020 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import adi
import matplotlib.pyplot as plt
import numpy as np

'''Setup'''
samp_rate = 30e6    # must be <=30.72 MHz if both channels are enabled
NumSamples = 2**12
f_carrier = 2.3e9
rx_lo = f_carrier
rx_mode = "manual"  # can be "manual" or "slow_attack"
rx_gain0 = 20
rx_gain1 = 20
tx_lo = rx_lo
tx_gain = -3
fc0 = int(200e3)
c = 299792458   # speed of light in m/s
time_cal = -80   # time offset (in ps) at boresight
num_scans = 500
invert_Rx = -1  # set to either -1 or 1.  This is to compensate for a 180 deg phase shift on the LO divider of Pluto

''' Set distance between Rx antennas '''
d_wavelength = 0.5                # distance between elements as a fraction of wavelength.  This is normally 0.5
wavelength = c/f_carrier              # wavelength of the RF carrier
d = d_wavelength*wavelength         # distance between elements in meters
time_max = d / c * 1E12  # max time (in ps) needed to steer from 0 to 90 deg
print("Set distance between Rx Antennas to ", int(d*1000), "mm")

'''Create Radio'''
sdr = adi.ad9361(uri='ip:192.168.2.1')

'''Configure properties for the Radio'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(samp_rate)
sdr.rx_lo = int(rx_lo)
sdr.gain_control_mode = rx_mode
sdr.rx_hardwaregain_chan0 = int(rx_gain0)
sdr.rx_hardwaregain_chan1 = int(rx_gain1)
sdr.rx_buffer_size = int(NumSamples)
sdr._rxadc.set_kernel_buffers_count(1)   # set buffers to 1 (instead of the default 4) to avoid stale data on Pluto
sdr.tx_rf_bandwidth = int(samp_rate)
sdr.tx_lo = int(tx_lo)
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = int(tx_gain)
sdr.tx_hardwaregain_chan1 = int(-88)
sdr.tx_buffer_size = int(2**18)

'''Program Tx and Send Data'''
fs = int(sdr.sample_rate)
N = 2**16
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i0 = np.cos(2 * np.pi * t * fc0) * 2 ** 14
q0 = np.sin(2 * np.pi * t * fc0) * 2 ** 14
iq0 = i0 + 1j * q0
sdr.tx([iq0,iq0])  # Send Tx data.

# Assign frequency bins and "zoom in" to the fc0 signal on those frequency bins
xf = np.fft.fftfreq(NumSamples, ts)
xf = np.fft.fftshift(xf)/1e6
signal_start = 0
signal_end = int(len(xf)-1)


def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

def calcTheta(delay_type, delay, freq):
    # calculates the steering angle for a given time or phase delay
    # if delay_type = "phase", then delay must be in degrees
    # if delay_type = "time", then delay must be in ps
    if delay_type == "phase":
        # steering angle is theta = arcsin(c*deltaphase/(2*pi*f*d)
        phase_rad = np.deg2rad(delay)
        arcsin_arg = phase_rad*3E8/(2*np.pi*freq*d)
    else:
        # steering angle is theta = arcsin(time_delay * c / d)        
        arcsin_arg = delay * 1E-12 * c / d
    arcsin_arg = max(min(1, arcsin_arg), -1)     # arcsin argument must be between 1 and -1, or numpy will throw a warning
    calc_theta = np.rad2deg(np.arcsin(arcsin_arg))
    return calc_theta


def time_delayer(data, do_phase_delay, do_time_delay, delay_time_ps, freq, samp_rate):
    delayed_data = data
    if do_phase_delay == True:
        delayed_data = delayed_data * np.exp(1j*2*np.pi*freq*delay_time_ps*1E-12)
    if do_time_delay == True:
        # Create and apply fractional delay filter
        # Dr. Marc Lichtman  https://pysdr.org/content/sync.html#adding-a-delay
        delay = np.int64(samp_rate) * delay_time_ps * 1E-12   # fractional delay, in samples
        N = 21 # number of taps
        n = np.arange(-N//2, N//2) # ...-3,-2,-1,0,1,2,3...
        h = np.sinc(n - delay) # calc filter taps
        h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
        h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
        delayed_data = np.convolve(delayed_data, h) # apply filter
        delayed_data = delayed_data[int(N/2+1):-int(N/2-1)]    # drop the first and last N/2 results
    return delayed_data


'''Collect Data'''
for i in range(20):  
    # let Pluto run for a bit, to do all its calibrations, then get a buffer
    data = sdr.rx()

delay_times = np.arange(-time_max, time_max, time_max/200)    # time delay in ps
steer_angles = []
for time_delay in delay_times:
    steer_angles.append(calcTheta("time", time_delay, f_carrier))

for i in range(num_scans):
    data = sdr.rx()
    Rx_0=data[0]
    Rx_1=data[1]        
    peak_sum = []
    peak_sum_time = []
    for time_delay in delay_times:   
        delayed_Rx_1 = time_delayer(Rx_1, True, False, time_delay+time_cal, f_carrier, samp_rate)  #  data, do_phase_delay, do_time_delay, delay_time_ps, freq, samp_rate
        delayed_sum = dbfs(Rx_0 + delayed_Rx_1 * np.sign(invert_Rx))
        peak_sum.append(np.max(delayed_sum[signal_start:signal_end]))  # this is the data for if we are only doing phase shifting
        
        delayed_Rx_1 = time_delayer(Rx_1, True, True, time_delay+time_cal, f_carrier, samp_rate)  #  data, do_phase_delay, do_time_delay, delay_time_ps, freq, samp_rate
        delayed_sum = dbfs(Rx_0 + delayed_Rx_1 * np.sign(invert_Rx))
        peak_sum_time.append(np.max(delayed_sum[signal_start:signal_end]))   # this is the data for if we are only doing time shifting
        
    peak_dbfs = np.max(peak_sum)
    peak_delay_index = np.where(peak_sum==peak_dbfs)
    peak_delay = delay_times[peak_delay_index[0][0]]
    peak_dbfs_time = np.max(peak_sum_time)
    peak_delay_time_index = np.where(peak_sum_time==peak_dbfs_time)
    peak_delay_time = delay_times[peak_delay_time_index[0][0]]
    steer_angle = int(calcTheta("time", peak_delay, f_carrier))
    steer_angle_time = int(calcTheta("time", peak_delay_time, f_carrier))
    
    plt.plot(steer_angles, peak_sum)
    plt.plot(steer_angles, peak_sum_time)
    plt.axvline(x=steer_angle, color='b', linestyle=':')
    plt.axvline(x=steer_angle_time, color='orange', linestyle=':')
    plt.ylim(top=0, bottom=-40)
    plt.xlabel("steering angle [deg]")
    plt.ylabel("Rx0 + Rx1 [dBfs]")
    plt.draw()
    plt.show()

sdr.tx_destroy_buffer()
if i>40: print('\a')    # for a long capture, beep when the script is done


