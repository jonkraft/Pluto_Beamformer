'''2 Element Adaptive Digital Beamforming Demonstration'''

# %%
# Imports
import sys
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
import adi

'''Key Parameters'''
sample_rate = 5e6
center_freq = 2.1e9
signal_freq = 100e3
rx_gain = 3     # must be between -3 and 70
tx_gain = -3
phase_offset = -0.3  # calibration between the two Rx channels in rad
c = 299792458  # speed of light in m/s
d = c/(2*center_freq) # distance between Rx antennas in meters. Set d to c/(2*f)
use_transmitter = True
Nr = 2  # number of receive channels

# %%
""" Program the basic hardware settings
"""
# Instantiate all the Devices
sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
my_sdr = adi.ad9361(uri=sdr_ip)
print("Pluto is connected!")

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
sample_rate = int(my_sdr.sample_rate)
my_sdr.rx_lo = int(center_freq)
my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 and Rx2
my_sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(rx_gain)  # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(rx_gain)  # must be between -3 and 70

# Configure SDR Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True
my_sdr.tx_hardwaregain_chan1 = -88  # set to -88 to disable tx0

N = int(2**18)
fc = int(signal_freq)
ts = 1 / float(sample_rate)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 1 * (i + 1j * q)

if use_transmitter == True:
    my_sdr.tx_hardwaregain_chan0 = int(tx_gain)  # must be between 0 and -88
    my_sdr.tx([iq, iq])  # transmit data from Pluto
else:
    my_sdr.tx_hardwaregain_chan0 = int(-88)  # must be between 0 and -88


def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

def w_mvdr(theta, data):
    # Calculate MVDR weights while SOI is at theta
    s = np.exp(-2j * np.pi * d * center_freq/c * np.arange(Nr) * np.sin(theta)) # steering vector in direction theta
    s = s.reshape(-1,1) # make into a column vector  
    R = data @ data.conj().T/len(data) # Calc covariance matrix.  Gives a Nr x Nr matrix
    Rinv = np.linalg.pinv(R) # pseudo-inverse tends to work better than a true inverse
    w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR equation
    return w.squeeze()

# %%
""" MVDR DOA
"""
plt.ion() # needed for realtime view
print("Starting, use control-c to stop")
try:
    while True:
        data = my_sdr.rx() # receive a batch of samples, it's a list of lists
        x = np.array(data) # shape is 2 x rx_buffer_size

        # DOA routine        
        theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 100) # between -90 and +90 degrees
        doa_conv = []
        doa_mvdr = []
        for theta_i in theta_scan:
            theta_i = theta_i + phase_offset
            s = np.exp(-2j * np.pi * d * center_freq/c * np.arange(Nr) * np.sin(theta_i)) # steering vector in the desired direction theta_i
            s = s.reshape(-1,1) # make into a column vector
            
            # Conventional DOA
            w = s
            y = w.conj().T @ x
            max_signal = np.max(dbfs(y))
            doa_conv.append(max_signal)
            
            # MVDR DOA
            w = w_mvdr(theta_i, x)
            y = w.conj().T @ x
            max_signal = np.max(dbfs(y))
            doa_mvdr.append(max_signal)
            
        doa_conv -= np.max(doa_conv) # normalize so peak is at 0 dB
        doa_mvdr -= np.max(doa_mvdr) # normalize so peak is at 0 dB

        # Plot the results real time
        plt.plot(theta_scan*180/np.pi, doa_conv, '.-')
        plt.plot(theta_scan*180/np.pi, doa_mvdr, '.-')
        plt.legend(['Conventional', 'MVDR'], loc='lower left', fontsize='8')
        plt.title("DOA Plot for 2 Element Array")
        plt.xlabel("Direction of Arrival [degrees]")
        plt.ylabel("Magnitude [dB]")
        plt.ylim(top=0, bottom=-20)
        plt.xlim(left=-90, right=90)
        plt.draw()
        plt.pause(0.3)
        plt.clf()

except KeyboardInterrupt:
    my_sdr.tx_destroy_buffer()
    print("Pluto Buffer Cleared!")
    sys.exit() # quit python




 
