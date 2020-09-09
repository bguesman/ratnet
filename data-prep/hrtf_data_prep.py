from scipy.signal import convolve
import numpy as np
import wavio
from scipy.io import wavfile
import os

def apply_conv(signal, impulse_response, out_path):
	# prep impulse response
	ir_obj = wavio.read(impulse_response)
	ir_bitdepth = float(2**(ir_obj.sampwidth*8 - 1))
	impulse = ir_obj.data

	#normalize
	impulse = impulse.astype(np.float32, order='C')/ir_bitdepth

	output = np.array([convolve(signal[:,0], impulse[:,0]), convolve(signal[:,1], impulse[:,1])]).astype(np.float32, order='C')
	# Scale to int16 range.
	output = output * 32768.0
	output = output.astype(np.int16, order='C').transpose()
	wavfile.write(out_path, 44100, output)

def process_ircam(audio_file,ircam_folder, outpath):
	# open audiofile and prep the signal
	af_obj = wavio.read(audio_file)
	af_bitdepth = float(2**(af_obj.sampwidth*8 - 1))
	af_data = np.tile(af_obj.data, (1,2))
	signal = af_data.astype(np.float32, order='C')/af_bitdepth

	for processed_path in os.listdir(ircam_folder):
		fullpath = os.path.join(ircam_folder, processed_path)
		pp = processed_path[:-4] # get rid of .wav
		pathsplit = pp.split('_')
		filename = pathsplit[0] + pathsplit[1] + "_azimuth=" + pathsplit[-2][1:] + "_elevation=" + pathsplit[-1][1:]
		print("Processing " + filename)
		out_path = outpath + filename + ".wav"
		apply_conv(signal, fullpath, out_path)


process_ircam('/Users/maxmines/Documents/DSP/Data-Generation/irmas_clean.wav',
'/Users/maxmines/Downloads/IRC_1002/COMPENSATED/WAV/IRC_1002_C','/Users/maxmines/Documents/DSP/AIGear/dld/train/')