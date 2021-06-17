import numpy
import scipy.io.wavfile
import librosa.display
from matplotlib import pyplot as plt

sample_rate,signal=scipy.io.wavfile.read('airport-barcelona-0-0-a.wav')
signal=signal[:,0]
signal=signal.astype(float)
print(sample_rate,len(signal))

#预加重
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

#分帧
frame_size=0.04
frame_stride=0.02
def audio2frame(signal,frame_size,frame_stride):
    '''将音频信号转化为帧。
	参数含义：
	signal:原始音频型号
	frame_length:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
	frame_step:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    frame_length=int(round(frame_size*sample_rate)) #一帧帧时间长度
    frame_step=int(round(frame_stride*sample_rate)) #相邻帧之间的步长
    frames_num=int(numpy.ceil((1.0*signal_length-frame_length)/frame_step))+2
    pad_signal_length = (frames_num-1) * frame_step + frame_length#所有帧加起来总的铺平后的长度
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal=numpy.append(signal,z)
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T  #相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)] #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    return frames,frame_length,indices  #返回帧信号矩阵
frames,frame_length,indices=audio2frame(signal,frame_size,frame_stride)

#加汉明窗
frames *= numpy.hamming(frame_length)

#傅立叶变换和功率谱
NFFT = 2048
mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

low_freq_mel = 0
#将频率转换为Mel
nfilt = 40
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))   # 将Hz转换为Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 使得Mel scale间距相等
hz_points = (700 * (10**(mel_points / 2595) - 1))  # 将Mel转换为Hz

bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # 数值稳定性
filter_banks = 20 * numpy.log10(filter_banks)  # dB
filter_banks = filter_banks.T
#绘图
fig, ax = plt.subplots()
img = librosa.display.specshow(filter_banks, x_axis='time',
                         y_axis='mel', sr=sample_rate,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')