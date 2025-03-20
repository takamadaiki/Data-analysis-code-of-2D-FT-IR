#Function of fft etc. 
#input xspec => 1D array, yspec => 1D array,pix_x,pix_y => int
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy import interpolate
from tqdm import tqdm

def func_FFT(yspec,pix_x,pix_y,N2=4096,pix_size=12*1e3,f=25*1e6,Dps=1.005661e6,phiy=44.874,center_x=320,center_y=256,wav1=5000,wav2=15000):
    #parameter setting
    N1 = len(yspec)
    xspec = np.arange(0,N1,1)
    M = Dps/N1#nm (位相シフト量1005 um)
    phi_y = phiy*np.pi/180.0 #(設置角 44.874 deg. => result of BPF calibration experiment)
    Wi = np.hamming(N1) #Hamming window function

    #Slope correction
    Hbar1 = np.poly1d(np.polyfit(xspec, yspec,1))(xspec)
    y2 = yspec-Hbar1
    #Center burst
    xmax = np.argmax(abs(y2))
    a_roll = np.roll(y2,int(N1/2-1)-xmax)
    #DFT conversion
    yf = fft(a_roll*Wi,n=N2) #n => zero-filling
    yspec_sam = np.abs(yf[1:int(N2/2)]) #>0

    ####Gakaku hosei#####
    a,b = -(pix_x-center_x)*pix_size, (pix_y-center_y)*pix_size
    theta_x = np.arctan(b/f)
    theta_y = np.arctan(a/f)
    theta_dash_y = theta_y+np.pi/2.0
    L = 2*M*np.cos(phi_y-theta_dash_y)/np.cos(theta_x) #nm
    T = L*1e-7 #cm
    freq = fftfreq(N2,T)
    Arr_w = 10000*np.ones(len(yspec_sam))/freq[1:int(N2/2)]

    ###Interpolate the data###
    yw = np.arange(wav1,wav2,10)#arrary of x(wavelength)
    fx1 = interpolate.interp1d(1000*Arr_w,yspec_sam)
    return fx1(yw)

def FFTimage(Zarr,xsta=0,xend=700,ysta=0,yend=700,N2=4096,pix_size=12*1e3,f=25*1e6,Dps=1.005661e6,phiy=44.874,center_x=320,center_y=256\
            ,wav1=5000,wav2=15000):
    #Declation of array
    nx0, ny0 = Zarr.shape[2], Zarr.shape[1]
    yw = np.arange(wav1,wav2,10)#arrary of x(wavelength)   
    #array of spectral
    Zs_sam = np.zeros((len(yw),ny0,nx0))
    for i in tqdm(range(ny0)):
        for j in range(nx0):
            if j>=xsta and j<=xend and i>=ysta and i<=yend:
                pix_x, pix_y = j,i #pixel position
                yspec = Zarr[:,pix_y,pix_x]
                Zs_sam[:,i,j] = func_FFT(yspec,pix_x,pix_y,N2=N2,pix_size=pix_size,f=f,Dps=Dps,phiy=phiy,\
                                         center_x=center_x,center_y=center_y,wav1=wav1,wav2=wav2)
            else:
                continue
              
    return Zs_sam,yw

