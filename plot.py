from settings_FSMLL import *

if (len(sys.argv) < 5):
    print("Usage: python simulation.py M P_pump FSR dirname [debug]")
    exit(1)

#! Lasing parameters of Erbium
N=3.959e26 # Er离子浓度: m^-3
sigma_sa=4.03e-25 # signal的吸收截面: m^2
sigma_se=6.76e-25 # signal的发射截面: m^2
sigma_pa=4.48e-25 # pump的吸收截面: m^2
sigma_pe=1.07e-25 # pump的发射截面: m^2
tau_g=10e-3 # 上能级寿命: s
lambda_s=1570e-9 # signal波长: m
lambda_p=1480e-9 # pump波长: m
nu_s=c0/lambda_s
nu_p=c0/lambda_p
omega_p=2*np.pi*nu_p
omega_s=2*np.pi*nu_s
A_s=0.9e-12 # signal有效模面积: m^2
A_p=0.9e-12 # pump有效模面积: m^2
T=300 # 温度: K
Gamma_s=0.9 # signal与Er离子的模斑交叠系数
Gamma_p=0.9 # pump与Er离子的模斑交叠系数
b_pa=h*nu_p/tau_g/sigma_pa
b_pe=h*nu_p/tau_g/sigma_pe
b_sa=h*nu_s/tau_g/sigma_sa
b_se=h*nu_s/tau_g/sigma_se
# P_sat=h*nu_s*A_s/tau_g/(sigma_sa+sigma_se) # 饱和功率，即增益降至小信号增益一半时的信号光功率: W
eta=sigma_se/sigma_sa
beta=np.exp(-1.0/k_B/T*h*c0*(1.0/lambda_p-1.0/lambda_s))

#! Micro-cavity parameters
beta2=-58e-27 # 色散: s^2/m
n2=1.8e-19 # LiNbO3的Kerr系数: m^2/W
FSR = float(sys.argv[3])
L_d=5e-3 * 25e9 / FSR # 腔长: m
T_R=1.0/FSR # roundtrip time: s
omega_m=2*np.pi*FSR
Omega_g=2*np.pi*3e8/(1570e-9)**2*85e-9 # 增益的半高半宽: rad
Q_ins=2e6 # 腔的本征Q
Q_exs=2e6 # 腔的耦合Q
Q_tots=1.0/(1.0/Q_ins+1.0/Q_exs) # 腔的总Q
Q_tots = 0.3e6
Q_inp=2e6 # 腔的本征Q, pump处
Q_exp=2e6
Q_totp=1.0/(1.0/Q_inp+1.0/Q_exp)
Q_totp = 0.2e6
gamma=0 # 电光梳在波导和谐振腔耦合处的损耗
k=omega_p/omega_m/Q_exp*2*np.pi # 电光梳在波导和谐振腔耦合处的耦合效率
# M=0.1 # 电光调制深度
# M = float(input("Type in the depth of modulation: "))
M = float(sys.argv[1])
phi_opt=0.0 # pump光的失谐
t=np.linspace(-T_R/2,T_R/2-T_R/1023,1024)
delta_t = t[-1]-t[-2]
q=np.linspace(-512,511,1024)
x=np.linspace(-np.pi,np.pi-np.pi*2/1023,1024)
total_loss=omega_p/omega_m/Q_totp*2*np.pi
D=-0.5*beta2*L_d
delta=n2*omega_s*L_d/(c0*A_s) # Kerr效应的系数δ
l=-0.5*np.log(1-2*np.pi*omega_s/Q_tots/omega_m) # Haus方程中的loss,包含signal光由于本征/耦合的损耗
l_p_in=np.exp(-2*np.pi*omega_p/omega_m/Q_inp/2)
P_pump = float(sys.argv[2])
prompt = "P_pump=" + str(P_pump*1000) + "mW" + ",M=" + str(M) + ",FSR=" + str(FSR/1e9) + "GHz"


# 切换到文件所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 改变输出位置
os.chdir("../outputs")
directory_name = sys.argv[4]
os.chdir(directory_name)
sys.stdout = open(prompt + ".txt", 'a')
print(prompt)

# 时间normalize到T_R
scale=1 # 每保存一次运行scale个roundtrip time
steps = 100
dT = 1/steps
save_round1=25000 # 一共保存save_round个中间结果
save_round2=300000
begin_to_save=00000


A_save = np.load(prompt + "_A_save.npy")
E_p_save = np.load(prompt + "_E_p_save.npy")
g_save = np.load(prompt + "_g_save.npy")
T=np.array(range(begin_to_save,int(save_round2)))*scale
T_G=np.array(range(int(save_round2)))*scale
x,y=np.meshgrid(T,t)

time_domain_p=[]
for i in range(-1-1024*4,-1):
    time_domain_p+=list(E_p_save[:,i])
time_domain_p=np.array(time_domain_p)*np.sqrt(2*np.pi*omega_p/Q_exp/omega_m)*1.0j+np.sqrt(1-2*np.pi*omega_p/Q_exp/omega_m)*np.sqrt(P_pump)
spectrum_p=fftshift(fft(time_domain_p*delta_t))
spectrum_p_log=10*np.log10(np.abs(spectrum_p/T_R/(len(time_domain_p)/1024))**2/1e-3)
freq_list_p=np.linspace(c0/lambda_p-512*FSR,c0/lambda_p+512*FSR,len(time_domain_p))
lamb_list_p=c0/freq_list_p


t_center_p=len(time_domain_p)//2
t_range_p=len(time_domain_p)//4096
plt.figure("Output EO comb pulse train",figsize=(9,3),dpi=100)
plt.plot((np.array(range(len(time_domain_p)))*T_R*1e9/1024)[t_center_p-t_range_p:t_center_p+t_range_p],(1e3*np.abs(time_domain_p)**2)[t_center_p-t_range_p:t_center_p+t_range_p],color="red")
plt.xlabel("time (ns)")
plt.ylabel("power (mW)")
plt.savefig(prompt + "_pump" + ".png",dpi=300,bbox_inches="tight",transparent=True)
# plt.show()


k_center_p=len(spectrum_p)//2
k_range_p=len(spectrum_p)//10
plt.figure("Spectrum of pump",figsize=(9,3),dpi=100)
plt.plot(1e9*lamb_list_p[k_center_p-k_range_p:k_center_p+k_range_p],spectrum_p_log[k_center_p-k_range_p:k_center_p+k_range_p],color="blue")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (dBm)")
plt.savefig(prompt + "_pump_spectrum" + ".png",dpi=300,bbox_inches="tight",transparent=True)
# plt.show()
print("max spectrum_p_log =", max(spectrum_p_log))


time_domain=[]
for i in range(-1-1024*8,-1):
    time_domain+=list(A_save[:,i])
time_domain=np.array(time_domain)*np.sqrt(2*np.pi*omega_s/Q_tots/omega_m)
spectrum=fftshift(fft(time_domain*delta_t))
spectrum_log=10*np.log10(np.abs(spectrum/T_R/(len(time_domain)/1024))**2/1e-3)
freq_list=np.linspace(c0/lambda_s-512*FSR,c0/lambda_s+512*FSR,len(time_domain))
lamb_list=c0/freq_list


t_center=len(time_domain)//2
t_range=len(time_domain)//2048
plt.figure("Pulse Train of signal",figsize=(9,3),dpi=100)
plt.plot((np.array(range(len(time_domain)))*T_R*1e9/1024)[t_center-t_range:t_center+t_range],(1e3*np.abs(time_domain)**2)[t_center-t_range:t_center+t_range],color="red")
plt.xlabel("time (ns)")
plt.ylabel("power (mW)")
plt.savefig(prompt + "_signal" + ".png",dpi=300,bbox_inches="tight",transparent=True)
# plt.show()


k_center=len(spectrum)//2
k_range=len(spectrum)//4

plt.figure("Spectrum of signal",figsize=(9,3),dpi=100)
plt.plot(1e9*lamb_list[k_center-k_range:k_center+k_range],spectrum_log[k_center-k_range:k_center+k_range],color="blue")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (dBm)")
plt.savefig(prompt + "_signal_spectrum" + ".png",dpi=600,bbox_inches="tight",transparent=True)
# plt.show()
print(max(spectrum_log))


signal_average_power=np.sum(abs(A_save[:,-1])**2)/T_R*delta_t
EO_comb_average_power=np.sum(abs(E_p_save[:,-1])**2)/T_R*delta_t
# print(signal_average_power)
# print(EO_comb_average_power)
print("EO comb peak power = "+str(np.max(abs(E_p_save[:,-1])**2)*1e3)+" mW")
print("EO comb average power = "+str(EO_comb_average_power*1e3)+" mW")


print("从腔内输出的signal pulse: ")

print("时域上单个pulse峰值功率(mW): ",end="")
print(np.max(np.abs(time_domain)**2)*1e3)
print("时域上单个pulse平均能量(J): ",end="")
print(np.sum(np.abs(time_domain)**2)*delta_t/(len(time_domain)/1024))
print("时域上单个pulse平均功率(mW): ",end="")
print(np.sum(np.abs(time_domain)**2)*delta_t/(len(time_domain)/1024)/T_R*1e3)
# print("频域上单个pulse平均能量(J): ",end="")
# print(np.sum(np.abs(spectrum)**2)*(freq_list[-1]-freq_list[-2])/(len(time_domain)/1024))
print("相对于泵浦光的转换效率(%): ",end="")
print(np.sum(np.abs(time_domain)**2)*delta_t/(len(time_domain)/1024)/T_R/P_pump*1e2)



# 本次运行的相关信息
print("pump光(功率)的耦合损耗: k= "+str(k))
print("pump光(功率)的本征损耗: α= "+str(2*np.pi*omega_p/Q_inp/omega_m))
print("pump光(功率)的总损耗: l= "+str(total_loss))
print("signal光(功率)的耦合损耗: k'= "+str(2*np.pi*omega_s/Q_exs/omega_m))
print("signal光(功率)的本征损耗: α'= "+str(2*np.pi*omega_s/Q_ins/omega_m))
print("signal光(功率)的总损耗: l'= "+str(2*np.pi*omega_s/Q_tots/omega_m))


plt.figure("Gain",figsize=(14,7),dpi=100)
plt.plot(T_G[::],g_save[::],color="red",label="Gain")
plt.plot(T_G[::],[l for i in range(0, save_round2)],color="blue",label="Loss")
plt.legend()
plt.xlabel("Roundtrip Time")
plt.ylabel("Gain")
# plt.show()
plt.savefig(prompt + "_gain" + ".png",dpi=300,bbox_inches="tight",transparent=True)
plt.cla()
plt.close()


# plt.figure("Time Evolution",figsize=(10,4),dpi=100)
# plt.contourf(x,y*1e12,1000*np.abs(A_save)**2,100,cmap=cm.jet)
# plt.xlabel("Roundtrip")
# plt.ylabel("t (ps)")
# plt.title("Intra-cavity signal Evolution (mW)")
# plt.colorbar()
# plt.savefig(prompt + "_signal_evolution" + ".png",dpi=300,transparent=True,bbox_inches="tight")
# # plt.show()
# plt.cla()
# plt.close()


# plt.figure("Time Evolution2",figsize=(10,4),dpi=100)
# plt.contourf(x,y*1e12,1000*np.abs(E_p_save)**2,100,cmap=cm.jet)
# plt.xlabel("Roundtrip")
# plt.ylabel("t (ps)")
# plt.title("Intra-cavity pump Evolution (mW)")
# plt.colorbar()
# plt.savefig(prompt + "_pump_evolution" + ".png",dpi=300,transparent=True,bbox_inches="tight")
# # plt.show()
# plt.cla()
# plt.close()